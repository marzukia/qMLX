# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API server for vllm-mlx with continuous batching.

This module provides a FastAPI server with vLLM-style continuous batching
for efficient handling of multiple concurrent requests.

Usage:
    # Start server with batching
    python -m vllm_mlx.server_v2 --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Test with curl
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "test", "messages": [{"role": "user", "content": "Hello!"}]}'
"""

import argparse
import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .engine import AsyncEngineCore, EngineConfig
from .request import RequestOutput, SamplingParams
from .scheduler import SchedulerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[AsyncEngineCore] = None
_model_name: Optional[str] = None
_is_mllm: bool = False
_default_max_tokens: int = 32768


# MLLM model detection patterns
MLLM_PATTERNS = [
    "-VL-", "-VL/", "VL-",
    "llava", "LLaVA",
    "idefics", "Idefics",
    "paligemma", "PaliGemma",
    "pixtral", "Pixtral",
    "molmo", "Molmo",
    "phi3-vision", "phi-3-vision",
    "cogvlm", "CogVLM",
    "internvl", "InternVL",
    "deepseek-vl", "DeepSeek-VL",
]


def is_mllm_model(model_name: str) -> bool:
    """Check if model name indicates a multimodal model."""
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in MLLM_PATTERNS)


# Special tokens to remove from output (keeps <think> blocks)
SPECIAL_TOKENS_PATTERN = re.compile(
    r'<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|'
    r'<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|'
    r'</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]'
)


def clean_output_text(text: str) -> str:
    """
    Clean model output by removing special tokens.
    Keeps <think>...</think> blocks intact.

    Args:
        text: Raw model output

    Returns:
        Cleaned text (with thinking preserved)
    """
    if not text:
        return text

    # Remove special tokens only (not thinking blocks)
    text = SPECIAL_TOKENS_PATTERN.sub('', text)

    # Clean up whitespace
    text = text.strip()

    return text


# ============================================================================
# Pydantic Models (OpenAI API compatible)
# ============================================================================

class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = None


class VideoUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Union[ImageUrl, dict, str]] = None
    video: Optional[str] = None
    video_url: Optional[Union[VideoUrl, dict, str]] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart], List[dict]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None  # Uses _default_max_tokens if not set
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None  # Uses _default_max_tokens if not set
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: Optional[str] = "stop"


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm-mlx"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Helper Functions
# ============================================================================

def extract_text_from_messages(messages: List[Message]) -> str:
    """Extract text content from messages and apply chat template."""
    processed = []
    for msg in messages:
        if isinstance(msg.content, str):
            processed.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            text_parts = []
            for item in msg.content:
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                elif hasattr(item, 'dict'):
                    item = item.dict()
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            processed.append({"role": msg.role, "content": "\n".join(text_parts)})
    return processed


def create_sampling_params(
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 32768,
    stop: Optional[List[str]] = None,
) -> SamplingParams:
    """Create SamplingParams from request parameters."""
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop or [],
    )


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage engine lifecycle."""
    global _engine
    if _engine is not None:
        _engine.start()
    yield
    if _engine is not None:
        await _engine.stop()


app = FastAPI(
    title="vllm-mlx API (v2 with Continuous Batching)",
    description="OpenAI-compatible API with vLLM-style continuous batching for MLX",
    version="0.2.0",
    lifespan=lifespan,
)


def get_engine() -> AsyncEngineCore:
    """Get the engine instance."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    engine = get_engine()
    stats = engine.get_stats()
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": _model_name,
        "model_type": "mllm" if _is_mllm else "llm",
        "engine_stats": stats,
    }


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models."""
    models = []
    if _model_name:
        models.append(ModelInfo(id=_model_name))
    return ModelsResponse(data=models)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    engine = get_engine()

    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    sampling_params = create_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or _default_max_tokens,
        stop=request.stop,
    )

    if request.stream:
        return StreamingResponse(
            stream_completion(engine, prompts[0], sampling_params, request.model),
            media_type="text/event-stream",
        )

    # Non-streaming - use optimized generate() with concurrent requests
    async def generate_one(i: int, prompt: str):
        output = await engine.generate(prompt=prompt, sampling_params=sampling_params)
        return i, output

    results = await asyncio.gather(*[generate_one(i, p) for i, p in enumerate(prompts)])

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, output in sorted(results, key=lambda x: x[0]):
        choices.append(CompletionChoice(
            index=i,
            text=clean_output_text(output.output_text),
            finish_reason=output.finish_reason,
        ))
        total_prompt_tokens += output.prompt_tokens
        total_completion_tokens += output.completion_tokens

    return CompletionResponse(
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion with continuous batching."""
    engine = get_engine()

    # Extract messages and build prompt
    messages = extract_text_from_messages(request.messages)

    # Apply chat template
    tokenizer = engine.engine.tokenizer
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt += "\nassistant:"

    sampling_params = create_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or _default_max_tokens,
        stop=request.stop,
    )

    if request.stream:
        return StreamingResponse(
            stream_chat_completion(engine, prompt, sampling_params, request.model),
            media_type="text/event-stream",
        )

    # Non-streaming - use optimized generate() method
    final_output = await engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
    )

    return ChatCompletionResponse(
        model=request.model,
        choices=[ChatCompletionChoice(
            message=Message(role="assistant", content=clean_output_text(final_output.output_text)),
            finish_reason=final_output.finish_reason,
        )],
        usage=Usage(
            prompt_tokens=final_output.prompt_tokens,
            completion_tokens=final_output.completion_tokens,
            total_tokens=final_output.prompt_tokens + final_output.completion_tokens,
        ),
    )


async def stream_completion(
    engine: AsyncEngineCore,
    prompt: str,
    sampling_params: SamplingParams,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream completion response with cancellation support."""
    request_id = await engine.add_request(
        prompt=prompt,
        sampling_params=sampling_params,
    )

    try:
        async for output in engine.stream_outputs(request_id):
            data = {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "text": clean_output_text(output.new_text),
                    "finish_reason": output.finish_reason if output.finished else None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        # Client disconnected - abort the request
        logger.info(f"Client disconnected, aborting request {request_id}")
        await engine.abort_request(request_id)
        raise
    except GeneratorExit:
        # Generator was garbage collected
        logger.info(f"Generator exit, aborting request {request_id}")
        await engine.abort_request(request_id)
        raise


async def stream_chat_completion(
    engine: AsyncEngineCore,
    prompt: str,
    sampling_params: SamplingParams,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream chat completion response with cancellation support."""
    request_id = await engine.add_request(
        prompt=prompt,
        sampling_params=sampling_params,
    )

    try:
        async for output in engine.stream_outputs(request_id):
            data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": clean_output_text(output.new_text)} if output.new_text else {},
                    "finish_reason": output.finish_reason if output.finished else None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        # Client disconnected - abort the request
        logger.info(f"Client disconnected, aborting request {request_id}")
        await engine.abort_request(request_id)
        raise
    except GeneratorExit:
        # Generator was garbage collected
        logger.info(f"Generator exit, aborting request {request_id}")
        await engine.abort_request(request_id)
        raise


@app.get("/v1/engine/stats")
async def get_engine_stats():
    """Get engine statistics (non-standard endpoint for monitoring)."""
    engine = get_engine()
    return engine.get_stats()


@app.get("/v1/cache/stats")
async def get_cache_stats():
    """Get prefix cache statistics.

    Returns cache hit/miss rates, tokens saved, and eviction counts.
    This is useful for monitoring cache effectiveness.
    """
    engine = get_engine()
    stats = engine.get_cache_stats()
    if stats is None:
        return {
            "enabled": False,
            "message": "Prefix caching is not enabled",
        }
    return {
        "enabled": True,
        **stats,
    }


# ============================================================================
# Model Loading
# ============================================================================

def load_model(
    model_name: str,
    scheduler_config: Optional[SchedulerConfig] = None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
) -> None:
    """Load a model and initialize the engine."""
    global _engine, _model_name, _is_mllm, _default_max_tokens

    _default_max_tokens = max_tokens

    logger.info(f"Loading model: {model_name}")

    _is_mllm = is_mllm_model(model_name)

    if _is_mllm:
        # Load MLLM model
        from mlx_vlm import load as load_vlm
        model, processor = load_vlm(model_name)
        tokenizer = processor
    else:
        # Load LLM model
        from mlx_lm import load

        # Qwen3 fix: eos_token changed from <|im_end|> to <|endoftext|>
        # but chat template still uses <|im_end|>, so we need to set it explicitly
        # See: https://kaitchup.substack.com/p/qwen3-when-im_end-suddenly-becomes
        tokenizer_config = None
        if "qwen3" in model_name.lower() or "Qwen3" in model_name:
            tokenizer_config = {"eos_token": "<|im_end|>"}
            logger.info("Qwen3 detected: setting eos_token to <|im_end|>")

        model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)

    # Create engine config with stream_interval for low-latency streaming
    engine_config = EngineConfig(
        model_name=model_name,
        scheduler_config=scheduler_config or SchedulerConfig(),
        stream_interval=stream_interval,
    )

    # Create engine
    _engine = AsyncEngineCore(model, tokenizer, engine_config)
    _model_name = model_name

    logger.info(f"Model loaded: {model_name} ({'MLLM' if _is_mllm else 'LLM'})")
    logger.info(f"Stream interval: {stream_interval} tokens")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        description="vllm-mlx OpenAI-compatible server with continuous batching",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences",
    )
    parser.add_argument(
        "--prefill-batch-size",
        type=int,
        default=8,
        help="Prefill batch size",
    )
    parser.add_argument(
        "--completion-batch-size",
        type=int,
        default=32,
        help="Completion batch size",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=more throughput)",
    )

    args = parser.parse_args()

    # Create scheduler config
    scheduler_config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        prefill_batch_size=args.prefill_batch_size,
        completion_batch_size=args.completion_batch_size,
    )

    # Load model with stream_interval
    load_model(args.model, scheduler_config, stream_interval=args.stream_interval)

    # Start server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
