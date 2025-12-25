# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API server for vllm-mlx.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for LLM and MLLM (Multimodal Language Model) inference using MLX on Apple Silicon.

Supports:
- Text-only LLM inference (mlx-lm)
- Multimodal MLLM inference with images and video (mlx-vlm)
- OpenAI-compatible chat/completions API
- Streaming responses

Usage:
    # LLM server
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # MLLM server (auto-detected)
    python -m vllm_mlx.server --model mlx-community/Qwen2-VL-2B-Instruct-4bit

The server provides:
    - POST /v1/completions - Text completions
    - POST /v1/chat/completions - Chat completions (with multimodal support)
    - GET /v1/models - List available models
    - GET /health - Health check
"""

import argparse
import asyncio
import logging
import re
import time
import uuid
from typing import AsyncIterator, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="vllm-mlx API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.1.0",
)

# Global model instance
_model = None
_model_name = None
_is_mllm = False  # Track if model is a multimodal language model
_default_max_tokens: int = 32768


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
    """
    if not text:
        return text
    text = SPECIAL_TOKENS_PATTERN.sub('', text)
    return text.strip()


# MLLM model detection patterns (auto-detects multimodal language models)
MLLM_PATTERNS = [
    "-VL-", "-VL/", "VL-",  # Qwen-VL, Qwen2-VL, Qwen3-VL, etc.
    "llava", "LLaVA",       # LLaVA models
    "idefics", "Idefics",   # Idefics models
    "paligemma", "PaliGemma",  # PaliGemma
    "pixtral", "Pixtral",   # Pixtral
    "molmo", "Molmo",       # Molmo
    "phi3-vision", "phi-3-vision",  # Phi-3 Vision
    "cogvlm", "CogVLM",     # CogVLM
    "internvl", "InternVL",  # InternVL
    "deepseek-vl", "DeepSeek-VL",  # DeepSeek-VL
]


def is_mllm_model(model_name: str) -> bool:
    """Check if model name indicates a multimodal language model."""
    model_lower = model_name.lower()
    for pattern in MLLM_PATTERNS:
        if pattern.lower() in model_lower:
            return True
    return False


# Backwards compatibility alias
is_vlm_model = is_mllm_model


# Request/Response models - OpenAI compatible multimodal format
class ImageUrl(BaseModel):
    url: str
    detail: str | None = None


class VideoUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str  # "text", "image_url", "video", "video_url"
    text: str | None = None
    image_url: ImageUrl | dict | str | None = None
    video: str | None = None
    video_url: VideoUrl | dict | str | None = None  # OpenAI-style video URL


class Message(BaseModel):
    role: str
    content: Union[str, list[ContentPart], list[dict]]  # Support text or multimodal


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None  # Uses _default_max_tokens if not set
    stream: bool = False
    stop: list[str] | None = None
    # MLLM-specific parameters
    video_fps: float | None = None  # FPS for video frame extraction
    video_max_frames: int | None = None  # Max frames from video


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None  # Uses _default_max_tokens if not set
    stream: bool = False
    stop: list[str] | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str | None = "stop"


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm-mlx"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


def get_model():
    """Get the loaded model, loading if necessary."""
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model


def load_model(model_name: str, force_mllm: bool = False, max_tokens: int = 32768):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        force_mllm: Force loading as MLLM even if not auto-detected
        max_tokens: Default max tokens for generation
    """
    global _model, _model_name, _is_mllm, _default_max_tokens

    _default_max_tokens = max_tokens

    # Auto-detect MLLM models
    _is_mllm = force_mllm or is_mllm_model(model_name)

    if _is_mllm:
        from vllm_mlx.models import MLXMultimodalLM
        logger.info(f"Loading MLLM model: {model_name}")
        _model = MLXMultimodalLM(model_name)
    else:
        from vllm_mlx.models import MLXLanguageModel
        logger.info(f"Loading LLM model: {model_name}")
        _model = MLXLanguageModel(model_name)

    _model.load()
    _model_name = model_name
    model_type = "MLLM" if _is_mllm else "LLM"
    logger.info(f"{model_type} model loaded: {model_name}")
    logger.info(f"Default max tokens: {_default_max_tokens}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_name": _model_name,
        "model_type": "mllm" if _is_mllm else "llm",
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
    model = get_model()

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    if request.stream:
        return StreamingResponse(
            stream_completion(model, prompts[0], request),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing
    start_time = time.perf_counter()
    choices = []
    total_completion_tokens = 0

    for i, prompt in enumerate(prompts):
        output = model.generate(
            prompt=prompt,
            max_tokens=request.max_tokens or _default_max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop if hasattr(model, 'generate') and not _is_mllm else None,
        )

        choices.append(CompletionChoice(
            index=i,
            text=clean_output_text(output.text),
            finish_reason=output.finish_reason,
        ))
        # Handle both LLM (tokens) and MLLM (completion_tokens) outputs
        if hasattr(output, 'tokens'):
            total_completion_tokens += len(output.tokens)
        elif hasattr(output, 'completion_tokens'):
            total_completion_tokens += output.completion_tokens

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(f"Completion: {total_completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

    return CompletionResponse(
        model=request.model,
        choices=choices,
        usage=Usage(
            completion_tokens=total_completion_tokens,
            total_tokens=total_completion_tokens,
        ),
    )


def extract_multimodal_content(messages: list[Message]) -> tuple[list[dict], list[str], list[str]]:
    """
    Extract text content, images, and videos from OpenAI-format messages.

    Returns:
        (processed_messages, images, videos)
    """
    processed_messages = []
    images = []
    videos = []

    for msg in messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal message - extract text and media
            text_parts = []
            for item in content:
                # Handle both Pydantic models and dicts
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                elif hasattr(item, 'dict'):
                    item = item.dict()

                item_type = item.get("type", "")

                if item_type == "text":
                    text_parts.append(item.get("text", ""))

                elif item_type == "image_url":
                    img_url = item.get("image_url", {})
                    if isinstance(img_url, str):
                        images.append(img_url)
                    elif isinstance(img_url, dict):
                        images.append(img_url.get("url", ""))

                elif item_type == "image":
                    images.append(item.get("image", item.get("url", "")))

                elif item_type == "video":
                    videos.append(item.get("video", item.get("url", "")))

                elif item_type == "video_url":
                    vid_url = item.get("video_url", {})
                    if isinstance(vid_url, str):
                        videos.append(vid_url)
                    elif isinstance(vid_url, dict):
                        videos.append(vid_url.get("url", ""))

            # Combine text parts
            combined_text = "\n".join(text_parts) if text_parts else ""
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages, images, videos


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support (similar to OpenAI format):
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Video can also be provided as:
    - Local path: {"type": "video", "video": "/path/to/video.mp4"}
    - Base64: {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}
    """
    model = get_model()

    # Extract text, images, and videos from messages
    messages, images, videos = extract_multimodal_content(request.messages)

    has_media = bool(images or videos)

    # Handle MLLM with multimodal content
    if _is_mllm and has_media:
        # Use MLLM model's chat method
        mllm_kwargs = {}
        if request.video_fps:
            mllm_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            mllm_kwargs["video_max_frames"] = request.video_max_frames

        # Build messages with images/videos for MLLM
        mllm_messages = []
        for msg in request.messages:
            mllm_messages.append({
                "role": msg.role,
                "content": msg.content if isinstance(msg.content, (str, list)) else str(msg.content)
            })

        if request.stream:
            # MLLM streaming (may fall back to non-streaming)
            return StreamingResponse(
                stream_mllm_chat_completion(model, mllm_messages, request, **mllm_kwargs),
                media_type="text/event-stream",
            )

        # Non-streaming MLLM response with timing
        start_time = time.perf_counter()

        output = model.chat(
            messages=mllm_messages,
            max_tokens=request.max_tokens or _default_max_tokens,
            temperature=request.temperature,
            **mllm_kwargs,
        )

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"MLLM completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionChoice(
                message=Message(role="assistant", content=clean_output_text(output.text)),
                finish_reason=output.finish_reason,
            )],
            usage=Usage(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.prompt_tokens + output.completion_tokens,
            ),
        )

    # Standard LLM chat completion
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(model, messages, request),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing
    start_time = time.perf_counter()

    output = model.chat(
        messages=messages,
        max_tokens=request.max_tokens or _default_max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    elapsed = time.perf_counter() - start_time
    completion_tokens = len(output.tokens) if hasattr(output, 'tokens') else 0
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    logger.info(f"Chat completion: {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

    return ChatCompletionResponse(
        model=request.model,
        choices=[ChatCompletionChoice(
            message=Message(role="assistant", content=clean_output_text(output.text)),
            finish_reason=output.finish_reason,
        )],
        usage=Usage(
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
        ),
    )


async def stream_completion(
    model,
    prompt: str,
    request: CompletionRequest,
) -> AsyncIterator[str]:
    """Stream completion response."""
    import json

    for chunk in model.stream_generate(
        prompt=prompt,
        max_tokens=request.max_tokens or _default_max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    ):
        data = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "text": chunk.text,
                "finish_reason": chunk.finish_reason if chunk.finished else None,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


async def stream_chat_completion(
    model,
    messages: list[dict],
    request: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """Stream chat completion response."""
    import json

    # Apply chat template - check for tokenizer or processor
    tokenizer = getattr(model, 'tokenizer', None) or getattr(model, 'processor', None)

    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt += "\nassistant:"

    # Check if model supports streaming
    if not hasattr(model, 'stream_generate'):
        # Fall back to non-streaming for models without stream support
        output = model.chat(
            messages=messages,
            max_tokens=request.max_tokens or _default_max_tokens,
            temperature=request.temperature,
        ) if hasattr(model, 'chat') else model.generate(
            prompt=prompt,
            max_tokens=request.max_tokens or _default_max_tokens,
            temperature=request.temperature,
        )
        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": clean_output_text(output.text)},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
        return

    for chunk in model.stream_generate(
        prompt=prompt,
        max_tokens=request.max_tokens or _default_max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    ):
        # Handle different chunk formats (text string or object)
        if isinstance(chunk, str):
            chunk_text = chunk
            is_finished = False
            finish_reason = None
        else:
            chunk_text = getattr(chunk, 'text', str(chunk))
            is_finished = getattr(chunk, 'finished', False)
            finish_reason = getattr(chunk, 'finish_reason', None) if is_finished else None

        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_text} if chunk_text else {},
                "finish_reason": finish_reason,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


async def stream_mllm_chat_completion(
    model,
    messages: list[dict],
    request: ChatCompletionRequest,
    **mllm_kwargs,
) -> AsyncIterator[str]:
    """Stream MLLM chat completion response."""
    import json

    # Try streaming, fall back to non-streaming if not supported
    try:
        for chunk in model.stream_chat(
            messages=messages,
            max_tokens=request.max_tokens or _default_max_tokens,
            temperature=request.temperature,
            **mllm_kwargs,
        ):
            data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk} if chunk else {},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0)

    except (AttributeError, NotImplementedError):
        # Fall back to non-streaming
        output = model.chat(
            messages=messages,
            max_tokens=request.max_tokens or _default_max_tokens,
            temperature=request.temperature,
            **mllm_kwargs,
        )
        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": clean_output_text(output.text)},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        description="vllm-mlx OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with an LLM model
    vllm-mlx --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Start with an MLLM model (auto-detected)
    vllm-mlx --model mlx-community/Qwen2-VL-2B-Instruct-4bit

    # Force MLLM mode
    vllm-mlx --model custom-model --mllm
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model)",
    )

    args = parser.parse_args()

    # Load model before starting server
    load_model(args.model, force_mllm=args.mllm)

    # Start server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
