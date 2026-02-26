# vLLM-MLX

**Production-grade OpenAI-compatible LLM server for Apple Silicon**

[![Fork](https://img.shields.io/badge/Fork-raullenchai%2Fvllm--mlx-orange?logo=github)](https://github.com/raullenchai/vllm-mlx)
[![Upstream](https://img.shields.io/badge/Upstream-waybarrios%2Fvllm--mlx-blue?logo=github)](https://github.com/waybarrios/vllm-mlx)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1500%2B-brightgreen.svg)](tests/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple)](https://support.apple.com/en-us/HT211814)

GPU-accelerated LLM inference on Mac via [MLX](https://github.com/ml-explore/mlx). Built on [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx), this fork adds **40+ commits** with tool calling, reasoning separation, prompt caching, smart cloud routing, and 1500+ tests.

---

## Highlights vs. Upstream

| Capability | Upstream | This Fork |
|-----------|----------|-----------|
| Tool calling | Not supported | Streaming + non-streaming, 7 parser formats |
| Reasoning separation | Not supported | Clean `reasoning_content` field (0% leak rate) |
| Multi-turn TTFT | Full prefill every turn | **10-30x faster** — persistent prompt cache |
| Long-context prefill | 50s for 52K tokens | **<1s** — smart cloud routing offloads to GPT-5/Claude |
| Decode speed | Baseline | 65-70 tok/s on M3 Ultra (Qwen3-Coder-Next-6bit) |
| KV cache quantization | Not available | 4-bit and 8-bit, halves memory for long contexts |
| Speculative decoding | Not available | `--draft-model` with prompt cache compatibility |
| Logprobs API | Not available | Per-token `logprobs` + `top_logprobs` |
| Test coverage | Minimal | **1500+ tests** |

---

## Quick Start

### 1. Install

```bash
pip install git+https://github.com/raullenchai/vllm-mlx.git
```

Or clone for development:

```bash
git clone https://github.com/raullenchai/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

### 2. Start the server

```bash
# Qwen3-Coder-Next — fast coding model (80B MoE, 3B active)
python -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

That's it. You now have an OpenAI-compatible server on `localhost:8000`.

### 3. Use it

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Works with any OpenAI-compatible client — Cursor, Continue, Aider, LangChain, or your own code.

---

## Features

### Tool Calling

Full OpenAI-compatible tool calling with streaming support. Works out of the box with 7 parser formats.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # "get_weather"
print(tool_call.function.arguments)  # '{"city": "Tokyo"}'
```

Supported parsers: `hermes`, `minimax`, `qwen`, `qwen3_coder`, `llama`, `deepseek`, `functionary`, and more. Use `--tool-call-parser <name>` to select.

### Reasoning Separation

Models with chain-of-thought (MiniMax-M2.5, Qwen3, DeepSeek-R1) output reasoning in a separate `reasoning_content` field — never mixed into `content`. 0% leak rate.

```bash
python -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --reasoning-parser minimax \
  --port 8000
```

### Prompt Cache (10-30x Faster Multi-Turn)

Persistent KV cache across requests. When consecutive requests share a prefix (system prompt + conversation history), only new tokens are prefilled:

| Context Size | Without Cache | With Cache | Speedup |
|-------------|---------------|------------|---------|
| 1K tokens | 0.7s | 0.3s | 2.3x |
| 4K tokens | 2.4s | 0.3s | 8x |
| 33K tokens | 28s | 0.3-0.9s | **30-90x** |

Always on in SimpleEngine (default mode). No flags needed.

### Smart Cloud Routing

**New.** Large-context requests are automatically routed to a cloud LLM when local prefill would be too slow. The routing decision is based on **new tokens** (after cache hit), not total input — so a 50K-token conversation with 2K new tokens stays local.

```bash
pip install litellm

# Route to GPT-5 when >20K new tokens need prefilling
OPENAI_API_KEY=sk-... python -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --cloud-model openai/gpt-5 \
  --cloud-threshold 20000 \
  --port 8000
```

```
Short request (44 new tokens)  → [LOCAL]  Qwen3 responds in 0.3s
Large request (15K new tokens) → [CLOUD]  GPT-5 responds in 3s (vs 50s local)
Next turn (cache hit, 200 new) → [LOCAL]  Back to local, 0.3s
```

Works with any litellm-supported provider: OpenAI, Anthropic, Google, Groq, etc. Clients see no difference — same API, transparent routing.

Disabled by default. Cost estimate: ~$0.02-0.05 per cloud-routed request with GPT-5.

---

## Supported Models

### Recommended

| Model | Quant | RAM | Decode | Best For |
|-------|-------|-----|--------|----------|
| Qwen3-Coder-Next | 4bit | 42GB | **70 tok/s** | Speed-first |
| Qwen3-Coder-Next | 6bit | 60GB | 65 tok/s | **Best balance** |
| Qwen3-Coder-Next | 8bit | 75GB | ~45 tok/s | Highest quality |
| MiniMax-M2.5 | 4bit | 120GB | 33-38 tok/s | Deep reasoning (192GB+ recommended) |

Benchmarks on Mac Studio M3 Ultra (256GB), 800 GB/s memory bandwidth.

### Any MLX Model

Any model from [mlx-community](https://huggingface.co/mlx-community) works:

```bash
# Llama
python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Mistral
python -m vllm_mlx.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Vision models
python -m vllm_mlx.server --model mlx-community/Qwen3-VL-4B-Instruct-MLX-4bit --mllm
```

---

## Server Flags

### Core

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | HuggingFace model name or local path | *(required)* |
| `--host` | Host to bind to | `0.0.0.0` |
| `--port` | Port to bind to | `8000` |
| `--max-tokens` | Default max tokens for generation | `32768` |
| `--continuous-batching` | Multi-user mode with scheduler | off |

### Tool Calling & Reasoning

| Flag | Description | Default |
|------|-------------|---------|
| `--tool-call-parser` | Parser: `hermes`, `minimax`, `qwen`, `qwen3_coder`, `llama`, `deepseek`, etc. | *(none)* |
| `--enable-auto-tool-choice` | Enable automatic tool choice (implied by `--tool-call-parser`) | off |
| `--enable-tool-logits-bias` | Jump-forward decoding for faster tool calls | off |
| `--reasoning-parser` | Parser: `minimax`, `qwen3`, `deepseek_r1`, `gpt_oss`, `harmony` | *(none)* |

### Performance

| Flag | Description | Default |
|------|-------------|---------|
| `--prefill-step-size` | Tokens per prefill chunk | `2048` |
| `--kv-bits` | KV cache quantization: `4` or `8` bit | *(full precision)* |
| `--draft-model` | Draft model for speculative decoding | *(none)* |
| `--num-draft-tokens` | Speculative tokens per step | `4` |

### Cloud Routing

| Flag | Description | Default |
|------|-------------|---------|
| `--cloud-model` | litellm model string (e.g. `openai/gpt-5`, `anthropic/claude-sonnet-4-5-20250929`) | *(disabled)* |
| `--cloud-threshold` | New token threshold to trigger cloud routing | `20000` |

### Security

| Flag | Description | Default |
|------|-------------|---------|
| `--api-key` | API key for authentication | *(no auth)* |
| `--rate-limit` | Requests per minute per client | *(unlimited)* |
| `--timeout` | Request timeout in seconds | `300` |

### Other

| Flag | Description | Default |
|------|-------------|---------|
| `--mllm` | Force multimodal (vision) mode | auto-detect |
| `--mcp-config` | MCP configuration file for tool integration | *(none)* |
| `--embedding-model` | Pre-load embedding model at startup | *(none)* |
| `--default-temperature` | Override default temperature | model default |

---

## Full Example Configurations

**Qwen3-Coder-Next — coding agent setup:**

```bash
python -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

**MiniMax-M2.5 — deep reasoning setup:**

```bash
python -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --reasoning-parser minimax \
  --prefill-step-size 4096 \
  --kv-bits 4 \
  --port 8000
```

**Hybrid local + cloud — best of both worlds:**

```bash
OPENAI_API_KEY=sk-... python -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --cloud-model openai/gpt-5 \
  --cloud-threshold 15000 \
  --port 8000
```

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │     OpenAI-compatible API (port 8000) │
                    │    /v1/chat/completions, /v1/models   │
                    └──────────────────┬───────────────────┘
                                       │
                              ┌────────┴────────┐
                              │  Cloud Router   │ (optional)
                              │  new_tokens >   │
                              │  threshold?     │
                              └───┬─────────┬───┘
                            yes   │         │  no
                     ┌────────────┘         └──────────────┐
                     ▼                                     ▼
          ┌─────────────────┐               ┌──────────────────────┐
          │  Cloud LLM      │               │   Local MLX Engine   │
          │  (via litellm)  │               │                      │
          │  GPT-5, Claude, │               │  ┌────────────────┐  │
          │  Gemini, etc.   │               │  │ SimpleEngine   │  │
          └─────────────────┘               │  │ + prompt cache │  │
                                            │  └───────┬────────┘  │
                                            │          │           │
                                            │  ┌───────┴────────┐  │
                                            │  │  mlx-lm/mlx-vlm│  │
                                            │  │  MLX + Metal   │  │
                                            │  └────────────────┘  │
                                            └──────────────────────┘
```

**SimpleEngine** (default) — Single-user, persistent prompt cache, maximum throughput.

**BatchedEngine** (`--continuous-batching`) — Multi-user, paged KV cache with prefix sharing.

**Cloud Router** (`--cloud-model`) — Routes large-context cold requests to cloud. Routing based on new tokens after cache hit, not total input.

---

## What This Fork Adds (vs. Upstream)

### Tool Calling & Reasoning (12 features)

- MiniMax reasoning parser — heuristic no-tag stripping (0% leak rate, was 60%)
- MiniMax tool call parser — streaming + non-streaming XML extraction
- `--tool-call-parser` flag — explicit parser selection for any model
- Auto-infer tool parser — `--reasoning-parser minimax` auto-selects matching tool parser
- Chunk-boundary leak fix — prevents XML leaking into reasoning stream
- Chinese reasoning pattern recognition
- Tool-use system prompt auto-injection (100% tool call rate, was 67%)
- Tool logits bias — jump-forward decoding for 2-5x faster structured output
- Hermes, Qwen, Qwen3-Coder, Llama, DeepSeek, Functionary parser support
- `developer` role normalization for chat template compatibility
- Logprobs API — per-token `logprobs` + `top_logprobs`
- Streaming disconnect guard — graceful handling of client disconnects

### Performance (6 features)

- Prompt cache (SimpleEngine) — persistent KV cache, 10-30x faster multi-turn
- `--prefill-step-size` — configurable prefill chunks for TTFT tuning
- `--kv-bits` — KV cache quantization (4/8 bit) for long contexts
- Speculative decoding — `--draft-model` with prompt cache compatibility
- Smart cloud routing — `--cloud-model` offloads large prefills to cloud LLMs
- Frequency-aware cache eviction — LRU-LFU hybrid under memory pressure

### Reliability (6 features)

- Accurate `prompt_tokens` reporting (was always 0)
- Prompt cache EOS fix — cache saved correctly on EOS
- Server crash prevention on malformed `response_format`
- GC control during generation to avoid latency spikes
- System prompt pinning in prefix cache
- **1500+ unit tests** across parsers, engine, server, and tool calling

---

## Roadmap

Research-backed optimizations ranked by impact-to-effort ratio. Papers surveyed from ICLR 2025, ICML 2025, NeurIPS 2025, ACL 2025.

| Priority | Technique | Expected Gain | Status |
|----------|-----------|---------------|--------|
| 1 | [ReDrafter](https://arxiv.org/abs/2403.09919) — Apple's speculative decoding (RNN draft head) | 1.4-1.5x decode | Not started |
| 2 | [KVSplit](https://github.com/dipampaul17/KVSplit) — Mixed-precision KV cache (8-bit K, 4-bit V) | 59% memory reduction | Not started |
| 3 | [DuoAttention](https://arxiv.org/abs/2410.10819) — Per-head adaptive KV cache | 2.5x memory, 2.2x decode | Not started |
| 4 | [FastKV](https://arxiv.org/abs/2502.01068) — Token-selective propagation | 1.8x prefill, 2.9x decode | Not started |
| 5 | [xKV](https://arxiv.org/abs/2503.18893) — Cross-layer SVD compression | 8x KV compression | Not started |
| 6 | [Medusa](https://arxiv.org/abs/2401.10774) — Multiple decoding heads | 2.2-2.8x decode | Not started |

---

## Contributing

Issues and PRs welcome at [github.com/raullenchai/vllm-mlx](https://github.com/raullenchai/vllm-mlx).

Built on [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) — all upstream features (multimodal, audio, embeddings, Anthropic API, MCP) are available.

## License

Apache 2.0 — see [LICENSE](LICENSE).
