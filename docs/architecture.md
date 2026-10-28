# qMLX System Architecture

## Overview

Inference requests flow through tokenize → prefix-cache lookup → prefill → decode → detokenize, all driven by the scheduler over the mlx-lm public API (`insert`/`next`/`remove`/`close`). The engine layer (`engine/`) wraps mlx-lm with continuous batching; speculative drafters (DFlash, SuffixDecoding, MTP) live in `spec_decode/`; reasoning and tool-call parsing live in `reasoning/` and `tool_parsers/` and feed the streaming `PostProcessor`.

Design principles:

1. **No monkey-patching** — use mlx-lm's public API (`insert`/`next`/`remove`/`close`).
2. **mlx-lm version agnostic** — the public API is stable across versions.
3. **Per-request parsers** — reasoning + tool-call parsers are instantiated per request, never shared.

## Design ethos

qMLX is an engine layer for the MLX ecosystem on three backends: Apple
Metal (today), `mlx-rocm`, and `mlx-cuda` (ported builds). The target
machines are **bandwidth-bound and memory-constrained**, and two
first-class design commitments follow from that.

### Multiplexing, not parallelisation

Many requests **interleave** on one stream; they do not **overlap** on
several. The engine core (`engine_core.py`) runs a single owner thread —
the `mlx-step` worker — and *all* model/KV-touching MLX work is
submitted to it. The asyncio event loop only dispatches: scheduling,
admission, streaming, and housekeeping happen there, but every forward
pass, cache mutation, and checkpoint write lands on the owner thread.
This is a design invariant, not a workaround: a second thread issuing
MLX work would overlap on stream state rather than interleave, and on a
bandwidth-bound device that contention costs more than it buys.

The one-owner-thread rule is also the **conformance contract for ported
backends**. A backend declares its stream semantics through the
device/backend seam (`vllm_mlx/device/backend.py`, "Stream contract")
instead of the engine probing for error strings, and it must honour the
owner-thread rule (or declare a multi-stream model the engine can
schedule around) — pinned by the backend conformance tests.

### Cold-cache restore is the memory tier

The KV cache has two tiers, and the disk is not an add-on to the RAM
tier — it *is* the RAM of the second tier:

1. **Tier 1 (RAM)** — the in-process prefix cache: hot, shared
   system-prompt blocks, deduplicated and reference-counted.
2. **Tier 2 (SSD)** — the disk-KV checkpoint tier
   (`runtime/disk_kv_checkpoint.py`): long-context caches are
   snapshotted at 256-token boundaries and, on a miss, the checkpoint is
   **restored instead of re-prefilled**. A resumed session, a shared
   system prompt, or an agent walking up to its context cap reads its
   cache back from the SSD; restore-on-miss is on by default. This is
   the headline feature that makes long-context serving viable on
   machines with limited RAM.

## Module Map

```
vllm_mlx/
├── server.py                  # App factory + model loading + CLI (1047 lines)
│
├── config/                    # ServerConfig singleton
│   └── server_config.py
│
├── service/                   # Request processing
│   ├── helpers.py             # Shared request helpers (_resolve_*, get_engine, etc.)
│   └── postprocessor.py       # Streaming pipeline (100% test coverage)
│
├── routes/                    # HTTP endpoints
│   ├── chat.py                # /v1/chat/completions
│   ├── completions.py         # /v1/completions
│   ├── anthropic.py           # /v1/messages (Anthropic API)
│   ├── health.py              # /health, /v1/cache/*, /v1/status
│   ├── models.py, embeddings.py, audio.py, mcp_routes.py
│
├── engine/                    # Engine abstraction
│   ├── base.py                # BaseEngine ABC, GenerationOutput
│   ├── batched.py             # BatchedEngine (default, continuous batching)
│
├── engine_core.py             # AsyncEngineCore (event loop + thread executor)
├── scheduler.py               # Scheduler (request queue + batch management)
│
├── reasoning/                 # 7 reasoning parsers (Qwen3, DeepSeek, MiniMax, etc.)
├── tool_parsers/              # 20+ tool call parsers
├── agents/                    # 12 agent profiles (YAML)
├── runtime/                   # Model registry, cache persistence
├── middleware/                # Auth, rate limiting
├── doctor/                    # User self-diagnostic
│
├── domain/                    # Domain types
│   └── events.py              # StreamEvent (seam between PostProcessor and SSE)
│
└── mcp/                       # MCP tool integration

scripts/                       # Dev-only (NOT shipped with pip)
├── dev_test.py                # Unified test entry point
├── stress_test.py             # 8-scenario stress test
├── agent_soak_test.py         # 10-min agent soak test
└── cross_model_stress.py      # Multi-model validation

tests/                         # pytest unit tests (2100+)
```

## Request Flow

### Streaming Chat Completion

```
Client POST /v1/chat/completions (stream=true)
    ↓
routes/chat.py: create_chat_completion()
    ├── Validate request
    ├── Apply chat template
    ├── Inject tool/reasoning system prompts
    ↓
routes/chat.py: stream_chat_completion()
    ├── Create StreamingPostProcessor (per-request parser instances)
    ├── engine.stream_chat() → engine.stream_generate()
    │       ↓
    │   engine_core.py: add_request() → scheduler
    │       ↓
    │   scheduler.py: _schedule_waiting() → decode.insert()
    │   scheduler.py: step() → decode.step() → TokenResult
    │       ↓
    │   engine_core.py: stream_outputs() → RequestOutput
    │       ↓
    │   batched.py: yield GenerationOutput
    ↓
    PostProcessor.process_chunk() → StreamEvent
    ↓
    SSE formatting → yield "data: {...}\n\n"
```

## Performance Architecture

```
                    ┌─────────────────────────────┐
                    │     Metal GPU (Apple Silicon) │
                    │                               │
                    │  Model Forward   ← bottleneck │
                    │  (~10-50ms/step)              │
                    │                               │
                    └──────────┬────────────────────┘
                               │
                    ┌──────────▼────────────────────┐
                    │     Python Scheduler           │
                    │     (~0.5-1ms/step)            │
                    │                               │
                    │  Request queue                 │
                    │  Batch management              │
                    │  Cache lookup                  │
                    │  Token emission                │
                    └──────────┬────────────────────┘
                               │
                    ┌──────────▼────────────────────┐
                    │     API Layer (FastAPI)         │
                    │     (~0.1ms/request)           │
                    │                               │
                    │  SSE formatting                │
                    │  PostProcessor                 │
                    │  Response serialization        │
                    └───────────────────────────────┘

Bottleneck is always Metal GPU compute, not Python scheduling.
C/C++ scheduler rewrite would save <3% throughput.
```
