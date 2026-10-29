# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────┐
│                     API Layer                            │
│  (OpenAI-compatible: chat, completions, responses,       │
│   Anthropic, tools, MCP, reasoning)                     │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                      qMLX Engine                         │
│   (Scheduler + BatchedEngine - continuous batching)      │
└──────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
┌─────────────────┐                 ┌─────────────────┐
│     mlx-lm      │                 │     mlx-vlm     │
│     (LLM)       │                 │     (Vision)    │
└─────────────────┘                 └─────────────────┘
        │                                     │
        └──────────────────┬──────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│                         MLX                               │
│          (Apple ML Framework - Metal kernels)            │
└──────────────────────────────────────────────────────────┘
```

## Engine Architecture

`BatchedEngine` is the sole engine (the older `SimpleEngine` was deleted —
single-request workloads pay zero batching overhead, so the split was no
longer earning its keep).

### BatchedEngine
- `AsyncEngineCore` with continuous batching
- Scheduler with priority queue
- One engine instance handles both single-request and multi-tenant workloads

### Concurrency model (design ethos)

**One owner thread per device.** `AsyncEngineCore` runs a single
`mlx-step` worker thread and submits *all* model/KV-touching MLX work
to it; the asyncio loop only dispatches. Bandwidth-bound devices
interleave, they do not overlap. The one-owner-thread rule is the
conformance contract ported backends (`mlx-rocm` / `mlx-cuda`) must
honour, with stream semantics declared through the device/backend seam
(`vllm_mlx/device/backend.py`, "Stream contract") rather than probed for
error strings.

**The disk is the RAM of the second tier.** The disk-KV checkpoint tier
(`runtime/disk_kv_checkpoint.py`) is the engine's memory tier, not an
add-on: long-context caches are snapshotted to the SSD at 256-token
boundaries and restored on a miss instead of re-prefilled (on by
default). Full write-up: [System Architecture: Design
ethos](../architecture.md#design-ethos).

## Paged KV Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PagedCacheManager                          │
├─────────────────────────────────────────────────────────────────┤
│  FreeKVCacheBlockQueue     │  BlockHashToBlockMap               │
│  (O(1) doubly linked list) │  (hash → block for prefix caching) │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐  │  {hash_0: block_5}                 │
│  │ 3 │↔│ 7 │↔│ 2 │↔│ 9 │  │  {hash_1: block_12}                │
│  └───┘ └───┘ └───┘ └───┘  │  {hash_2: block_5}  (shared!)      │
│   LRU ───────────▶ MRU    │                                     │
├─────────────────────────────────────────────────────────────────┤
│  CacheBlock[0..N]:                                              │
│  - block_id, ref_count, block_hash                              │
│  - prev_free_block, next_free_block (doubly linked)             │
│  - cache_data: List[(keys, values)] per layer                   │
└─────────────────────────────────────────────────────────────────┘
```

### Cache Flow

```
Request Completion                    Cache Storage
       │                                    │
       ▼                                    ▼
┌──────────────────┐              ┌─────────────────────┐
│ response.cache() │ ───────────▶ │ Extract .state      │
│ (KVCache objects)│              │ (keys, values)      │
└──────────────────┘              └─────────────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │ Slice into 64-token │
                                  │ blocks + chain hash │
                                  └─────────────────────┘
                                            │
       New Request                          ▼
       │                          ┌─────────────────────┐
       ▼                          │ BlockHashToBlockMap │
┌──────────────────┐              │ deduplicate & share │
│ compute_block_   │ ◀─────────── └─────────────────────┘
│ hash(parent, tok)│
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Reconstruct via  │
│ mx.concatenate() │
│ + KVCache.from_  │
│ state()          │
└──────────────────┘
```

## Key Features

| Feature | Benefit |
|---------|---------|
| **1.14x Speedup** | Faster inference by reusing cached KV computations |
| **80% Memory Savings** | Share system prompt blocks across concurrent users |
| **vLLM Architecture** | FreeKVCacheBlockQueue, BlockHashToBlockMap, chain hashing |
| **Real Tensor Storage** | Extracts actual KV data using `.state` |
| **Block Deduplication** | Hash-based detection prevents duplicate storage |
| **Copy-on-Write (COW)** | Shared blocks only copied when modified |
| **O(1) LRU Eviction** | Doubly linked list for efficient cleanup |

## Module Structure

```
vllm_mlx/
├── api/
│   ├── models.py         # Pydantic models
│   ├── utils.py          # Shared utilities
│   ├── streaming.py      # Streaming JSON encoder
│   ├── tool_calling.py   # Tool call parsing
│   ├── guided.py         # Structured output (json_schema_to_pydantic)
│   └── anthropic_adapter.py, responses_adapter.py
├── device/
│   ├── backend.py        # Device/backend seam spec
│   └── metal.py          # Metal backend (reference implementation)
├── engine/
│   ├── base.py           # BaseEngine ABC
│   └── batched.py        # BatchedEngine (continuous batching)
├── mcp/
│   ├── client.py         # MCP client
│   ├── config.py         # Config loading
│   ├── executor.py       # Tool execution
│   ├── security.py       # Command validation
│   ├── tools.py          # Tool sandbox
│   ├── types.py          # MCP type definitions
│   └── manager.py        # Server management
├── models/               # Model package (registry and alias plumbing)
├── tool_parsers/         # Tool call parsers (6 registered names: hermes, nous, qwen, qwen3, qwen3_coder, qwen3_xml)
├── reasoning/            # Reasoning parser registry (qwen3 registered; base + harmony stop helpers)
├── server.py             # FastAPI app factory
├── engine_core.py        # AsyncEngineCore
├── scheduler.py          # LLM request scheduler
├── paged_cache.py        # Paged KV cache
├── prefix_cache.py       # Prefix cache manager
├── output_collector.py   # Request output collector
├── runtime/
│   ├── model_registry.py # Model detection & registry
│   └── disk_kv_checkpoint.py # Disk KV restore
└── cli.py                # CLI commands
```

## Request Flow

1. **API Request** → FastAPI endpoint (auth, rate limit)
2. **Engine** → BatchedEngine (single engine, continuous batching)
3. **Template Application** → Chat template formatting (with tool definitions if enabled)
4. **Generation** → mlx-lm (text; Metal reference backend)
5. **Post-processing** → Tool call parsing, reasoning extraction
6. **Streaming** → SSE response chunks
7. **Caching** → KV cache storage for reuse

## Hardware Detection

qmlx auto-detects the device through the backend seam:
- Chip name (M1, M2, M3, M4)
- Total memory
- Memory bandwidth (GB/s)
- GPU cores

```python
from vllm_mlx.optimizations import detect_hardware

hw = detect_hardware()
print(f"{hw.chip_name} ({hw.total_memory_gb:.0f} GB)")
```
