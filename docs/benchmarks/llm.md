# LLM Benchmarks

## Running LLM Benchmarks

```bash
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 5 --max-tokens 256
```

## Results (M4 Max, 128GB)

| Model | Gen Speed | TTFT* | Memory |
|-------|-----------|-------|--------|
| Qwen3-0.6B-8bit | 402.3 tok/s | 58.6 ms | 0.68 GB |
| Llama-3.2-1B-Instruct-4bit | 463.6 tok/s | 49.2 ms | 0.69 GB |
| Qwen2.5-1.5B-Instruct-4bit | 308.5 tok/s | 86.2 ms | 0.84 GB |
| Llama-3.2-3B-Instruct-4bit | 200.1 tok/s | 81.4 ms | 1.79 GB |
| Qwen3-30B-A3B-4bit | 123.9 tok/s | 126.9 ms | 16.05 GB |

*TTFT = Time to First Token (latency until the model starts generating)

## Continuous Batching Results

*Updated January 2026 with streaming detokenizer optimization (Phase 9.1)*

| Model | Throughput | Completion tok/s | Prompts/sec |
|-------|------------|------------------|-------------|
| Qwen3-0.6B-8bit | **743.6 tok/s** | 723.8 tok/s | 2.83 |
| Llama-3.2-1B-Instruct-4bit | **717.5 tok/s** | 692.4 tok/s | 3.14 |
| Qwen2.5-1.5B-Instruct-4bit | **589.5 tok/s** | 573.8 tok/s | 2.24 |
| Llama-3.2-3B-Instruct-4bit | **378.7 tok/s** | 364.8 tok/s | 1.73 |

*Test: 5 prompts, 256 max tokens, continuous batching mode*

### Improvement from Streaming Detokenizer

| Model | Before | After | Improvement |
|-------|--------|-------|-------------|
| Llama-3.2-1B-Instruct-4bit | 613.0 tok/s | 717.5 tok/s | **+17%** |
| Qwen2.5-1.5B-Instruct-4bit | 322.2 tok/s | 589.5 tok/s | **+83%** |
| Llama-3.2-3B-Instruct-4bit | 208.1 tok/s | 378.7 tok/s | **+82%** |

## Streaming Performance

| Model | TTFT | Generation Speed |
|-------|------|------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1ms | 140.3 tok/s |

## Prefix Cache Results

*Updated January 2026*

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit

  Test Results:
  Test   | Description          | Expected | Actual | Time    | Status
  -------+----------------------+----------+--------+---------+-------
  TEST 1 | First request        | MISS     | MISS   | 185.3ms | ✓
  TEST 2 | Same prompt (cached) | HIT      | HIT    | 175.8ms | ✓
  TEST 3 | Different prompt     | MISS     | MISS   | 169.9ms | ✓

  Final Cache Statistics:
  Metric           | Value
  -----------------+------
  Total Requests   | 3
  Cache Hits       | 1
  Cache Misses     | 2
  Hit Rate         | 33.3%
  Tokens Saved     | 15
  Speedup (cached) | 1.05x

======================================================================
  ✓ ALL TESTS PASSED - Prefix cache working correctly!
======================================================================
```

## Paged Cache Results

*Updated January 2026 - Test: 20 real inference requests in 2 rounds with ~286 token shared system prompt*

```
======================================================================
  PAGED KV CACHE - REAL INFERENCE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit
  Requests: 20 (2 rounds of 10)
  System prompt: ~286 tokens (shared)

--------------------------------------------------
Test 1: WITHOUT Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 3.63s
  Total completion tokens: 1000
  Throughput: 275.3 tok/s
  Cache hits: 0
  Tokens saved: 0

--------------------------------------------------
Test 2: WITH Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 3.48s
  Total completion tokens: 1000
  Throughput: 287.4 tok/s

  Paged Cache Stats:
    Blocks allocated: 25
    Shared blocks: 4
    Cache hits: 10
    Tokens saved: 2560

==================================================
SUMMARY
==================================================
  Without paged cache: 275.3 tok/s
  With paged cache:    287.4 tok/s

  Speedup: 1.04x
  Cache hits: 10 (all Round 2 requests)
  Tokens saved: 2,560 (~256 tokens × 10 requests)
==================================================
```

## Streaming Detokenizer Optimization

*Phase 9.1: Replaced naive `tokenizer.decode([token])` with mlx-lm's `BPEStreamingDetokenizer`*

### The Problem

The naive approach calls `decode()` for each token, which has O(T²) complexity because BPE tokenizers need to re-process context for each decode call.

### The Solution

Use mlx-lm's `StreamingDetokenizer` which maintains state and provides O(T) complexity:
- **BPEStreamingDetokenizer** for GPT/Qwen models (ByteLevel decoder)
- **SPMStreamingDetokenizer** for Llama/Mistral models (SentencePiece)

### Benchmark Results (M4 Max)

```bash
vllm-mlx bench-detok
```

| Sequence | Tokens | Naive decode() | Streaming | Speedup |
|----------|--------|----------------|-----------|---------|
| Short | 8 | 0.020ms | 0.019ms | 1.05x |
| Medium | 103 | 0.155ms | 0.097ms | 1.59x |
| Long | 511 | 0.752ms | 0.371ms | **2.03x** |
| 1K tokens | 1191 | 1.743ms | 0.833ms | **2.09x** |
| 2K tokens | 2381 | 3.493ms | 1.737ms | **2.01x** |
| 4K tokens | 4761 | 7.125ms | 3.806ms | **1.87x** |

**Average speedup: 1.77x** (up to **2.33x** in real-world generation)

### Real-World Impact

With ~2000 generated tokens:
```
Method                            Time    Speedup
----------------------------------------------------------------------
Naive decode():                 3.37ms      1.00x
Streaming detokenizer:          1.45ms      2.33x
----------------------------------------------------------------------
Time saved per request:         1.92ms
Per-token savings:               1.0µs
```

### Where It's Active

| Mode | Command | Optimization |
|------|---------|--------------|
| Continuous Batching | `--continuous-batching` | Scheduler `_detokenizer_pool` |
| Simple Mode | (default) | mlx-lm's internal streaming |

Both modes automatically use the optimized detokenizer - no configuration needed.

## Metrics Reference

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token - latency until model starts responding (ms) |
| **TPOT** | Time Per Output Token - time between each generated token (ms/token) |
| **Generation TPS** | Output tokens per second (tok/s) |
| **Processing TPS** | Input/prompt tokens processed per second (tok/s) |
| **End-to-End Latency** | Total time from request to complete response |
| **Total Throughput** | Overall tokens (input + output) per second |

## Running Benchmarks

```bash
# Basic benchmark
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# With more prompts
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --prompts 10

# Save results
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results.json

# Continuous batching test
python tests/test_continuous_batching.py

# Prefix cache test
python tests/test_prefix_cache.py

# Paged cache test
python tests/test_paged_cache_real_inference.py

# Streaming detokenizer benchmark
vllm-mlx bench-detok
vllm-mlx bench-detok mlx-community/Llama-3.2-1B-Instruct-4bit --iterations 5
```
