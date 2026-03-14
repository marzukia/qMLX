# Rapid-MLX Optimization Roadmap

> Goal: For every popular model on Apple Silicon, Rapid-MLX should be the fastest engine — **zero configuration required**. Users pick a model, we auto-apply the best optimizations.

## Strategy

1. **Profile** each popular model — measure decode speed, TTFT, memory usage
2. **Apply** the best optimization techniques per model (MTP, prompt cache, KV quant, speculative decode, EAGLE)
3. **Benchmark** against Ollama, llama.cpp, mlx-lm, LM Studio
4. **Publish** comparison table in README — users see the speed advantage and switch

---

## Optimization Techniques

### Already Implemented

| Technique | Speedup | Status | Notes |
|-----------|---------|--------|-------|
| **Prompt Cache** | 5-30x TTFT | Shipped (SimpleEngine) | Core advantage. Always on. |
| **KV Cache Quantization** | 1.0-1.3x + 4x memory | Shipped | `--kv-bits 4/8`. KV4 can be faster than unquantized on Apple Silicon. |
| **MTP (Qwen3-Next)** | 1.2-2.1x decode | Shipped (BatchedEngine only) | Need to port to SimpleEngine. |
| **Tool Call Recovery** | N/A (reliability) | Shipped | 17 parsers, auto-recovery. |

### To Implement

| Priority | Technique | Expected Speedup | Effort | Applicable Models |
|----------|-----------|-----------------|--------|-------------------|
| **P0** | MTP in SimpleEngine | 1.4x decode | Low | Qwen3-Next, Qwen3.5, DeepSeek-V3, Nemotron |
| **P1** | Standard Speculative Decode | 1.5-2.3x decode | Medium | Any model with small draft variant |
| **P1** | Auto-Optimization per model | N/A | Medium | All models — auto-detect and apply best technique |
| **P2** | EAGLE-3 on Metal | 3-6.5x decode | High | Qwen3-32B, Qwen3-8B, GPT-OSS, Llama-3 |
| **P3** | ReDrafter | 1.4-1.5x | Medium | Needs training heads per model (Apple has MLX code) |

### Not Pursuing

| Technique | Reason |
|-----------|--------|
| Medusa | Superseded by EAGLE-3, only old Vicuna heads available |
| Prompt Lookup Decoding | Benchmarked — minimal benefit, Qwen ArraysCache not trimmable |

---

## Model Optimization Matrix

For each model: which techniques apply, expected speedup, and benchmark status.

| # | Model | Active Params | RAM (4bit) | Optimization Plan | vs Ollama Target | Status |
|---|-------|--------------|-----------|-------------------|-----------------|--------|
| 1 | Qwen3.5-9B | 9B | 5.1 GB | Prompt cache + KV quant + auto-config | **2.7x** | **DONE** |
| 2 | Llama 3.2 3B | 3B | ~2 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 3 | Phi-4 Mini 14B | 14B | ~9 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 4 | Mistral 7B | 7B | ~4.4 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 5 | Mistral Small 24B | 24B | ~14 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 6 | Gemma 3 12B | 12B | ~8 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 7 | DeepSeek-R1-Distill 14B | 14B | ~9 GB | Prompt cache + KV quant + reasoning parser | 1.5-2x | Not started |
| 8 | Qwen 2.5 Coder 14B | 14B | ~9 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 9 | GPT-OSS 20B | 20B | ~12 GB | Prompt cache + KV quant + seed_oss parser | 1.5-2x | Not started |
| 10 | GLM-4.7-Flash 9B | 9B | ~6 GB | Prompt cache + KV quant + glm47 parser | 1.5-2x | Not started |
| 11 | Llama 3.3 70B | 70B | ~40 GB | Prompt cache + KV quant + spec decode (Llama-8B draft) | 1.5-2x | Not started |
| 12 | Gemma 3 27B | 27B | ~16 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 13 | Qwen3-30B-A3B | 3B active | ~18 GB | Prompt cache + KV quant + MTP | 2-2.5x | Not started |
| 14 | Qwen3.5-122B-A10B | 10B active | 65 GB | Prompt cache + KV quant + MTP | 2-2.5x | Not started |
| 15 | Qwen3-Coder-Next 80B | 3B active | ~30 GB | Prompt cache + KV quant + MTP | 2-2.5x | Not started |
| 16 | Llama 4 Scout 109B | 17B active | ~55 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 17 | DeepSeek R1 671B | 37B active | ~404 GB | Prompt cache + KV quant + MTP | 1.5-2x | Not started |
| 18 | Mixtral 8x7B | 13B active | ~26 GB | Prompt cache + KV quant | 1.5-2x | Not started |

### Auto-Optimization Vision

When a user runs:
```bash
rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --port 8000
```

The engine should automatically:
1. Detect model family → Qwen3.5
2. Apply best tool parser → `hermes`
3. Apply best reasoning parser → `qwen3`
4. Enable prompt cache → always on
5. Set optimal `--prefill-step-size` → based on model size
6. Apply KV cache quantization if beneficial → auto KV4/8
7. Enable MTP if model has MTP head → auto-detect
8. Set optimal temperature/sampling defaults

**Zero flags needed. Just `serve <model>` and get the best performance.**

---

## Benchmark Table (README Target)

The goal is to publish this table in README. Each cell = tok/s decode speed on the same hardware.

### Decode Speed (tok/s) — Apple M3 Ultra 256GB

| Model | Quant | Rapid-MLX | Ollama | llama.cpp | mlx-lm | LM Studio | Speedup |
|-------|-------|----------|--------|-----------|--------|-----------|---------|
| Qwen3.5-9B | 4bit | ? | ? | ? | ? | ? | ?x |
| Llama 3.2 3B | 4bit | ? | ? | ? | ? | ? | ?x |
| Phi-4 Mini 14B | 4bit | ? | ? | ? | ? | ? | ?x |
| ... | ... | ... | ... | ... | ... | ... | ... |

### TTFT — First Request (Cold) vs Second Request (Cached)

| Model | Quant | Rapid-MLX (cold) | Rapid-MLX (cached) | Ollama | llama.cpp | Cache Speedup |
|-------|-------|-----------------|-------------------|--------|-----------|---------------|
| Qwen3.5-9B | 4bit | ? | ? | ? | ? | ?x |
| ... | ... | ... | ... | ... | ... | ... |

---

## Progress Log

### 2026-03-13: Roadmap created
- Identified 18 most popular models on Mac (from Reddit r/LocalLLaMA, r/ollama, benchmarks)
- Mapped optimization techniques to each model
- Starting with Qwen3.5-9B as first model to profile + benchmark

### 2026-03-13: Qwen3.5-9B benchmark complete
- **Rapid-MLX: 108 tok/s decode, 0.14s cached TTFT**
- **Ollama: 41 tok/s decode, 0.28s cached TTFT**
- **Result: 2.7x faster decode, 2.0x faster multi-turn TTFT**
- Hardware: Mac Studio M3 Ultra 256GB
- Quantization: both 4-bit
