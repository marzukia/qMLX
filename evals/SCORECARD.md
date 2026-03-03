# vllm-mlx Model Scorecard

*Auto-generated on 2026-03-03 01:33 UTC*

## Comparison Table

| Model | Quant | Hardware | Decode (s) | Decode (l) | Tools | Coding | Reasoning | General | Parser | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-OSS-20B-mxfp4-q8 | mxfp4-q8 | Apple M3 Ultra (256GB) | 89.1 t/s | 124 t/s | 17% | 90% | 90% | 100% | minimax | 2026-03-03 |
| MiniMax-M2.5-4bit | 4bit | Apple M3 Ultra (256GB) | 44.9 t/s | 50.6 t/s | 87% | 40% | 60% | 90% | minimax | 2026-03-03 |
| Qwen3-0.6B-4bit | 4bit | Apple M3 Ultra (256GB) | 293.8 t/s | 372.3 t/s | 50% | 0% | 30% | 50% | hermes | 2026-03-03 |
| Qwen3-Coder-Next-4bit | 4bit | Apple M3 Ultra (256GB) | 35.5 t/s | 73.9 t/s | 90% | 100% | 80% | 100% | hermes | 2026-03-03 |
| Qwen3-Coder-Next-6bit | 6bit | Apple M3 Ultra (256GB) | 33.1 t/s | 67.7 t/s | 87% | 100% | 90% | 100% | hermes | 2026-03-03 |
| Qwen3.5-122B-A10B-8bit | 8bit | Apple M3 Ultra (256GB) | 39.8 t/s | 43.3 t/s | 100% | 70% | 30% | 50% | hermes | 2026-03-03 |
| Qwen3.5-122B-A10B-mxfp4 | mxfp4 | Apple M3 Ultra (256GB) | 52.3 t/s | 57.7 t/s | 97% | 90% | 40% | 70% | hermes | 2026-03-03 |
| Qwen3.5-35B-A3B-4bit | 4bit | Apple M3 Ultra (256GB) | 92.3 t/s | 104.5 t/s | 93% | 100% | 40% | 60% | hermes | 2026-03-03 |
| Qwen3.5-35B-A3B-8bit | 8bit | Apple M3 Ultra (256GB) | 74.3 t/s | 82 t/s | 97% | 90% | 40% | 70% | hermes | 2026-03-03 |

## Details

### GPT-OSS-20B-mxfp4-q8

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: minimax
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser minimax`
- **Date**: 2026-03-03
- **TTFT**: cold=0.4s, warm=0.2s
- **Decode**: short=89.1 t/s, long=124 t/s
- **Tool Calling**: 17% (5/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 90% (9/10)
- **General**: 100% (10/10)
- **Eval time**: 98.5s

### MiniMax-M2.5-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: minimax
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser minimax`
- **Date**: 2026-03-03
- **TTFT**: cold=1.4s, warm=0.5s
- **Decode**: short=44.9 t/s, long=50.6 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 40% (4/10)
- **Reasoning**: 60% (6/10)
- **General**: 90% (9/10)
- **Eval time**: 385.3s

### Qwen3-0.6B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.1s, warm=0.1s
- **Decode**: short=293.8 t/s, long=372.3 t/s
- **Tool Calling**: 50% (15/30)
- **Coding**: 0% (0/10)
- **Reasoning**: 30% (3/10)
- **General**: 50% (5/10)
- **Eval time**: 77.5s

### Qwen3-Coder-Next-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.7s, warm=0.0s
- **Decode**: short=35.5 t/s, long=73.9 t/s
- **Tool Calling**: 90% (27/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 80% (8/10)
- **General**: 100% (10/10)
- **Eval time**: 128.1s

### Qwen3-Coder-Next-6bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.8s, warm=0.0s
- **Decode**: short=33.1 t/s, long=67.7 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 90% (9/10)
- **General**: 100% (10/10)
- **Eval time**: 138.8s

### Qwen3.5-122B-A10B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=1.6s, warm=0.0s
- **Decode**: short=39.8 t/s, long=43.3 t/s
- **Tool Calling**: 100% (30/30)
- **Coding**: 70% (7/10)
- **Reasoning**: 30% (3/10)
- **General**: 50% (5/10)
- **Eval time**: 512.8s

### Qwen3.5-122B-A10B-mxfp4

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=1.0s, warm=0.0s
- **Decode**: short=52.3 t/s, long=57.7 t/s
- **Tool Calling**: 97% (29/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 40% (4/10)
- **General**: 70% (7/10)
- **Eval time**: 375.5s

### Qwen3.5-35B-A3B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.5s, warm=0.0s
- **Decode**: short=92.3 t/s, long=104.5 t/s
- **Tool Calling**: 93% (28/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 40% (4/10)
- **General**: 60% (6/10)
- **Eval time**: 207.8s

### Qwen3.5-35B-A3B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.6s, warm=0.0s
- **Decode**: short=74.3 t/s, long=82 t/s
- **Tool Calling**: 97% (29/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 40% (4/10)
- **General**: 70% (7/10)
- **Eval time**: 271.9s

---

## How to Add Your Results

1. Start vllm-mlx with your model: `vllm-mlx serve <model> --port 8000`
2. Run the eval: `python evals/run_eval.py --model "<model-name>" --quantization <quant>`
3. Your results are saved to `evals/results/<model>.json`
4. Regenerate this table: `python evals/generate_scorecard.py`
5. Submit a PR with your JSON file!

