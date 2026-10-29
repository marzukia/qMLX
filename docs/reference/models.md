# Supported Models

qMLX serves LLMs from
[mlx-community on HuggingFace](https://huggingface.co/mlx-community/models)
(any compatible quant by its full repo id) plus local model paths; the
aliases below cover the tuned defaults - Qwen-family first, but the
alias list spans many families (Gemma, Mistral, Phi, DeepSeek, GPT-OSS,
and more). Run `qmlx models` for the full alias list shipped with your
build.

## Qwen language models (tuned defaults)

| Model Family | Sizes | Quantization |
|--------------|-------|--------------|
| Qwen2 / Qwen3 | 0.5B to 72B (incl. 30B-A3B MoE) | Various |
| Qwen3-Coder | 30B | 4-bit |
| Qwen3.5 | 4B, 9B, 27B | 4-bit, 8-bit |
| Qwen3.6 | 27B, 35B-A3B | 4-bit, 8-bit |

### Recommended models

| Use Case | Model | Memory |
|----------|-------|--------|
| Fast/Light | `mlx-community/Qwen3-0.6B-8bit` | ~0.7 GB |
| Balanced | `qwen3.5-4b-4bit` | ~2.5 GB |
| Large | `mlx-community/Qwen3-30B-A3B-4bit` | ~16 GB |

## Using models

### From HuggingFace

```bash
qmlx serve mlx-community/Qwen3-4B-Instruct-4bit
```

### Local path

```bash
qmlx serve /path/to/local/model
```

## Finding models

Filter mlx-community models by:
- **Family**: `Qwen`, `Gemma`, `Mistral`, `Phi`, `DeepSeek`, `GPT-OSS`, …
- **Size**: `1B`, `3B`, `7B`, `8B`, `30B`, …
- **Quantization**: `4bit`, `6bit`, `8bit`, `2bit`, `bf16`, `mxfp4`
