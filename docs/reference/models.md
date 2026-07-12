# Supported Models

qMLX serves Qwen-family text models. Any compatible quant from
[mlx-community on HuggingFace](https://huggingface.co/mlx-community/models)
can be served by its full repo id; the aliases below cover the tuned
defaults. Run `qmlx models` for the full alias list shipped with your build.

## Qwen language models

| Model Family | Sizes | Quantization |
|--------------|-------|--------------|
| Qwen2 / Qwen3 | 0.5B to 72B (incl. 30B-A3B MoE) | Various |
| Qwen3-Coder | 30B | 4-bit |
| Qwen3.5 | 4B, 9B, 27B | 4-bit, 8-bit |
| Qwen3.6 | 27B, 35B-A3B, Coder | 4-bit, 8-bit |

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
- **Family**: `Qwen`
- **Size**: `1B`, `3B`, `7B`, `8B`, `30B`
- **Quantization**: `4bit`, `8bit`, `bf16`
