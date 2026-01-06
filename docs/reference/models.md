# Supported Models

All quantized models from [mlx-community on HuggingFace](https://huggingface.co/mlx-community/models) are compatible.

Browse thousands of pre-optimized models at: **https://huggingface.co/mlx-community/models**

## Language Models (via mlx-lm)

| Model Family | Sizes | Quantization |
|--------------|-------|--------------|
| Llama 3.x | 1B, 3B, 8B, 70B | 4-bit |
| Mistral | 7B, Mixtral 8x7B | 4-bit, 8-bit |
| Qwen2/Qwen3 | 0.5B to 72B | Various |
| Phi-3 | 3.8B, 14B | 4-bit |
| Gemma 2 | 2B, 9B, 27B | 4-bit |
| DeepSeek | 7B, 33B, 67B | 4-bit |

### Recommended Models

| Use Case | Model | Memory |
|----------|-------|--------|
| Fast/Light | `mlx-community/Qwen3-0.6B-8bit` | ~0.7 GB |
| Balanced | `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~1.8 GB |
| Quality | `mlx-community/Llama-3.1-8B-Instruct-4bit` | ~4.5 GB |
| Large | `mlx-community/Qwen3-30B-A3B-4bit` | ~16 GB |

## Multimodal Models (via mlx-vlm)

| Model Family | Example Models |
|--------------|----------------|
| **Qwen-VL** | `Qwen3-VL-4B-Instruct-3bit`, `Qwen3-VL-8B-Instruct-4bit`, `Qwen2-VL-2B/7B-Instruct-4bit` |
| **LLaVA** | `llava-1.5-7b-4bit`, `llava-v1.6-mistral-7b-4bit`, `llava-llama-3-8b-v1_1-4bit` |
| **Idefics** | `Idefics3-8B-Llama3-4bit`, `idefics2-8b-4bit` |
| **PaliGemma** | `paligemma2-3b-mix-224-4bit`, `paligemma-3b-mix-224-8bit` |
| **Pixtral** | `pixtral-12b-4bit`, `pixtral-12b-8bit` |
| **Molmo** | `Molmo-7B-D-0924-4bit`, `Molmo-7B-D-0924-8bit` |
| **Phi-3 Vision** | `Phi-3-vision-128k-instruct-4bit` |
| **DeepSeek-VL** | `deepseek-vl-7b-chat-4bit`, `deepseek-vl2-small-4bit` |

### Recommended VLM Models

| Use Case | Model | Memory |
|----------|-------|--------|
| Fast/Light | `mlx-community/Qwen3-VL-4B-Instruct-3bit` | ~3 GB |
| Balanced | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | ~6 GB |
| Quality | `mlx-community/Qwen3-VL-30B-A3B-Instruct-6bit` | ~20 GB |

## Model Detection

vllm-mlx auto-detects multimodal models by name patterns:
- Contains "VL", "Vision", "vision"
- Contains "llava", "idefics", "paligemma"
- Contains "pixtral", "molmo", "deepseek-vl"

## Using Models

### From HuggingFace

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Local Path

```bash
vllm-mlx serve /path/to/local/model
```

## Finding Models

Filter mlx-community models by:
- **LLM**: `Llama`, `Qwen`, `Mistral`, `Phi`, `Gemma`
- **VLM**: `-VL-`, `llava`, `paligemma`, `pixtral`, `molmo`, `idefics`, `deepseek-vl`
- **Size**: `1B`, `3B`, `7B`, `8B`, `70B`
- **Quantization**: `4bit`, `8bit`, `bf16`
