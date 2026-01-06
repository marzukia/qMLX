# Installation

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## From Source (Recommended)

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

pip install -e .
```

### Optional: Vision Support

For video processing with transformers:

```bash
pip install -e ".[vision]"
```

## What Gets Installed

- `mlx`, `mlx-lm`, `mlx-vlm` - MLX framework and model libraries
- `transformers`, `tokenizers` - HuggingFace libraries
- `opencv-python` - Video processing
- `gradio` - Chat UI
- `psutil` - Resource monitoring

## Verify Installation

```bash
# Check CLI commands
vllm-mlx --help
vllm-mlx-bench --help
vllm-mlx-chat --help

# Test with a small model
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 1
```

## Troubleshooting

### MLX not found

Ensure you're on Apple Silicon:
```bash
uname -m  # Should output "arm64"
```

### Model download fails

Check your internet connection and HuggingFace access. Some models require authentication:
```bash
huggingface-cli login
```

### Out of memory

Use a smaller quantized model:
```bash
vllm-mlx serve mlx-community/Llama-3.2-1B-Instruct-4bit
```
