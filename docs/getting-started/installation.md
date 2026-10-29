# Installation

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4) - current platform support (Metal is the reference backend; mlx-rocm and mlx-cuda ported builds are in progress)
- Python 3.10+

## Install with uv (recommended)

```bash
uv tool install qmlx-serve@latest
```

One command, isolated tool venv, no Python-version juggling — uv finds (or
installs) the right Python automatically. Upgrade later with
`uv tool upgrade qmlx-serve`. If you don't have uv yet, install it first:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Install with pip

```bash
pip install qmlx-serve
```

If `python3 --version` reports 3.9 (macOS default), install a newer Python
first: `brew install python@3.12` then `python3.12 -m pip install qmlx-serve`.

### From source (for development)

```bash
git clone https://github.com/marzukia/qMLX.git
cd qMLX
pip install -e .
```

## Optional Extras

The base text-only install is ~460 MB. Vision and other capabilities ship as opt-in extras.

| Extra | Install | Adds |
|---|---|---|
| `vision` | `pip install 'qmlx-serve[vision]'` | mlx-vlm + torch (~322 MB) for the Gemma 4 / Qwen-VL model paths |
| `chat` | `pip install 'qmlx-serve[chat]'` | Gradio web UI (~150 MB) |
| `guided` | `pip install 'qmlx-serve[guided]'` | outlines (~80 MB) for schema-constrained JSON |
| `all` | `pip install 'qmlx-serve[all]'` | Everything above (~1.1 GB) |

## Verify Installation

```bash
# Check CLI
qmlx --help
qmlx version

# Smallest interactive smoke test (downloads ~2.5 GB on first run)
qmlx chat qwen3.5-4b-4bit
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
qmlx serve qwen3.5-4b-4bit
```
