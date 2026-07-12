# Installation

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Install with uv (recommended)

```bash
uv tool install qmlx-serve@latest
```

One command, isolated tool venv, no Python-version juggling — uv finds (or
installs) the right Python automatically. Upgrade later with
`uv tool upgrade qmlx`. If you don't have uv yet, install it first:
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

### `Refusing to load formula ... from untrusted tap`

Homebrew 4.x refuses installs from third-party taps until you mark them
trusted. Run the three-step install at the top of this page (tap, trust,
install). The `brew trust` line is the one that flips the refusal off.
Only needs to be done once per machine.

### `brew install` fails with `Operation not permitted`

Brew 5.x's install sandbox sometimes can't auto-tap `homebrew/core` mid-install.
Pre-tap it once, then retry:

```bash
brew tap homebrew/core --force   # ~1.3 GB, one-time
brew tap raullenchai/qmlx
brew trust raullenchai/qmlx
brew install qmlx-serve
```
