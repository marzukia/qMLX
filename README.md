<h1 align="center">qMLX</h1>

<p align="center">
  <strong>A Qwen-specialized MLX inference engine for Apple Silicon.</strong>
  <br>
  <em>Fork of <a href="https://github.com/raullenchai/Rapid-MLX">Rapid-MLX</a>, tuned for Qwen3.5 / Qwen3.6, with disk-backed KV cache restore.</em>
</p>

<p align="center">
  <a href="https://github.com/marzukia/qMLX"><img src="https://img.shields.io/badge/repo-marzukia%2FqMLX-blue?logo=github" alt="Repo"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

---

## What this is

qMLX is a fork of [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) that narrows the focus to the Qwen family on Apple Silicon. Upstream aims to be a fast general engine for any MLX model. This fork keeps that engine but specializes it for Qwen3.5 and Qwen3.6, including their hybrid DeltaNet plus attention MoE variants, and adds disk-backed KV cache restore so large prefixes survive a restart instead of being re-prefilled from scratch.

Everything from upstream still works: the OpenAI and Anthropic compatible server, continuous batching, the radix prompt cache, tool-call parsing, and MTP speculative decode. The difference is what gets tuned and tested first. Qwen is the target; other families still load, but they are not the priority.

The Python package is still `vllm_mlx` and the CLI is still `rapid-mlx`. Those names are unchanged so existing installs, scripts, and launchd units keep working. Only the project branding changed.

### What the fork adds

- **Disk KV restore.** Prefix KV cache can be written to and restored from SSD, so a warm cache survives a server restart. Hybrid (DeltaNet + attention) models round-trip correctly, not just plain attention models.
- **Qwen3.5 / 3.6 specialization.** Parser defaults, MTP wiring, and KV codec eligibility are tuned for the Qwen hybrid MoE architecture first.
- **A Qwen-first roadmap.** See [ROADMAP.md](ROADMAP.md) for the optimization plan and current benchmark status.

Credit for the underlying engine goes to [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX).

---

## Install

qMLX runs from source. It is not published to PyPI or Homebrew.

```bash
git clone https://github.com/marzukia/qMLX.git
cd qMLX
python3.12 -m pip install -e .
```

The editable install exposes the `rapid-mlx` CLI on your PATH. Text inference is the base install (~460 MB). Vision, audio, embeddings, and DFlash speculative decoding are opt-in extras, e.g. `pip install -e '.[audio]'`.

Python 3.10 or newer is required. macOS ships 3.9, so install a newer one first if needed (`brew install python@3.12`).

---

## Quick start

Chat with a Qwen model in the terminal:

```bash
rapid-mlx chat qwen3.5-4b-4bit
```

First run downloads the weights and drops you into a REPL. Type `/help` for slash commands, `/exit` to quit.

Serve it over HTTP for other apps:

```bash
rapid-mlx serve qwen3.5-4b-4bit
```

This starts an OpenAI-compatible server on `http://localhost:8000`. Point any OpenAI SDK or client at `http://localhost:8000/v1`. Claude Code and the Anthropic SDK use `http://localhost:8000`; the Anthropic messages route is at `/v1/messages` under the same host.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say hello"}],
).choices[0].message.content)
```

---

## Engine features

| | |
|---|---|
| **Apple-Silicon-native** | Pure MLX kernels, no llama.cpp fallback. Continuous batching, radix prompt cache with DeltaNet RNN snapshots, and the TurboQuant K8V4 KV codec run at native MLX bandwidth on M1 through M4. |
| **Disk KV restore** | Prefix KV cache persists to SSD and restores after a restart, including for Qwen hybrid DeltaNet plus attention models. |
| **Drop-in OpenAI / Anthropic API** | `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/embeddings`, `/v1/audio/*`. Same wire as OpenAI and Anthropic clients, no adapter. |
| **Agent and framework coverage** | Codex CLI, Claude Code, OpenCode, Qwen Code, OpenHands, Hermes Agent, Aider, Kilo Code, plus LangChain, PydanticAI, and smolagents are wire-tested against real weights. |

---

## Choose a model

qMLX targets Qwen3.5 and Qwen3.6. `rapid-mlx models` lists every alias; `rapid-mlx info <alias>` shows the per-alias profile (parser, MoE / hybrid flags, KV codec eligibility, speculative-decoding gates).

| RAM | Recommended | One-shot |
|---|---|---|
| **8–23 GB** MacBook Air/Pro | `qwen3.5-4b-4bit` | `rapid-mlx serve qwen3.5-4b-4bit` |
| **24–47 GB** MacBook Pro / Mac Mini | `qwen3.5-27b-4bit` | `rapid-mlx serve qwen3.5-27b-4bit` |
| **48–95 GB** Mac Studio | `qwen3.6-35b-8bit` | `rapid-mlx serve qwen3.6-35b-8bit` |
| **96 GB+** Mac Studio / Pro | `qwen3.5-122b-a10b-8bit` | `rapid-mlx serve qwen3.5-122b-a10b-8bit` |

Other model families still load through the upstream engine, but they are outside the fork's tuning and test focus.

---

## Command reference

```bash
rapid-mlx --help                    # top-level command list
rapid-mlx <subcommand> --help       # per-subcommand flags
```

Covers chat, serve, share, agents (setup / test), bench, models, pull, rm, ps, info, doctor, upgrade, telemetry, launch, and jlens.

Run the built-in self-check if something is off:

```bash
rapid-mlx doctor
```

Common issues:

- **Slower than expected.** Qwen3.5 / 3.6 default to thinking-on. Add `--no-think` to skip chain-of-thought.
- **Out of memory.** Pick a smaller quant from the table above.
- **Tool calls arriving as plain text.** Auto-recovery handles most cases; if not, set `--tool-call-parser` explicitly for your model.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report bugs and request models through the [qMLX issues](https://github.com/marzukia/qMLX/issues).

qMLX inherits opt-in anonymous telemetry from upstream. It is off by default and needs an explicit `rapid-mlx telemetry enable`. No prompts, completions, paths, IPs, or API keys are collected.

---

## License

Apache 2.0, same as upstream. See [LICENSE](LICENSE). This project is a fork of [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX).
