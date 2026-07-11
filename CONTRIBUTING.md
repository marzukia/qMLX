# Contributing to qMLX

qMLX is a small, focused fork of [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). Its scope is deliberately narrow: disk KV checkpoint and restore for hybrid recurrent plus attention MoE models (Qwen3.5-122B-A10B first), honest phase-split metrics, and the eviction and divergence-logging work that supports them. Contributions that fit that scope are welcome. General qMLX features, new model families, and broad serving improvements are better sent upstream.

## Scope check before you start

Good fits for a qMLX PR:

- Bugs in the disk KV restore path, eviction, or checkpointing
- Metrics correctness (decode tok/s, prefill throughput, restore hit rate, TTFT)
- Hybrid-cache correctness on Qwen models
- Documentation fixes for anything qMLX-specific

Probably better upstream:

- New model aliases, parsers, or model family support
- API surface changes
- Anything not touched by the fork

If you are not sure, open an issue first and ask.

## Development setup

qMLX is not published to PyPI or Homebrew. There is no release automation on this fork. Install from source:

```bash
git clone https://github.com/marzukia/qMLX.git
cd qMLX
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff
```

**Requirements:** Python 3.11+, macOS with Apple Silicon. The disk-restore path is developed and tested on an M3 Ultra; most unit tests run on any Apple Silicon Mac without a model download.

The package is still imported as `vllm_mlx` and the CLI is still `qmlx`. The `qmlx_*` metric names, `QMLX_*` environment variables, and the `~/.cache/qmlx/` cache path are also unchanged. These are kept on purpose for compatibility; please do not rename them in a PR.

To run a dev server against the model this fork targets:

```bash
qmlx serve mlx-community/Qwen3.5-122B-A10B-4bit \
  --text-only --max-num-seqs 1 \
  --enable-prefix-cache --prefix-cache-index radix \
  --enable-disk-kv-restore
```

## Running tests

```bash
# All unit tests (no model needed for most)
python3 -m pytest tests/ -x -q

# A specific test file
python3 -m pytest tests/test_disk_kv_checkpoint.py -v

# Lint and format
ruff check .
ruff format --check .
```

Some tests require a running server or a downloaded model; they skip themselves when the environment is missing. `ruff check` and `ruff format --check` must pass before a PR will be reviewed.

## Pull request workflow

1. Fork the repo and create a branch: `feat/`, `fix/`, `docs/`, `refactor/`
2. Make your changes, with tests where the change is testable
3. Run `ruff check` and `ruff format` before committing
4. Open a PR against `main` with a clear description

The PR template asks why the change is needed and whether AI was used. Both questions are inherited from upstream and kept because they work. Say what problem the change fixes in concrete terms, and if AI wrote or reviewed part of the diff, say which part and how you verified it. Nobody asks for prompt transcripts. The standard is simple: you should be able to explain the intent and behaviour of every change in your PR.

The repository ships upstream's PR validation pipeline (`scripts/pr_validate/`). Running it is optional but useful:

```bash
PR_VALIDATE_NO_DEEPSEEK=1 PR_VALIDATE_NO_STRESS=1 \
    python3 -m scripts.pr_validate.pr_validate <PR#>
```

## Code style

- `ruff` for linting and formatting
- Type hints encouraged but not required
- Keep changes focused: one fix or feature per PR
- British or American spelling both fine in code; docs lean British

## Licensing of contributions

qMLX's original work is MIT licensed; inherited qMLX code remains Apache-2.0 (see LICENSE, LICENSE-APACHE, and NOTICE). By submitting a PR you agree your contribution is licensed under the MIT License.
