# AGENTS.md

qMLX: disk KV restore for hybrid MoE models on Apple Silicon — the `qmlx serve` CLI keeps a 122B Qwen model warm on a Mac.

## Commands

- Lint: `make lint` (wraps `scripts/dev_test.py lint` — ruff check + ruff format --check)
- Audit: `make audit` (CLI ↔ Config fidelity, ~1s)
- Test: `make test` (pytest unit suite) · `make smoke` (lint + audit + unit)
- Ruff direct: `ruff check vllm_mlx/ tests/` · `ruff format --check vllm_mlx/ tests/` (apply: `ruff format vllm_mlx/ tests/`)
- Tests: `pytest` (asyncio_mode auto, see pytest.ini)
- Serve: `python -m vllm_mlx.server` (mise.toml task `serve`)

## Layout

- `vllm_mlx/` — engine (serve, MTP, disk KV, CLI)
- `tests/` — unit + integration suites
- `scripts/` — dev tools (dev_test.py, audit, release gates)
- bench tiers: `qmlx bench --tier smoke|speed|harness|all` (engine-side, see Makefile help)
- `docs/` — guides + development docs

## Gotchas

- Python >=3.10 floor; macOS Metal target only.
- `HF_HUB_CACHE` env var for the model cache.
- pre-commit runs ruff + mypy.
