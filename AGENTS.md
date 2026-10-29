# AGENTS.md

qMLX: multi-backend MLX inference engine (Metal, mlx-rocm, mlx-cuda) with OpenAI-compatible serving, disk KV restore for hybrid MoE models, and speculative decoding, on a single device/backend seam.

## Commands

- Lint: `make lint` (wraps `scripts/dev_test.py lint`: ruff check + ruff format --check)
- Audit: `make audit` (CLI ↔ Config fidelity, ~1s)
- Test: `make test` (pytest unit suite) · `make smoke` (lint + audit + unit)
- Ruff direct: `ruff check vllm_mlx/ tests/` · `ruff format --check vllm_mlx/ tests/` (apply: `ruff format vllm_mlx/ tests/`)
- Tests: `pytest` (asyncio_mode auto, see pytest.ini)
- Serve: `python -m vllm_mlx.server` (mise.toml task `serve`)

## Layout

- `vllm_mlx/`: engine (serve, MTP, disk KV, CLI)
- `tests/`: unit + integration suites
- `scripts/`: dev tools (dev_test.py, audit, release gates)
- bench tiers: `qmlx bench --tier smoke|speed|harness|all` (engine-side, see Makefile help)
- `docs/`: guides + development docs

## Gotchas

- Python >=3.10 floor; Metal is the reference backend (mlx-rocm/mlx-cuda ride the same device/backend seam).
- `HF_HUB_CACHE` env var for the model cache.
- pre-commit runs ruff + mypy.
