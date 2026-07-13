# harness/ — removed tooling

The qMLX "doctor" regression harness this directory used to document
(`qmlx doctor smoke|check|full|benchmark`, `qmlx bench --tier ...`, and the
`vllm_mlx/doctor/` code behind them) was removed in the Qwen-only
simplification (#20). There is no `qmlx doctor` or `qmlx bench` command anymore.

The gates that remain:

- `make release-smoke` (== `python scripts/release_smoke.py`) — the clean-room
  install + import gate, and the one mandatory pre-publish check. See
  `docs/development/releasing.md`.
- `ci.yml` — lint (ruff + audit) and the test matrix, on every PR and push.
- `make smoke` / `make stress` / `make soak` — the dev test tiers that still
  exist (see the Makefile).

The baseline, threshold, and scorecard files still under `harness/` are
leftovers from the old doctor and are not read by anything current.
