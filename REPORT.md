# Fix round 2 — PR #95 (feat/max-context-length): no-mlx matrix breakage

**Verdict: FIXED (option c) — the file's "runs without mlx" docstring claim is now true; CI-parity run shows 2 failed (pre-existing baseline) / 798 passed with ZERO failures from test_max_context_length.py.**

## What changed

`tests/test_max_context_length.py` only (no ci.yml change needed — the file stays
in the no-mlx matrix list):

- Added a module-scoped, autouse `mlx_stubs` fixture that installs a minimal
  `mlx` / `mlx_lm` surface into `sys.modules` for the module's tests and
  restores the previous `sys.modules` state on teardown — the merged
  `tests/test_diffusion_shared_mode.py` (PR #97) save/restore pattern, no
  conftest pollution.
- The stub covers exactly the module-level import chain the tests touch:
  `mlx` / `mlx.core` (the `_mlx_compat` shim is a no-op since
  `mx.new_thread_local_stream` is absent), `mlx_lm.generate.BatchGenerator`,
  `mlx_lm.sample_utils.{make_logits_processors,make_sampler}`,
  `mlx_lm.tokenizer_utils.NaiveStreamingDetokenizer`, and
  `mlx_lm.models.cache.MambaCache` (bound at module level by
  `vllm_mlx/utils/mamba_cache.py`, imported by the scheduler).
- Each stub module carries a real `importlib.machinery.ModuleSpec` so
  `importlib.util.find_spec` (transformers' `is_mlx_available` probe, reached
  via the `is_mllm_model` local-probe path in the real-`BatchedEngine` tests)
  does not raise `ValueError: mlx.__spec__ is None`.
- Docstring updated: states the no-mlx claim is realized via the saved/restored
  `sys.modules` stubs (real mlx wins again on Apple-Silicon hosts after
  teardown).

All 17 tests are unchanged in substance; the 3 real-`BatchedEngine` tests
(cap=8000 + 400-gate pins) stay in the no-mlx matrix.

## Verification numbers

1. **Faithful CI-parity venv** (fresh `python3 -m venv`, exact ci.yml pip list:
   `pytest pytest-asyncio pytest-cov pydantic fastapi jsonschema httpx psutil
   transformers requests`, no mlx, no qmlx; ran the full ci.yml test-matrix
   list + the `-k "not Integration and not InjectJson and not TestMLXMultimodalLMCache"`
   filter + cov flags):

   **`2 failed, 798 passed, 13 skipped, 11 deselected`**
   - The 2 failures are exactly the pre-existing baseline
     (`TestIsMllmModelWeightsPresenceOverride::test_single_file_no_multimodal_tensors_is_text_only`,
     `::test_single_file_with_vision_tensors_is_vlm` — `ModuleNotFoundError: No module named 'mlx'`,
     identical at dev; dev baseline: 2 failed / 781 passed).
   - **0 failures from `tests/test_max_context_length.py`** (was 15/17 before the fix).
   - 798 = dev's 781 + the 17 tests this file adds to the list.

2. **`test_max_context_length.py` alone in the same no-mlx venv** (the
   import-satisfiable check — the fixture itself is the mlx stub):
   **17 passed** (both before and after `ruff format`).

3. **Ruff** (`uv run ruff check vllm_mlx/ tests/` — CI scope): output identical
   to dev (1 pre-existing fixable finding, same on both; diff of the two runs
   is empty → **zero new findings**). `uv run ruff check tests/test_max_context_length.py`:
   all checks passed. `uv run ruff format --check tests/test_max_context_length.py`:
   clean.

## Notes

- Re-verified live at start: dev tip still `568c2e8` (branch base), so no rebase
  was needed; fix commits directly on top of `ec38a8b`.
