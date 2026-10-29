# qMLX #88 post-merge adversarial review — seam migration

VERDICT: PASS (no real defects; 3 class-m style findings, none blocking; no FIX-OWED)

webdrop: https://drop.fleet.example/c3b65e24d43c079b4e2ebb8d.md (TTL 30d)

Squash commit 63390dd29bec0344c6cc1d8d4fa7e388b874e1e2 (parent ca06558), verified on
`remotes/gh/dev` tip. Read-only: no commits, no issue changes, no pushes.

## Gate 1 — Completeness (residual direct backend usage): PASS

Grep for `mx.metal.* / mx.get_active_memory / mx.get_peak_memory /
mx.get_cache_memory / mx.set_memory_limit / mx.set_cache_limit / device_info /
sysctl hw.memsize` outside `vllm_mlx/device/` returns exactly 3 hits, all
matching the PR body's "Legitimate exceptions" list:

- `vllm_mlx/runtime/diffusion_lane.py:185` — docstring mention only (not a call)
- `vllm_mlx/optimizations.py:74` — docstring mention only (not a call)
- `vllm_mlx/optimizations.py:96` — `sysctl -n hw.memsize` subprocess: documented
  fallback for hosts where no GPU backend resolves (PR body lists it)

Broader sweep of the 11 migrated modules: remaining `mx.*` calls are
compute/evaluator functions only (`mx.eval`, `mx.array`, `mx.async_eval`,
`mx.clear_cache`, `mx.contiguous`, `mx.stream`, `mx.default_stream`,
`mx.default_device`, `mx.new_thread_local_stream` in `_mlx_compat`) — none in
the gate's target list. No call site bypasses the seam for the listed
quantities.

## Gate 2 — No behaviour change: PASS

Walked all 18 files hunk-by-hunk (+420/-209). Every value source is 1:1 parity,
verified in `vllm_mlx/device/metal.py` (docstring table + code):
`recommended_working_set_bytes()` = the exact two-level
`max_recommended_working_set_size`/`memory_size` fallback; `memory_stats()` = the
same three `mx.get_*` calls; `set_memory_limit`/`set_cache_limit` = the same
`mx` calls; `device_name()` = same key-dig; `capabilities().flash_attention` =
same `hasattr` check; `supports_secondary_streams()` = same one-time probe,
cached.

- Gate semantics preserved: `get_backend() is not None` ⇔
  `mx.metal.is_available()` (verified in `backend.py:_try_builtin_metal`:
  `None` when mlx missing or Metal unavailable). All migrated gates
  (`batched.py:_set_metal_limits`, `engine_core.py` pressure threshold + check,
  `scheduler.py` cap resolution / active read / periodic log / get_stats) keep
  the same gate and the same values.
- D-METAL-CAP admission semantics UNCHANGED: only the value source moved to the
  seam (`_resolve_metal_cap_bytes`, `_current_metal_active_bytes`); the
  admission check, backpressure path, and the `num_metal_cap_violations` /
  `qmlx_metal_cap_violations_total` counter names are untouched (verified
  consistent between `scheduler.py` and `routes/metrics.py`).
- `ubc_evict` routed through the backend capability, not `sys.platform`:
  `eviction_supported()` → `backend.capabilities().os_memory_eviction`; on Metal
  that flag is `sys.platform == "darwin"` evaluated at call time (lazy, so
  monkeypatched `sys.platform` is honored — pinned by the new
  `test_os_memory_eviction_capability`); the no-backend fallback is the same
  raw check. Metal behavior unchanged.
- `_mlx_compat.py`: on the default device the backend-declared answer is the
  same one-time hardware probe (cached in the backend); the re-entrancy guard
  is required (the backend probe re-enters the patched symbol) and correct
  (inner call returns the raw stream; no double probe, no recursion);
  non-default devices and no-backend builds fall back to the original local
  probe, unchanged. `get_backend()` at install time is safe without mlx
  (returns `None`).
- `get_system_memory_gb`: on Apple Silicon the seam's `total_memory_bytes()`
  (unified memory) equals `sysctl hw.memsize` (same physical pool); sysctl
  kept as fallback; ordering change is intentional for ported builds (PR body).
- `detect_hardware`: non-Metal host — old `mx.device_info()` raised → except →
  `HardwareInfo("Unknown", ...)`; new: backend `None` → empty name → no profile
  match → same `HardwareInfo("Unknown", ...)` (only the warning log differs).
- One error-path change, strictly more graceful (n1 below):
  `get_optimization_status()` on a no-backend host now returns zeros where the
  old code raised (no try/except around the raw `mx` calls). Consistent with
  the seam's documented "None = no GPU backend" semantics.
- Log text changes (Metal → Device, added `backend=`/`enforcement=` fields) and
  comment/docstring updates only. No logic, default, or error-path change
  beyond n1.

## Gate 3 — Tests (CPU host, venv /srv/fleetwork/tmp-qmlx/venv): PASS

Baseline run at ca06558 first, then the same suites at 63390dd:

| suite | base ca06558 | head 63390dd | delta |
|---|---|---|---|
| test_device_backend + test_engine_step_thread + test_memory_stability + test_metrics_route + test_routes | 3 failed, 127 passed, 1 skipped | 3 failed, 128 passed, 1 skipped | +1 = new `test_os_memory_eviction_capability` |
| test_healthz_perf_budget + test_honest_metrics + test_kv_cache_dtype_metrics + test_optimizations + test_streaming_detokenizer(+bpe) + test_tokenizer_gemma4_hybrid + test_metal_cap_enforcement | 3 failed, 118 passed, 1 deselected | 3 failed, 118 passed, 1 deselected | identical |
| full unit suite (dev_test.py unit scope) | 107 FAILED lines | 107 FAILED lines | sorted FAILED sets byte-identical |

The 6 targeted failures are the same 3 pre-existing failures the PR body lists
(`test_no_eager_eval_in_extraction_path`,
`test_retrieve_known_alias_surfaces_recommended_sampling`,
`test_diffusion_alias_keeps_text_diffusion_modality`,
`test_route_renders_kv_restore_hit_miss`, `test_kv_restore_counter_sticky`,
`test_tokenizer_wrapper_has_optimized_detokenizer`) — present at base.
The 16 pytest errors are all in
`tests/parsers/regressions/test_issue_513_harmony_streamable_parser.py` and
reproduce identically at base (pre-existing host issue). 0 NEW failures.

Lint: `ruff check` on all 18 touched files — All checks passed.
`ruff format --check`: only `vllm_mlx/cli.py` would be reformatted, and it is
unformatted at base ca06558 too (pre-existing, PR body says so).

## Gate 4 — Doc style on touched prose: m findings (non-blocking)

- m1: em dashes in ~20 lines of NEW prose (plus ~8 carried over from
  pre-existing lines that were re-wrapped), e.g.
  `vllm_mlx/device/backend.py:86` ("``os_memory_eviction`` — the backend's OS
  can actively free"), `vllm_mlx/device/metal.py:149` ("at call time — UBC is
  a macOS property"), `vllm_mlx/runtime/ubc_evict.py:87` ("capability — routed
  through the seam"). Calibration: the base tree already carries 4328
  em-dash lines, so the prose follows the repo's established convention; the
  gate's letter is violated, the codebase norm is followed.
- m2: American spelling "behavior" in 2 new-prose lines:
  `vllm_mlx/runtime/ubc_evict.py` (`eviction_supported` docstring: "so behavior
  is unchanged") and `vllm_mlx/utils/tokenizer.py` (`_post_load_ubc_evict`
  docstring, same phrase). Repo base is mixed (82 "behavior" vs 79
  "behaviour"); gate specifies British English.
- m3: contraction "don't" retained in a modified comment line
  (`vllm_mlx/utils/tokenizer.py` — "callers on backends without the capability
  don't pay for path resolution"); the wording pre-dates the PR.
- No AI tells found (no delve/moreover/furthermore/leverage/seamless/crucial/
  pivotal/it-is-worth-noting in added prose). ASD-STE100: added prose is
  simple and technical; no idioms beyond the above.

## Gate 5 — User-visible metal_* → device_* rename: PASS

- `vllm_mlx/routes/health.py`: `/v1/status` block `metal` → `device`
  (lines 296-299); `metal_active_bytes` → `device_active_bytes`
  (line 437).
- `vllm_mlx/routes/metrics.py`: `qmlx_metal_{active,peak,cache}_memory_bytes`
  → `qmlx_device_*` (lines 1221-1232).
- `vllm_mlx/_ui.py`: banner rows `metal_active_gb`/`metal_peak_gb` →
  `device_active_gb`/`device_peak_gb` (lines 336, 377-378), reads the
  `device` block (line 356).
- Residual `metal` in the three files: comments only (health.py:435,
  metrics.py:478/1216) + the `qmlx_metal_cap_violations_total` /
  `num_metal_cap_violations` exception (PR body lists it; verified consistent
  between `scheduler.py:2715/4198/6739` and `metrics.py:1469/1481`).
- No `metal_active|metal_peak|metal_cache` user-visible fields remain anywhere
  in `vllm_mlx/` (only internal names `_compute_metal_cache_limit`,
  `_current_metal_active_bytes` — PR body lists them as kept).

## Findings

- m1 (style): em dashes in new prose across `device/backend.py`,
  `device/metal.py`, `_mlx_compat.py`, `_ui.py`, `engine/batched.py`,
  `optimizations.py`, `routes/health.py`, `routes/metrics.py`,
  `runtime/ubc_evict.py`, `scheduler.py`, `utils/tokenizer.py` — consistent
  with the repo's existing 4328-line em-dash convention.
- m2 (style): "behavior" (US) in `vllm_mlx/runtime/ubc_evict.py`
  (`eviction_supported` docstring) and `vllm_mlx/utils/tokenizer.py`
  (`_post_load_ubc_evict` docstring).
- m3 (style): contraction "don't" in modified comment,
  `vllm_mlx/utils/tokenizer.py` (pre-existing wording).
- n1 (note): `vllm_mlx/optimizations.py:get_optimization_status` — no-backend
  host now returns zeros where the old code raised (no try/except around the
  raw `mx` calls). More graceful; consistent with seam "None = no GPU
  backend" semantics. Not a defect.
- n2 (note): `vllm_mlx/engine_core.py:73` — worker-thread stream setup
  `mx.default_stream(mx.default_device())` still calls `mx` directly
  (pre-existing, untouched by this PR, not in the gate's target list; the
  seam exposes `default_stream()` — candidate for a later item).
- n3 (note): `vllm_mlx/cli.py:2699-2700` — comment re-wrap leaves "The"
  stranded at end of line ("only treats as a / guideline. The"); cosmetic;
  cli.py is unformatted at base (ruff).

No class-B (blocking) findings. No FIX-OWED: no real defect found; the commit
on dev is verified as shipped.

## UNVERIFIED

- Real Apple Silicon / Metal runtime behavior (CPU host: `mx.metal`
  unavailable, so the seam resolves `None` here; parity verified by code walk
  + `tests/test_device_backend.py` conformance, not on real hardware).
- Ported backends (mlx-rocm / mlx-cuda) do not exist yet — their
  `os_memory_eviction` / stream-declaration paths are untested here by
  design.
- The PR body's full-suite numbers (102 failed at 13309fe) differ from my
  baseline (107 at ca06558) — host-state drift between runs; my own
  base-vs-head comparison (identical FAILED sets) is authoritative.
- `qmlx serve` end-to-end on a Metal host (boot limits log line,
  `/v1/status` + `/metrics` live output) — not runnable on this CPU host.
