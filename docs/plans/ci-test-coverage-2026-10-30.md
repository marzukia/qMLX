# CI test-coverage audit — 2026-10-30

Base: `dev` @ 568c2e8 (branch `ci/dev-preview-gate`). Companion to the CI hardening PR:
dev is the integration branch; its CI gate now previews production (`on.push/pull_request
branches: [main, dev]`).

## Method

- Clean `uv` venv, CPython 3.11, exact `test-matrix` dep set (no qmlx, no mlx):
  `pytest pytest-asyncio pytest-cov pydantic fastapi jsonschema httpx psutil transformers requests`
- One batched `pytest` run over all 292 no-module-level-mlx files with `--junitxml` + `-rA`,
  `--continue-on-collection-errors`, and the CI `-k "not Integration and not InjectJson and not
  TestMLXMultimodalLMCache"` filter (pytest.ini addopts `-m "not slow and not integration"` applied).
- Per-file status parsed from the per-test summary; reasons from junit failure messages + `--tb=line`.
- AS-parity check: `pip install -e ".[vision]"` venv confirmed the mlx 0.32.4 **Linux** wheel is
  broken (`ImportError: libmlx.so` — fleet memory 177), so AS wiring is a **static import-resolution
  check** against the current tree (all 148 candidates pass: every `vllm_mlx.*` module and symbol
  exists). The **first dev-targeted CI run adjudicates** the AS list on real Apple Silicon.

## Census (303 `test_*.py` files)

| Bucket | Files | Meaning |
|---|---|---|
| Linux-wired (`test-matrix`) | 138 | green under CI-parity deps (112 files with passing tests, 26 files fully skipped/deselected under CI filters) |
| AS-wired (`test-apple-silicon`) | 148 | 11 module-level mlx imports + 78 transitive-mlx collection errors + 61 failing on Linux only via mlx/mlx_lm (or AS-provided deps uvicorn/jinja2) at request time |
| Deleted (dead) | 3 | import/feature target removed from the tree — see below |
| Failed-not-wired | 14 | fail for env/dep/pre-existing reasons on **both** jobs — see below |

138 + 148 + 3 + 14 = 303 (clean partition; no file is on both lists).

Pre-wiring, `test-matrix` covered 20 files and `test-apple-silicon` 11. The old Linux list
was RED on exactly one file: `tests/test_api_utils.py` (4 tests fail — `No module named 'mlx'`
at request time on Linux); it is now in the AS list and the Linux job is green end-to-end
(2969 passed, 0 failed). The `ruff check` and `ruff format --check` gates were also RED on
dev before this PR (see Gates).

## Dead tests removed (hard evidence)

| File | Evidence | Tests |
|---|---|---|
| `tests/test_schema_v2_accepts_v1.py` | `community-benchmarks/` removed in commit 4207fe4 (*Remove upstream-project cruft: community-benchmarks corpus…*); every test loads `community-benchmarks/schema.json` / iterates `community-benchmarks/submissions/` | 2 |
| `tests/test_schema_v2_accepts_minimal_v2.py` | same — schema.json + submissions/ absent from the tree | 4 |
| `tests/test_schema_v2_rejects_partial_harness.py` | same | 7 |

Not deleted (mixed files — reported, follow-up needed):

- `tests/test_parent_watchdog.py` — 3 spawner tests dead (`vllm_mlx/share/` and `vllm_mlx/bench/`
  removed in 5970c5b *Qwen-only simplification phase 2*); 36/42 tests pass. Keep file, prune the 3.
- `tests/test_tool_call_streaming_parity.py` — 3 parametrized cases dead (parsers `hy_v3`,
  `ui_tars`, `qwen3_coder_xml` gone from the registry in 5970c5b; docstring-only references remain
  in `vllm_mlx/tool_parsers/__init__.py`); 8/14 tests pass.
- `SimpleEngine` (PR #156): `test_simple_engine.py` / `test_llm.py` / `test_deltanet_snapshot.py`
  already gone; remaining `SimpleEngine` mentions in `tests/` are docstring-only.

## Failed-not-wired (14) — reported, NOT deleted

| File | One-line reason |
|---|---|
| `tests/test_aliases_contract.py` | 2 tests fail on dev: qwen3.5-122b-oq4-mtp is_hybrid=True but supports_spec_decode=True (mutually-exclusive invariant broken in alias data) |
| `tests/test_event_loop.py` | module-level import aiohttp — absent from both CI matrix deps and AS -e .[vision] deps |
| `tests/test_kv_cache_dtype_cli.py` | 2 tests fail on dev: expected actionable conflict error, got raw traceback (CLI behavior mismatch) |
| `tests/test_model_aliases.py` | 2 tests fail on dev: llama3-3b-4bit HF path expectation stale (alias drift) |
| `tests/test_model_profiles_ssot.py` | 4 tests fail on dev: 16 aliases vs expected 15 (table grew, expectation stale) |
| `tests/test_no_out_of_band_routing.py` | 4 tests fail on dev: QMLX_PLAIN missing from ALLOWED_QMLX_ENV_VARS; _is_mllm written outside __init__ |
| `tests/test_parent_watchdog.py` | 6 of 42 tests fail: vllm_mlx/share/ and vllm_mlx/bench/ removed in 5970c5b (FileNotFoundError); 3 spawner tests dead |
| `tests/test_prompt_lookup_bench.py` | module-level import aiohttp — absent from both CI matrix deps and AS -e .[vision] deps |
| `tests/test_serve_host_loopback_default.py` | 4 tests fail on dev: collision message must name host 127.0.0.1, CLI prints 0.0.0.0 |
| `tests/test_serve_live_pty.py` | 2 tests fail: PTY capture never reached the warming-up strip (needs live serve / warm model) |
| `tests/test_tool_call_e2e.py` | 10 tests fail on dev: Expected text response / assert None is not None (no missing dep) |
| `tests/test_tool_call_streaming_parity.py` | 6 of 14 tests fail: tool parsers hy_v3/ui_tars/qwen3_coder_xml removed in Qwen-only strip 5970c5b (KeyError: not in registry); 3 tests dead |
| `tests/test_ubc_evict.py` | 4 tests fail on dev: assert 0 == 1 (UBC evict logic regression) |
| `tests/test_vision_extra_install.py` | 2 of 14 tests fail on dev: README no longer documents pip install qmlx[vision] |

## Pre-existing test failures (not CI's fault — dev-branch state)

The 14 not-wired files above are all pre-existing dev-branch issues (stale expectations,
removed-feature residue, or deps absent from both jobs' dep sets). None of the 138 Linux-wired
files has a failing test.

## Gates (run locally on the new lists, 2026-10-30)

- `ruff check vllm_mlx/ tests/` — **PASS** (after fixing 1 pre-existing I001 in `tests/test_cli_argcomplete.py` — dev was RED on this gate)
- `ruff format --check vllm_mlx/ tests/` — **PASS** (after `ruff format` on 13 pre-existing unformatted files — dev was RED on this gate; repo formatted with pre-commit ruff v0.1.9, CI installs latest ruff 0.17.2)
- Full new Linux list (138 files, CI-parity venv): **PASS** — 2969 passed, 125 skipped, 147 deselected, 6 xfailed, 0 failed
- `python scripts/audit_cli_config_fidelity.py` — **PASS** (CLI ↔ Config fidelity: OK)

## Per-file results

Status: `linux-wired` = in the new `test-matrix` list (green). `AS-wired` = in the new
`test-apple-silicon` list (static import-resolved; first dev CI run adjudicates). `not-wired` =
fails on both jobs. `DELETED` = dead, removed.

| File | Status | Result / reason |
|---|---|---|
| `tests/integrations/test_agents_matrix.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/integrations/test_anthropic_sdk.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/integrations/test_frameworks_matrix.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/integrations/test_hermes.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/integrations/test_langchain.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/integrations/test_librechat_docker.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/integrations/test_openwebui.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/integrations/test_pydantic_ai_full.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/integrations/test_smolagents_full.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/parsers/regressions/test_issue_447_gemma4_thought_marker_leak.py` | linux-wired | 14 tests passed |
| `tests/parsers/regressions/test_issue_448_hermes_function_prefix_leak.py` | linux-wired | 36 tests passed |
| `tests/parsers/regressions/test_issue_455_harmony_commentary_tool_channel.py` | linux-wired | 6 tests passed |
| `tests/parsers/regressions/test_issue_468_tool_choice_required_harmony_compound.py` | linux-wired | 0 tests passed |
| `tests/parsers/regressions/test_issue_513_harmony_streamable_parser.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/parsers/test_infra_smoke.py` | linux-wired | 68 tests passed |
| `tests/test_447_stream_tool_choice_auto.py` | AS-wired | 4 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_agent_query_context_skip.py` | linux-wired | 24 tests passed |
| `tests/test_agent_test_runner_refresh_config.py` | linux-wired | 4 tests passed |
| `tests/test_alias_hybrid_classification.py` | linux-wired | 98 tests passed |
| `tests/test_alias_recommended_sampling.py` | linux-wired | 18 tests passed |
| `tests/test_aliases_contract.py` | not-wired | 2 tests fail on dev: qwen3.5-122b-oq4-mtp is_hybrid=True but supports_spec_decode=True (mutually-exclusive invariant broken in alias data) |
| `tests/test_anthropic_adapter.py` | linux-wired | 160 tests passed |
| `tests/test_anthropic_models.py` | linux-wired | 70 tests passed |
| `tests/test_anthropic_output_config.py` | AS-wired | 10 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_anthropic_route_auth.py` | AS-wired | 16 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_anthropic_route_streaming.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_anthropic_route_thinking_gate.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_anthropic_spec_polish_bundle.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_anthropic_stop_sequences.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_anthropic_stream_finalize.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_anthropic_stream_reasoning.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_anthropic_streaming_reasoning.py` | AS-wired | 10 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_anthropic_think_leak_r12_m1b.py` | linux-wired | 66 tests passed |
| `tests/test_anthropic_tool_choice_validation.py` | linux-wired | 38 tests passed |
| `tests/test_anthropic_tool_param_validation.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_anthropic_tool_usage_d.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_anthropic_tool_validation_scope.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_api_models.py` | linux-wired | 204 tests passed |
| `tests/test_api_utils.py` | AS-wired | 4 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_api_validation_bundle.py` | AS-wired | 30 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_batched_engine_chat_template.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_batched_engine_mllm_fallback.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_batched_engine_output_router.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_batched_engine_tool_call_normalization.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_batching.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_batching_deterministic.py` | AS-wired | 20 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_bench_vs_ollama.py` | linux-wired | 126 tests passed |
| `tests/test_body_receive_timeout.py` | AS-wired | 28 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'uvicorn'"; (no message) |
| `tests/test_cache_routes.py` | AS-wired | 44 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'"; (no message) |
| `tests/test_cancelled_requests_metric.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_capabilities_field.py` | AS-wired | 26 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_casual_chat_auto_disable_thinking.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_logprobs_channel_routing.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_n_must_equal_1.py` | linux-wired | 58 tests passed |
| `tests/test_chat_route_forced_tool_prefix.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_route_forced_tool_prefix_wiring.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_route_tool_choice_enforcement.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_route_tool_tag_leak.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_streaming_guided.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_streaming_no_reasoning_alias.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_streaming_spec.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_template_kwargs.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_chat_template_marker_sanitize.py` | linux-wired | 28 tests passed |
| `tests/test_chat_template_sidecar.py` | linux-wired | 16 tests passed |
| `tests/test_chat_template_tool_call_arguments.py` | AS-wired | 48 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'jinja2' |
| `tests/test_check_gha_pinning.py` | linux-wired | 22 tests passed |
| `tests/test_check_mlx_upstream_calls.py` | linux-wired | 20 tests passed |
| `tests/test_clean_error_messages.py` | AS-wired | 22 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_cli_argcomplete.py` | linux-wired | 26 tests passed |
| `tests/test_cli_chat.py` | linux-wired | 150 tests passed |
| `tests/test_cli_config_fidelity.py` | AS-wired | 2 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'uvicorn' |
| `tests/test_cli_info.py` | linux-wired | 8 tests passed |
| `tests/test_cli_jlens.py` | linux-wired | 42 tests passed |
| `tests/test_cli_models.py` | linux-wired | 42 tests passed |
| `tests/test_cloud_router.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_codex_profile.py` | linux-wired | 14 tests passed |
| `tests/test_completions_log_redaction.py` | AS-wired | 6 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'" |
| `tests/test_completions_response_format_json_schema.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_completions_response_format_logprobs.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_completions_spec_parity.py` | AS-wired | 58 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_config_and_middleware.py` | AS-wired | 44 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_conformance.py` | linux-wired | 44 tests passed |
| `tests/test_content_block_strict_types.py` | linux-wired | 82 tests passed |
| `tests/test_context_length_exceeded.py` | AS-wired | 42 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_context_overflow_400.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_continuous_batching.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_cors_env_configurable.py` | AS-wired | 36 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'uvicorn'"; (no message) |
| `tests/test_cors_lockdown_response_shape.py` | AS-wired | 12 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'uvicorn'" |
| `tests/test_d_anthro_validation.py` | AS-wired | 14 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_ddtree_eligibility.py` | AS-wired | 2 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_ddtree_integration.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_deep_nest_dos.py` | linux-wired | 66 tests passed |
| `tests/test_deepseek_v32_indexer_gate.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_delta_kv_checkpoints.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_dense_sampler_fastpath.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_dequantize_cache.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_device_backend.py` | linux-wired | 38 tests passed |
| `tests/test_dflash_adapter_signature.py` | AS-wired | 10 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx'; (no message) |
| `tests/test_dflash_eligibility.py` | linux-wired | 42 tests passed |
| `tests/test_dflash_integration.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_dflash_spec_decode.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_diffusion_engine.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_disconnect_counter_prod_shape.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_disconnect_guard_aborts_scheduler.py` | AS-wired | 24 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_disk_kv_checkpoint.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_disk_space_check.py` | linux-wired | 16 tests passed |
| `tests/test_download_gate.py` | linux-wired | 114 tests passed |
| `tests/test_enable_thinking_prompt_accounting.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_enable_thinking_warning_header.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_engine_router_non_stream.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_engine_step_thread.py` | AS-wired | 30 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_envelope_field_extraction.py` | linux-wired | 54 tests passed |
| `tests/test_event_loop.py` | not-wired | module-level import aiohttp — absent from both CI matrix deps and AS -e .[vision] deps |
| `tests/test_exception_handlers.py` | linux-wired | 42 tests passed |
| `tests/test_generation_config_loader.py` | AS-wired | 2 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_guided.py` | linux-wired | 48 tests passed |
| `tests/test_harmony_stop_final_channel_only.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_healthz_perf_budget.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_help_fuzzy.py` | linux-wired | 6 tests passed |
| `tests/test_hermes_harness_contract.py` | linux-wired | 2 tests passed |
| `tests/test_honest_metrics.py` | AS-wired | 24 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'"; (no message); ModuleNotFoundError: No module named 'mlx' |
| `tests/test_hy3_chat_template_default.py` | linux-wired | 52 tests passed |
| `tests/test_idle_event_wakeup.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_image_aware_cache_keys.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_image_block_requires_vision_model.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_image_url_must_be_object.py` | linux-wired | 20 tests passed |
| `tests/test_internal_route_auth.py` | AS-wired | 46 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'" |
| `tests/test_kv_cache_dtype.py` | linux-wired | 52 tests passed |
| `tests/test_kv_cache_dtype_cli.py` | not-wired | 2 tests fail on dev: expected actionable conflict error, got raw traceback (CLI behavior mismatch) |
| `tests/test_kv_cache_dtype_metrics.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_kv_cache_position_ids.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_launch_cli.py` | linux-wired | 84 tests passed |
| `tests/test_legacy_functions_field.py` | linux-wired | 22 tests passed |
| `tests/test_log_namespace_rebrand.py` | linux-wired | 26 tests passed |
| `tests/test_lone_surrogate_validator.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_max_generation_tokens_ceiling.py` | linux-wired | 66 tests passed |
| `tests/test_max_tokens_resolver.py` | AS-wired | 16 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_mcp_config_discovery.py` | linux-wired | 18 tests passed |
| `tests/test_mcp_security.py` | linux-wired | 134 tests passed |
| `tests/test_media_digest.py` | linux-wired | 26 tests passed |
| `tests/test_memory_cache.py` | linux-wired | 42 tests passed |
| `tests/test_memory_capacity_check.py` | linux-wired | 22 tests passed |
| `tests/test_memory_stability.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_metal_cap_enforcement.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_metal_error_recovery.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_metrics_route.py` | AS-wired | 38 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'"; (no message) |
| `tests/test_mirror_pull.py` | linux-wired | 126 tests passed |
| `tests/test_mirror_to_r2.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_mlx_compat.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_modality_field.py` | AS-wired | 12 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_model_aliases.py` | not-wired | 2 tests fail on dev: llama3-3b-4bit HF path expectation stale (alias drift) |
| `tests/test_model_auto_config.py` | AS-wired | 10 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_model_profiles_ssot.py` | not-wired | 4 tests fail on dev: 16 aliases vs expected 15 (table grew, expectation stale) |
| `tests/test_model_registry.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_models_command_layout.py` | linux-wired | 6 tests passed |
| `tests/test_mtp_cli_wiring.py` | AS-wired | 106 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_mtp_gemma4_assistant_inject.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_mtp_inject_and_install.py` | AS-wired | 4 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_mtp_lossless.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_mtp_real_weights.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_mtp_spec_decode.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_mtp_state_release.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_mtp_stream_contract.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_mtp_tape_rollback.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_mxfp4_moe_guardrail.py` | linux-wired | 84 tests passed |
| `tests/test_native_tool_format.py` | linux-wired | 36 tests passed |
| `tests/test_no_out_of_band_routing.py` | not-wired | 4 tests fail on dev: QMLX_PLAIN missing from ALLOWED_QMLX_ENV_VARS; _is_mllm written outside __init__ |
| `tests/test_no_pydantic_error_leak.py` | linux-wired | 48 tests passed |
| `tests/test_openai_tool_choice_validation.py` | linux-wired | 64 tests passed |
| `tests/test_optimizations.py` | AS-wired | 8 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_orphan_tool_validation.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_output_router.py` | linux-wired | 106 tests passed |
| `tests/test_packaging_data_files.py` | linux-wired | 4 tests passed |
| `tests/test_paged_cache.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_paged_cache_benefits.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_paged_cache_real_inference.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_paged_cache_real_model.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_paired_compat.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_parallel_tool_calls.py` | linux-wired | 18 tests passed |
| `tests/test_param_validation_r5e.py` | linux-wired | 136 tests passed |
| `tests/test_param_validation_r7b.py` | linux-wired | 294 tests passed |
| `tests/test_parent_watchdog.py` | not-wired | 6 of 42 tests fail: vllm_mlx/share/ and vllm_mlx/bench/ removed in 5970c5b (FileNotFoundError); 3 spawner tests dead |
| `tests/test_per_request_thinking_budget.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_postprocessor.py` | linux-wired | 220 tests passed |
| `tests/test_prefix_boundary_path_parity.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_prefix_cache.py` | linux-wired | 56 tests passed |
| `tests/test_prefix_cache_persistence.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_probe_fastpath.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_prompt_lookup_bench.py` | not-wired | module-level import aiohttp — absent from both CI matrix deps and AS -e .[vision] deps |
| `tests/test_pull_summary.py` | linux-wired | 8 tests passed |
| `tests/test_r10_scrub_validation_bundle.py` | linux-wired | 114 tests passed |
| `tests/test_r12_4_strict_json_schema_constraint_matrix.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_r12_4_strict_json_schema_helpers.py` | linux-wired | 92 tests passed |
| `tests/test_r12_h06_repair_context_guard.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_r12_m3_responses_stream_leading_items_order.py` | AS-wired | 8 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_r12_reasoning_sanitizer_required.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_r15_routes_stability_batch.py` | AS-wired | 24 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_ready_banner_timing.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_reasoning_content_null_rescue.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_reasoning_effort_translation.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_reasoning_parser.py` | linux-wired | 66 tests passed |
| `tests/test_reasoning_parser_truncation_finalize.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_release_check_m3_port_thread.py` | linux-wired | 6 tests passed |
| `tests/test_release_check_random.py` | linux-wired | 14 tests passed |
| `tests/test_request.py` | linux-wired | 100 tests passed |
| `tests/test_request_body_size_limit.py` | AS-wired | 22 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'" |
| `tests/test_request_cancellation.py` | AS-wired | 18 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_response_format_json_schema_strict.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_response_format_streaming_fence_strip.py` | linux-wired | 60 tests passed |
| `tests/test_response_format_strict_types.py` | linux-wired | 64 tests passed |
| `tests/test_responses_591_followups.py` | AS-wired | 8 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_responses_adapter.py` | linux-wired | 170 tests passed |
| `tests/test_responses_budget_exhaust_streaming.py` | AS-wired | 28 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_responses_bundle.py` | AS-wired | 44 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_responses_chat_template_kwargs.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_responses_engine_failure_envelope.py` | AS-wired | 18 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_responses_input_default_type.py` | linux-wired | 10 tests passed |
| `tests/test_responses_param_validation.py` | AS-wired | 10 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_responses_route.py` | AS-wired | 66 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_responses_sse_event_order.py` | AS-wired | 8 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_rm_command.py` | linux-wired | 14 tests passed |
| `tests/test_route_engine_contract.py` | AS-wired | 18 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_routes.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_routes_cached_tokens.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_routes_models.py` | AS-wired | 6 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx'; (no message) |
| `tests/test_routes_models_effective_parsers.py` | AS-wired | 24 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_rst_mid_sse_zombie_kv.py` | AS-wired | 8 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_runtime_cache_path.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_sampler_fast_path.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_sampling_param_finite_range.py` | AS-wired | 164 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'" |
| `tests/test_sampling_params_passthrough.py` | AS-wired | 10 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx'; ModuleNotFoundError: No module named 'mlx_lm' |
| `tests/test_sampling_validation.py` | AS-wired | 188 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'" |
| `tests/test_sanitize_output.py` | linux-wired | 30 tests passed |
| `tests/test_scheduler_disk_kv_hook.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_scheduler_stop_decoder_surface.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_schema_v2_accepts_minimal_v2.py` | DELETED | community-benchmarks/ corpus removed in 4207fe4 — schema.json/submissions/ absent; all tests load that path |
| `tests/test_schema_v2_accepts_v1.py` | DELETED | community-benchmarks/ corpus removed in 4207fe4 — schema.json/submissions/ absent; all tests load that path |
| `tests/test_schema_v2_rejects_partial_harness.py` | DELETED | community-benchmarks/ corpus removed in 4207fe4 — schema.json/submissions/ absent; all tests load that path |
| `tests/test_seed_reproducibility.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_serve_host_loopback_default.py` | not-wired | 4 tests fail on dev: collision message must name host 127.0.0.1, CLI prints 0.0.0.0 |
| `tests/test_serve_listen_fd.py` | AS-wired | 14 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'uvicorn'"; ModuleNotFoundError: No module named 'uvicorn' |
| `tests/test_serve_live_pty.py` | not-wired | 2 tests fail: PTY capture never reached the warming-up strip (needs live serve / warm model) |
| `tests/test_serve_port_collision.py` | AS-wired | 16 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'uvicorn';  +    where <function _port_is_busy at 0x7f810d126520> = cli._port_is_busy |
| `tests/test_server.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_server_api_key_env_fallback.py` | AS-wired | 18 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'uvicorn'; assert '--api-key' in 'Traceback (most recent call last):\n  File "<frozen runpy>", line 198, in _run_module_as_main\n  File "<frozen runpy>...1002/vllm |
| `tests/test_server_auth_ordering.py` | AS-wired | 22 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): ModuleNotFoundError: No module named 'mlx' |
| `tests/test_server_load_model_order.py` | AS-wired | 24 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'" |
| `tests/test_server_utils.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_signal_observability.py` | linux-wired | 20 tests passed |
| `tests/test_speculative_config.py` | linux-wired | 190 tests passed |
| `tests/test_sse_keepalive.py` | AS-wired | 24 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'" |
| `tests/test_stop_matcher_completions.py` | AS-wired | 12 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): failed on setup with "ModuleNotFoundError: No module named 'mlx'" |
| `tests/test_stop_string_enforcement.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_stream_forced_tool_reasoning.py` | linux-wired | 78 tests passed |
| `tests/test_stream_include_usage_honored.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_streaming.py` | linux-wired | 52 tests passed |
| `tests/test_streaming_detokenizer.py` | AS-wired | module-level import mlx/mlx_lm (Linux cannot collect) |
| `tests/test_streaming_detokenizer_bpe.py` | linux-wired | 32 tests passed |
| `tests/test_streaming_json_encoder.py` | linux-wired | 30 tests passed |
| `tests/test_streaming_latency.py` | linux-wired | 2 tests passed |
| `tests/test_streaming_newlines.py` | linux-wired | 14 tests passed |
| `tests/test_streaming_pipeline_integration.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_streaming_reasoning_split_r8c.py` | linux-wired | 20 tests passed |
| `tests/test_streaming_reasoning_with_structured_output.py` | linux-wired | 12 tests passed |
| `tests/test_streaming_simulator.py` | linux-wired | 74 tests passed |
| `tests/test_streaming_think_router.py` | linux-wired | 36 tests passed |
| `tests/test_streaming_tool_filter.py` | linux-wired | 56 tests passed |
| `tests/test_stress_test_stream_usage.py` | linux-wired | 4 tests passed |
| `tests/test_structured_output.py` | linux-wired | 50 tests passed |
| `tests/test_submit_gitignore.py` | linux-wired | 2 tests passed |
| `tests/test_suffix_bench_methodology.py` | linux-wired | 26 tests passed |
| `tests/test_suffix_decoding.py` | AS-wired | 12 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_suffix_decoding_tier.py` | linux-wired | 56 tests passed |
| `tests/test_tempfile_safe.py` | linux-wired | 30 tests passed |
| `tests/test_text_only_media_gate.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_think_scaffold_roundtrip.py` | linux-wired | 20 tests passed |
| `tests/test_tmax_merge_vision.py` | linux-wired | 10 tests passed |
| `tests/test_tokenizer_gemma4_hybrid.py` | linux-wired | 32 tests passed |
| `tests/test_tool_call_e2e.py` | not-wired | 10 tests fail on dev: Expected text response / assert None is not None (no missing dep) |
| `tests/test_tool_call_promotion.py` | linux-wired | 72 tests passed |
| `tests/test_tool_call_streaming_parity.py` | not-wired | 6 of 14 tests fail: tool parsers hy_v3/ui_tars/qwen3_coder_xml removed in Qwen-only strip 5970c5b (KeyError: not in registry); 3 tests dead |
| `tests/test_tool_calling.py` | linux-wired | 112 tests passed |
| `tests/test_tool_choice_enforcement.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_tool_injection.py` | linux-wired | 36 tests passed |
| `tests/test_tool_logits.py` | linux-wired | 82 tests passed |
| `tests/test_tool_logits_schema_guards.py` | linux-wired | 56 tests passed |
| `tests/test_tool_param_enforcement.py` | AS-wired | 28 tests fail on Linux (mlx/mlx_lm or AS-provided dep at request time): (no message) |
| `tests/test_tool_param_enforcement_f141a.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_tool_parser_coverage.py` | linux-wired | 10 tests passed |
| `tests/test_tool_parser_wire_formats.py` | linux-wired | 8 tests passed |
| `tests/test_tool_parsers.py` | linux-wired | 136 tests passed |
| `tests/test_tool_result_content_array.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_tools_auto_disable_thinking.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_truncation_no_synthetic_text.py` | AS-wired | collection error on Linux: mlx required transitively via vllm_mlx import closure |
| `tests/test_tui_extra_gate.py` | linux-wired | 2 tests passed |
| `tests/test_tui_workers_pin.py` | linux-wired | no tests executed under CI -k/-m filters (all skipped/deselected) — exits 0 |
| `tests/test_ubc_evict.py` | not-wired | 4 tests fail on dev: assert 0 == 1 (UBC evict logic regression) |
| `tests/test_upgrade_cli.py` | linux-wired | 8 tests passed |
| `tests/test_validate_release_subject.py` | linux-wired | 32 tests passed |
| `tests/test_version_check.py` | linux-wired | 88 tests passed |
| `tests/test_vibethinker_function_xml.py` | linux-wired | 26 tests passed |
| `tests/test_vision_extra_install.py` | not-wired | 2 of 14 tests fail on dev: README no longer documents pip install qmlx[vision] |
| `tests/test_xfail_audit.py` | linux-wired | 2 tests passed |
