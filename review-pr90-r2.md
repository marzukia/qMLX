# PR #90 — Round 2 re-review (fix commit ab3fb4b over 5eebb39)

Fresh session, zero prior context. Base dev @ 44bff0c. Head ab3fb4bcc5eb81664a9b2028d571c78bf414e824.

## VERDICT: PASS

## Scope check (mechanical) — PASS

`git diff --name-only 5eebb39..ab3fb4b` touches exactly the three expected files and nothing else:

- `.github/workflows/ci.yml` (+1)
- `docs/development/backend_conformance.md` (+2/-3)
- `vllm_mlx/device/conformance.py` (+3/-2)

## Fix verification

### F1 — phantom `--backend NAME` doc row — FIXED
Row removed from the flag table. The table now lists only `--json FILE` and `--gaps FILE`, which matches the code: `argparse` in `run_conformance` has exactly two `add_argument` calls (conformance.py:1270, 1275). No `--backend` flag exists in code.

### F2 — doc SEAM_CONTRACT reword + memory-cap values — FIXED
- SEAM_CONTRACT row now reads: "…STRUCTURALLY SATISFIES THE BACKEND PROTOCOL (ALL REQUIRED ATTRIBUTES PRESENT, TYPE-CHECKED, NEVER-RAISE CONTRACTS HOLD). THE CAPABILITIES THE BACKEND SAYS IT HAS MATCH THE RAW API THE SUITE OBSERVES." — matches the claimed wording.
- The reword is accurate against the code: `_check_seam_contract` (conformance.py:353) does `isinstance` checks (type-checked attributes: bool `is_available`, `BackendKind`/`EnforcementMode` kinds, non-negative int caps) and never-raise contracts ("contract is a bool that never raises", conformance.py:396).
- MEMORY_CAP_ENFORCEMENT row now says behaviour is recorded as `hint` or `hard` — matches the code (`observed = "hint"` / `observed = "hard"`, conformance.py:491/497). The old phantom `ENFORCED` value is gone.

### F3 — mypy rename in run_conformance — FIXED
`import mlx.core as _mlx` with `_mlx = None` in the except, then `mx = _mlx`. The `# noqa: F811` is gone (no longer needed). mypy: error SET identical to base 44bff0c (mypy 1.8.0, pre-commit pin; see check lines) — zero new errors, no no-redef.

### F4 — tests/test_conformance.py in CI test-matrix — FIXED
`tests/test_conformance.py \` appended to the no-MLX unit list in ci.yml (after test_check_mlx_upstream_calls.py, before the flags). The file exists and passes on this no-mlx host (20 tests, all green — see check lines).

### F5 — intentionally untouched — CONFIRMED
The conformance.py diff contains only the mx import block; the runpy warning path and the `quantized_kv_roundtrip` check (conformance.py:530, 1247) are unchanged.

## Check lines (run exactly as CI, no-mlx host)

| Check | Result |
| --- | --- |
| `ruff check vllm_mlx/ tests/` | 1 error: I001 in `tests/test_cli_argcomplete.py` — **matches parity baseline** (pre-existing). PASS (at baseline) |
| `ruff format --check vllm_mlx/ tests/` | 13 files would be reformatted — **pre-existing**: file set byte-identical at 5eebb39 (verified by checkout + diff). None of the 13 are PR-touched files. See R2-F1 |
| `python scripts/audit_cli_config_fidelity.py` | `CLI ↔ Config fidelity: OK` (exit 0). PASS |
| `mypy vllm_mlx/ --ignore-missing-imports --no-error-summary` (mypy 1.8.0 = pre-commit pin) | Error set at head **identical** to base 44bff0c (2 environment-dependent errors: numpy `.pyi` "Type statement" syntax; yaml stubs not installed). Zero NEW errors, no no-redef. PASS (advisory) |
| ci.yml test-matrix unit list (incl. `tests/test_conformance.py`), `-k "not Integration and not InjectJson and not TestMLXMultimodalLMCache"`, no-mlx host | 779 passed, 3 skipped, 10 failed — the 10 failures are pre-existing host artifacts, failure set identical at 5eebb39 (759 passed there; the +20 delta is the new test_conformance.py). See R2-F2. PASS (no regression) |
| `pytest tests/test_device_backend.py` (AST import audit) | 19 passed, 1 skipped (`test_parity_against_real_mlx` — "mlx not installed (Linux CI)"). PASS |

## Findings

- **R2-F1 (info)**: `ruff format --check` fails on 13 files (test_cli_argcomplete, test_cli_models, test_ddtree_integration, test_dflash_integration, test_honest_metrics, test_models_command_layout, test_mtp_spec_decode, test_mtp_tape_rollback, test_serve_live_pty, test_tui_workers_pin, cli.py, model_aliases.py, spec_decode/mtp/generator.py). Verified the flagged set is byte-identical at 5eebb39 — pre-existing, not introduced by the fix commit, and no PR-touched file is flagged. Likely ruff-version-dependent formatting (the PR's own CI presumably passed with its pinned ruff). No action required for this PR.
- **R2-F2 (info)**: The 10 test-matrix failures on this host are `ImportError: libmlx.so: cannot open shared object file` in `TestStripBackslashBeforeUnicode` (8) and `TestIsMllmModelWeightsPresenceOverride` (2). Root cause: the skip guard is `skipif(not _MLX_AVAILABLE)` with `_MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None` — on this host the mlx wheel is installed but the Metal `libmlx.so` is absent, so the guard is fooled and the tests run. On real no-MLX CI (no mlx package at all) they skip. Pre-existing, unrelated to PR #90.

## Conclusion

All four claimed fixes (F1–F4) landed correctly and are accurate against the code; F5 is untouched as intended. Scope is clean (3 files only). No check line regressed vs base: ruff-check at parity baseline, ruff-format/mypy/test-matrix failures all verified pre-existing at 5eebb39/44bff0c.

**VERDICT: PASS**
