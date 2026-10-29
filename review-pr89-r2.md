# Re-review: PR #89 — docs staleness sweep (branch monky/qmlx-docs-staleness @ 02a1bc6b)

Reviewer: cross-agent gate (fresh session, different agent than t876/monky). Read-only on the
PR branch. True PR base = dev @ 63390dd; PR diff is docs-only: 20 files, +143/−246
(fix rounds added +7/−5 vs the original +136/−241). Fix commits since the FAIL:
65c4acf (round 1: parser table, auto-detect claims, em dashes), c5cbb5a (round 2: dead
issue links, timeout default, roadmap anchor), 02a1bc6b (pr_merge_sop #320 upstream link).

## VERDICT: PASS

All 6 defects from the t876 FAIL round (plus the follow-up #320 link) are fixed and
independently verified against the live branch. Fresh adversarial pass over the full
diff found no new factual breakage from the fix rounds; two minor pre-existing
non-gating notes (one grammar slip, one cosmetic inconsistency).

---

## Check A — the 6 FAIL-round defects, each re-verified at 02a1bc6b: ALL FIXED

1. **docs/benchmarks/llm.md:81 — issue #472**: now
   `https://github.com/raullenchai/Rapid-MLX/issues/472` → **200** (curl -L).
   `marzukia/qMLX/issues/472` still 404, confirming the re-point was necessary. FIXED.
2. **docs/guides/codex-cli.md:185 — issue #549**: now
   `https://github.com/raullenchai/Rapid-MLX/issues/549` → **200**; qMLX #549 404. FIXED.
3. **docs/guides/ai-clients.md:94 — #47#issuecomment-4410012225**: now upstream.
   Verified via API: `GET /repos/raullenchai/Rapid-MLX/issues/comments/4410012225`
   → 200, `html_url` = exactly the linked URL. FIXED.
4. **docs/guides/ai-clients.md:150 — #47 thread**: now upstream
   `raullenchai/Rapid-MLX/issues/47` → 200; API title = "Test Rapid-MLX with more AI
   clients and report results" — the correct AI-client compatibility thread (qMLX #47
   is the unrelated MTP memory-bloat issue). FIXED.
5. **--timeout default (server.md:34 + configuration.md:21)**: both now document
   **1800**; live default is 1800.0 (cli.py:6036-6039, help text "default: 1800 =
   30 min"). FIXED. Bonus: the configuration.md `--continuous-batching` row changed to
   "Accepted for back-compat (continuous batching is always on) | `true`" — verified
   accurate (cli.py:5821-5825: `action="store_true", default=True`).
6. **docs/inference-efficiency-papers-2025.md:38 — stale "On roadmap (#3)"**: now
   "[Cut from the roadmap](../README.md) in 4996590 -- not started". Verified:
   (a) `../README.md` resolves (README.md exists at repo root); (b) README has no
   DuoAttention mention (only roadmap line: "Windowed attention … is on the
   roadmap"); (c) commit 4996590 ("docs: restructure README — Quick Start first,
   trim 42%") diff shows the deletion of the numbered row "| 3 | DuoAttention …"
   and its commit message says "removed stale paper-only items (KVSplit,
   DuoAttention, …)". Claim accurate. FIXED.
7. **docs/development/pr_merge_sop.md:255 — issue #320** (follow-up fix 02a1bc6b):
   now `https://github.com/raullenchai/Rapid-MLX/issues/320` → **200**; qMLX #320
   404; text adds "(upstream Rapid-MLX issue, never migrated)". FIXED.

## Check B — the two wrong "verified accurate" claims: FIXED

- **Parser table (tool-calling.md:55-58)**: was a single `hermes` row with a literal
  newline inside the Format cell (broken markdown table) and no `qwen` row. Now both
  rows present, each a single well-formed line (verified with repr()), Format cells
  clean. Aliases match the live registry exactly: hermes_tool_parser.py:86 registers
  ["hermes","nous","qwen3_coder"]; qwen_tool_parser.py:28 registers
  ["qwen","qwen3","qwen3_xml"] = the 6 names listed. FIXED.
- **Auto-detect wording (tool-calling.md:49-52, 90-94; quickstart.md:102-105)**: now
  "the `hermes` parser for Qwen3.x, Mistral/Devstral and most other chat models
  (none for Qwen3.6). The `qwen` parser is explicit-only / not auto-assigned".
  Verified against live code: `grep 'tool_call_parser="qwen"' model_auto_config.py`
  → no matches (every auto-assigned parser is `hermes` or `None`); all 18 Qwen3.6
  aliases in aliases.json have `tool_call_parser: null`; magistral/mistral|devstral/
  gemma (2/3) patterns all assign `hermes`; GLM-4.7/GPT-OSS/Llama correctly dropped
  from the quickstart family list (no GLM/Llama aliases; all GPT-OSS aliases
  `tool_call_parser: None`). FIXED.

## Check C — fresh adversarial pass over the full diff (20 files, docs-only)

- **All 47 unique github.com URLs in docs/ → 200** (curl -L), incl. the 5 re-pointed
  upstream issue links, the API-verified comment link, and the retained
  telemetry-golden-profile.md history link (Rapid-MLX #236 → 200).
- **All relative links in docs/ resolve** (scripted check, .md and path targets).
- **Remaining `marzukia/qMLX` refs are all legitimate**: generic `/issues` page links
  (benchmarks/README.md:58, contributing.md:116), clone URLs, `gh --repo` commands.
- **No stale script refs remain**: `benchmark_engines` / `bench_all_models` → 0 hits;
  `scripts/bench_vs_ollama.py` (supports `--runs`, `--output-dir`, no `--engine` flag
  — matches all three doc usages), `examples/benchmark_all_models.py`,
  `tests/test_continuous_batching.py` all exist.
- **architecture.md tree**: routes/ has models.py, responses.py, metrics.py,
  mcp_routes.py (embeddings.py/audio.py gone — correctly dropped); device/ seam
  present; reasoning/ = base + harmony_stop + qwen3_parser (registry registers only
  `qwen3`, __init__.py:85 — "qwen3 registered; base + harmony stop helpers"
  accurate); 10 agent profile YAMLs; doctor/ gone; scripts tree matches
  (dev_test.py has lint/audit/unit/stress/soak/cross-model tiers;
  cross_model_stress.py file gone — its dev_test.py reference is code-side
  staleness, not a doc error); "2100+" tests understates the live 5539 test
  functions (true lower bound).
- **development/architecture.md**: api/guided.py json_schema_to_pydantic (line 41),
  anthropic_adapter.py, responses_adapter.py, mcp/types.py, runtime/{model_registry,
  disk_kv_checkpoint}.py all exist; audio/, reasoning_parsers/, mllm_scheduler.py,
  mllm_batch_generator.py all gone; `from vllm_mlx.optimizations import
  detect_hardware` is the live import (HardwareInfo has chip_name/total_memory_gb).
- **pr_merge_sop.md**: dropped `--ignore`s (test_mllm.py, test_mllm_cache.py,
  test_mllm_continuous_batching.py, test_video.py) all absent from tests/; retained
  ignores (tests/integrations, tests/test_event_loop.py) exist.
- **installation.md**: tap/trust brew section removed cleanly (only remaining brew
  ref is core `brew install python@3.12`); `uv tool upgrade qmlx-serve` matches the
  dist name (pyproject `name = "qmlx-serve"`); extras table (vision/chat/guided/all)
  in pyproject.toml.
- **quickstart.md**: `qmlx-chat` console script exists (pyproject [chat] extra);
  "on by default" accurate (auto-config sets both --tool-call-parser and
  --enable-auto-tool-choice).
- **models.md**: "Qwen3.6 | 27B, 35B-A3B" — "Coder" correctly removed (no Qwen3.6
  Coder alias exists); family/size/quant lists consistent with aliases.json.
- **tier0 plan baseline**: evals/results/baseline_benchmark.json exists; content
  matches the parenthetical (model = lmstudio-community/MiniMax-M2.5-MLX-4bit,
  timestamp = 2026-02-24 20:21:20).
- **server.md / cli.md / configuration.md**: `--reasoning-parser` now lists only
  `qwen3` (matches live registry); `--tool-call-parser` lists exactly the 6
  registered names; `reasoning_content` matches the live API (routes/chat.py:3424,
  3530-3537); devstral-24b-4bit alias exists; deepseek-r1-8b/32b-4bit aliases have
  reasoning_parser: null, matching the "no registered reasoning parser for this
  alias" comments; `(#19)` refs resolve to qMLX PR #19 "Qwen-only simplification
  (phase 1): strip non-Qwen reasoning/tool parsers + audio" (closed/merged) —
  accurate; deepseek_r1 survives in code only in comments/docstrings.
- **index.md**: identity reframe consistent with the merged repositioning
  (#81 device/backend seam spec with metal/rocm/cuda BackendKind; #88 seam
  migration). No residual "vLLM-MLX" / "Qwen-specialised" / "Rapid-MLX" identity
  framing (only the retained telemetry history link + demo.cast recording asset).

### Minor findings (non-gating)

1. **Grammar slip, docs/guides/reasoning.md:138-140** (present since the base PR,
   not introduced by the fix rounds; missed by the first review):
   "The `qwen3` parser covers the same implicit-think case it was for: DeepSeek-R1
   distills on a Qwen base **emit** the same think tags and **are** served with
   `--reasoning-parser qwen3`." — two unconnected finite verbs ("distills" +
   "emit") and a number-agreement mismatch ("distills" singular vs "are" plural).
   Factual content is accurate; the sentence needs a conjunction, e.g. "…DeepSeek-R1
   distills on a Qwen base and emits the same think tags; it is served with
   `--reasoning-parser qwen3`."
2. **Cosmetic inconsistency, docs/guides/ai-clients.md:94**: the fix replaced the
   em dash with ", " while every adjacent list line still uses "—" (the fix-round
   em-dash replacements in reasoning.md/models.md are consistent within those
   files). No meaning change.
3. **Loose label, docs/development/architecture.md**: "models/ # Model package
   (registry and alias plumbing)" — vllm_mlx/models/ contains only __init__.py;
   the registry lives in runtime/model_registry.py. Not a false claim (the
   package exists and the plumbing is real), just imprecise.

None of the three affects factual accuracy, link resolution, or flag/registry
correctness; all six FAIL-round defects plus the #320 follow-up are verified fixed.

## Notes (non-gating)

- PR body says "13 docs changed"; the diff is 20 files (carried over from the first
  review — self-report inaccuracy in the PR description only).
- `scripts/dev_test.py:130` still references the removed
  `scripts/cross_model_stress.py` — code-side staleness, out of scope for a
  docs-only PR (the doc tree correctly omits the file).
