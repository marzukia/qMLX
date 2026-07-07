# Integration tests

End-to-end tests that exercise Rapid-MLX from a real client library.

These are **not** run as part of `pytest tests/` because they need a running
Rapid-MLX server on `http://localhost:8000` and a loaded model — the fixtures
`skip` cells when no server is reachable, so a naïve `pytest tests/` still
comes out green.

## Two matrices — 11 agents + 3 frameworks × 4 families

0.10.2 PR-2 pilot expanded the matrices to the 0.10.2 **Tier-1 four
families** (added DeepSeek V4) and the finalized **top-10** commercial /
open-source agents (three commercial-CLI cells added via docs-confirmed
BYOK routes). Both matrices share the harness in `conftest.py`:

- `test_agents_matrix.py` — **11 Tier-1 agents × 4 families** (Qwen 3.6,
  Gemma 4, DeepSeek V4, gpt-oss) = 44 cells. Each cell is a lightweight
  smoke; deep flows live in the dedicated files below.
- `test_frameworks_matrix.py` — **3 Tier-1 frameworks × 4 families** =
  12 cells.

Total: **56 cells** (up from 33 in #1030).

> **Pilot scope note.** This pilot runs the Qwen 3.6 35B-A3B-8bit
> family end-to-end and leaves Gemma 4 / DeepSeek V4 Flash / gpt-oss
> 120B for sibling PRs (see PR-2c-1 / PR-2c-2 / PR-2c-3 in the parent
> issue). All four aliases resolve correctly in
> `vllm_mlx/aliases.json` today; only the pilot family is proven with
> real inference in this PR.

Support ≡ a real integration test that boots the server + real model + real
client flow, not just a YAML profile. See `workflow.md` W3 taxonomy §B.3.

### Pre-flight verdict — commercial CLI top-10 finalization

Task #461 asked the pilot to verify five commercial CLIs against a
custom OpenAI base_url before wiring their matrix cells. Verdicts
recorded here (no CLI binaries were installed — verdict is based on
official BYOK docs review):

| CLI | Pre-flight verdict | Reason | Kept in top-10? |
|---|---|---|---|
| GitHub Copilot | **PASS** | `COPILOT_PROVIDER_BASE_URL` + `COPILOT_PROVIDER_API_KEY` env vars documented at [docs.github.com](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models) | ✅ (new cell `TestCopilot`) |
| Factory AI Droid | **PASS** | `~/.factory/settings.json` `customModels` array with `provider: generic-chat-completion-api`, docs at [docs.factory.ai](https://docs.factory.ai/cli/byok/overview) | ✅ (new cell `TestDroid`) |
| Moonshot Kimi Code | **PASS** | `~/.kimi/config.toml` provider block with `type = "openai"` + `base_url`, docs at [moonshotai.github.io](https://moonshotai.github.io/kimi-cli/en/configuration/providers.html) | ✅ (new cell `TestKimiCode`) |
| Cursor CLI | **DEFERRED** | Cursor IDE honors custom OpenAI base URL, but the CLI/agent path routes exclusively through Cursor's backend (per community forum + docs). Not integrable at custom endpoint | ❌ — fallback promoted: `qwen-code` |
| Alibaba Qoder | **DEFERRED** | Native Qoder CLI has no first-party OpenAI base_url hook — only third-party proxy wrappers (`qoder-proxy`, `qoder-cli-api`). Wire-smoke would misrepresent Qoder's native shape | ❌ — fallback promoted: `hermes-agent` |

**Final top-10** (order preserves task's ranking with fallbacks
inserted at the DEFERRED slots): `codex-cli`, `claude-code`, `opencode`,
`openhands`, `copilot`, `qwen-code` (for Cursor), `droid`, `kimi-code`,
`hermes-agent` (for Qoder), `aider`. `kilo-code` is retained from
#1030's Tier-1 list (11 cells total instead of exactly 10 — flagged
for operator scope call in the PR body).

### Tier-1 agents

| Agent | Wire | Matrix cell | Deep flow |
|---|---|---|---|
| codex-cli | `/v1/responses` | `TestCodexCLI` | (matrix only) |
| claude-code | `/v1/messages` | `TestClaudeCode` | `test_anthropic_sdk.py` |
| opencode | `/v1/chat/completions` | `TestOpenCode` (wire smoke via OpenAI SDK) | (matrix only) |
| qwen-code | `/v1/chat/completions` | `TestQwenCode` (wire smoke via OpenAI SDK) | (matrix only) |
| openhands | `/v1/chat/completions` | `TestOpenHands` (**wire smoke only** — does not exercise the real OpenHands binary / LiteLLM shim) | (Docker E2E deferred to 0.10.6 Phase 4) |
| hermes-agent | `/v1/chat/completions` | `TestHermesAgent` (wire smoke via OpenAI SDK) | `test_hermes.py` (real 62-tool E2E) |
| aider | `/v1/chat/completions` | `TestAider` (**wire smoke only** — does not exercise Aider's edit format or CLI) | `test_aider.sh` (real CLI edit-and-write) |
| kilo-code | `/v1/chat/completions` | `TestKiloCode` (wire smoke via OpenAI SDK) | (matrix only) |
| copilot | `/v1/chat/completions` | `TestCopilot` (**wire smoke only** — real CLI needs `gh auth login`, deferred) | (subprocess cell deferred) |
| droid | `/v1/chat/completions` | `TestDroid` (**wire smoke only** — real CLI needs Factory session token, deferred) | (subprocess cell deferred) |
| kimi-code | `/v1/chat/completions` | `TestKimiCode` (**wire smoke only** — real CLI needs Moonshot auth flow, deferred) | (subprocess cell deferred) |

### Tier-1 frameworks

| Framework | Wire | Matrix cell | Deep flow |
|---|---|---|---|
| LangChain (+ LangGraph) | `/v1/chat/completions` | `TestLangChain` | `test_langchain.py` |
| PydanticAI | `/v1/chat/completions` | `TestPydanticAI` | `test_pydantic_ai_full.py` |
| smolagents | `/v1/chat/completions` | `TestSmolagents` | `test_smolagents_full.py` |

## Running

Start the server first (positional model arg — never `--model`):

```bash
rapid-mlx serve qwen3.5-4b-4bit \
    --tool-call-parser hermes --enable-auto-tool-choice
```

Then run either matrix or a specific deep file. **Strict mode requires
one family shard per booted server** — the ``_guard_family_matches_server``
autouse fixture in ``conftest.py`` fails cells that ask for a family the
running server doesn't serve. In practice this means: pick the family
that matches your ``rapid-mlx serve`` alias and shard the other two into
separate server boots (or CI jobs).

```bash
# All 44 agent cells; only the family matching the running server passes,
# the other three skip (non-strict) or fail (strict). Use for local sanity;
# for CI, prefer per-family shards below.
pytest tests/integrations/test_agents_matrix.py -v

# 12-cell framework matrix (same shard rule as above)
pytest tests/integrations/test_frameworks_matrix.py -v

# Strict CI — per-family shard (this is the intended workflow: four
# CI jobs, one per family, each with its own booted server).
RAPID_MLX_MATRIX_STRICT=1 RAPID_MLX_AGENT_MATRIX_FAMILY=qwen36 \
    pytest tests/integrations/test_agents_matrix.py

# One agent's cells across all families (still shard-restricted)
pytest tests/integrations/test_agents_matrix.py -k QwenCode

# Deep flows (Python)
python3 tests/integrations/test_pydantic_ai_full.py
python3 tests/integrations/test_smolagents_full.py
python3 tests/integrations/test_langchain.py
python3 tests/integrations/test_anthropic_sdk.py
python3 tests/integrations/test_openwebui.py

# Deep flows (CLI / Docker)
bash tests/integrations/test_aider.sh
python3 tests/integrations/test_librechat_docker.py
```

## Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `RAPID_MLX_BASE_URL` | `http://localhost:8000/v1` | Where matrix clients point |
| `RAPID_MLX_AGENT_MATRIX_FAMILY` | (all) | Restrict to `qwen36` / `gemma4` / `deepseek` / `gptoss` |
| `RAPID_MLX_MATRIX_STRICT` | `0` | If `1`, missing-server → fail (default: skip) |

## Cheap-alias policy

The matrix boots the smallest available alias per family — 4B for Qwen 3.5
(3.6 has no <8B SKU), 12B for Gemma 4 (smallest text-only SKU, ~7 GB @ 4-bit),
20B for gpt-oss (no smaller SKU in the family, MXFP4-Q8 ~11 GB). The 27-35B
family flagships are reserved for the weekly Golden Path job. This keeps the
per-process resident footprint under the W5 OOM budget on M3 Ultra (operator
services baseline + matrix + Metal overhead ≤ 150 GB).

Family choice per matrix run:

| Family | Alias used | Rationale |
|---|---|---|
| Qwen 3.6 | `qwen3.5-4b-4bit` | 3.6 has no <27B SKU; 3.5-4B shares `hermes` / `qwen3` parsers |
| Gemma 4 | `gemma-4-12b-4bit` | Smallest text-only alias; ~7 GB at 4-bit |
| DeepSeek | `deepseek-r1-32b-4bit` | 0.10.2 PR-2 pilot swapped from `deepseek-v4-flash-8bit` (~155 GB weights, single-node-infeasible on 256 GB M3 Ultra + G11 100 GB floor). R1-Distill-Qwen-32B-4bit at ~16 GB stays above the "no cheap-alias" bar and exercises the same `deepseek` tool-call + `deepseek_r1` reasoning parsers V4-Flash would have. **Full DeepSeek V4 Flash Tier-1 slot tracked in follow-up issue #1041** (hardware plan needed) |
| gpt-oss | `gpt-oss-20b-mxfp4-q8` | Smallest gpt-oss; ~11 GB |

## Current cell status (2026-07-06 · 0.10.2)

Populated as tests land. Empty (🔲) cells will be filled by the 0.10.6 Phase
4 plumbing per `0.10-TODO.md`.

### Agent × Family matrix (11 × 4 = 44)

Pilot execution 2026-07-06 — **serial 3-family run** against real
inference under `RAPID_MLX_MATRIX_STRICT=1`. All PASS cells exercised
real tool-call routing (function name = `get_weather`, arg parses as
JSON, `city == "Tokyo"`, no `<think>` leak, no `<|channel|>analysis`
leak). PydanticAI + smolagents cells assert the tool implementation
itself was invoked (closure counter check), not just that the model
produced final text. OpenHands and Aider `XFAIL` structurally
because their native wire is text-action / edit-and-write, not
OpenAI function calling — real coverage lives in a Docker E2E
harness / bash harness respectively.

DeepSeek family Tier-1 rep was **swapped** from
`deepseek-v4-flash-8bit` (~155 GB, single-node-infeasible on M3 Ultra
+ G11 100 GB floor) to `deepseek-r1-32b-4bit`
(`mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit`, ~16 GB) after HF-API
size verification showed every complete V4 Flash quant is > 96 GB. The
swap preserves parser coverage (same `deepseek` tool-call parser +
`deepseek_r1` reasoning parser) while fitting the pilot's disk budget.
Full V4 Flash coverage tracked in follow-up issue **#1041**
("hardware plan needed").

| Family | Boot alias | Boot time | Wall time (14 cells) | Result |
|---|---|---|---|---|
| Qwen 3.6 | `qwen3.6-35b-8bit` (MoE, 3 B active) | ~15 s | 11.32 s | 12 PASS / 2 XFAIL |
| Gemma 4 | `gemma-4-31b-4bit` (dense) | ~10 s | 17.45 s | 12 PASS / 2 XFAIL |
| DeepSeek | `deepseek-r1-32b-4bit` (R1-distilled Qwen 32B, dense) | ~18 s | 191.29 s | 3 PASS / 11 XFAIL (9 arch-XFAIL R1-Distill tool-call gap, 2 pre-existing OpenHands/Aider) |
| gpt-oss | `gpt-oss-120b-mxfp4-q8` (MoE) | ~15 s | 14.61 s | 12 PASS / 2 XFAIL |

| Agent | Qwen 3.6 | Gemma 4 | DeepSeek | gpt-oss |
|---|---|---|---|---|
| codex-cli | ✅ | ✅ | ✅ | ✅ |
| claude-code | ✅ | ✅ | ✅ | ✅ |
| opencode | ✅ | ✅ | XFAIL (arch) | ✅ |
| qwen-code | ✅ | ✅ | XFAIL (arch) | ✅ |
| openhands | XFAIL | XFAIL | XFAIL | XFAIL |
| hermes-agent | ✅ | ✅ | XFAIL (arch) | ✅ |
| aider | XFAIL | XFAIL | XFAIL | XFAIL |
| kilo-code | ✅ | ✅ | XFAIL (arch) | ✅ |
| copilot | ✅ | ✅ | XFAIL (arch) | ✅ |
| droid | ✅ | ✅ | XFAIL (arch) | ✅ |
| kimi-code | ✅ | ✅ | XFAIL (arch) | ✅ |

### Framework × Family matrix (3 × 4 = 12)

| Framework | Qwen 3.6 | Gemma 4 | DeepSeek | gpt-oss |
|---|---|---|---|---|
| LangChain (+ LangGraph) | ✅ | ✅ | XFAIL (arch) | ✅ |
| PydanticAI | ✅ | ✅ | XFAIL (arch) | ✅ |
| smolagents | ✅ | ✅ | ✅ | ✅ |

Legend: ✅ passing (real inference · real tool call · semantic assertion)
· XFAIL = strict expected-fail with reason (Docker / shell harness required
for OpenHands + Aider; **XFAIL (arch)** = R1-Distill architectural
tool-emission gap, see next paragraph and issue #1041)

**Totals across all 4 families**: 56 cells run → **39 PASS · 17 XFAIL · 0 FAIL**
(15 XFAIL are structural expected-fails documented in `conftest.py` /
`test_agents_matrix.py` docstrings; 2 additional XFAILs are Aider +
OpenHands DeepSeek variants).

**DeepSeek family — architectural tool-emission gap.** The 9 DeepSeek
tool-call cells (7 agents + LangChain + PydanticAI) are marked
`xfail(strict=True)` via `pytest_collection_modifyitems` in
`conftest.py`. Root cause verified on **both** 4bit (16 GB) and 8bit
(34.8 GB) R1-Distill-Qwen-32B weights: R1's post-training was
reasoning-only per DeepSeek's own paper (arXiv 2501.12948 §2.3.3), and
distillation into Qwen 32B lost the base model's tool-emission
behavior. The refusal
`"I cannot provide the current weather in Tokyo as I cannot access the get_weather tool."`
reproduces deterministically at both quant levels — not a rapid-mlx
parser bug, not a quant artifact. Text-only cells (CodexCLI +
ClaudeCode) and smolagents' code-execution routing PASS on the same
booted server, proving the wire is healthy. Full tool-trained coverage
for the family needs V4-Chat / V4-Coder / V4-Flash weights, all
> 96 GB and single-node-infeasible on M3 Ultra under G11 — tracked in
follow-up issue #1041 (hardware plan).

## Historical deep-file coverage (pre-0.10.2)

For reference — this is what the deep flows historically covered on the
2026-06 M3 Ultra baseline before the matrix restructure:

| Test | Plain | Stream | Tool | Multi-tool | Structured | Notes |
|---|---|---|---|---|---|---|
| `test_pydantic_ai_full.py` | x | x | x | x | x | + multi-turn |
| `test_smolagents_full.py` | x | — | x | x | — | CodeAgent + ToolCallingAgent |
| `test_langchain.py` | x | x | x | x | x | + system prompt |
| `test_anthropic_sdk.py` | x | x | x | — | — | `/v1/messages` endpoint |
| `test_openwebui.py` | — | x | — | — | — | Docker: register, login, models, chat |
| `test_aider.sh` | — | — | — | — | — | CLI edit-and-write workflow |
| `test_librechat_docker.py` | — | — | — | — | — | Docker: register, login, endpoints, models |
| `test_hermes.py` | x | x | x | x | — | 62-tool Hermes Agent E2E + API stress test |

Model is auto-detected from the running server (`/v1/models` endpoint).

Run all agent tests automatically via:

```bash
rapid-mlx agents hermes --test
rapid-mlx agents                    # list all supported agents
```
