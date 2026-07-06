# Integration tests

End-to-end tests that exercise Rapid-MLX from a real client library.

These are **not** run as part of `pytest tests/` because they need a running
Rapid-MLX server on `http://localhost:8000` and a loaded model — the fixtures
`skip` cells when no server is reachable, so a naïve `pytest tests/` still
comes out green.

## Two matrices — 8 agents + 3 frameworks

0.10.2 restructured this directory around **two matrices**, both sharing the
harness in `conftest.py`:

- `test_agents_matrix.py` — **8 Tier-1 agents × 3 families** (Qwen 3.6,
  Gemma 4, gpt-oss) = 24 cells. Each cell is a lightweight smoke; deep flows
  live in the dedicated files below.
- `test_frameworks_matrix.py` — **3 Tier-1 frameworks × 3 families** = 9
  cells.

Support ≡ a real integration test that boots the server + real model + real
client flow, not just a YAML profile. See `workflow.md` W3 taxonomy §B.3.

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
# All 24 agent cells; only the family matching the running server passes,
# the other two skip (non-strict) or fail (strict). Use for local sanity;
# for CI, prefer per-family shards below.
pytest tests/integrations/test_agents_matrix.py -v

# 9-cell framework matrix (same shard rule as above)
pytest tests/integrations/test_frameworks_matrix.py -v

# Strict CI — per-family shard (this is the intended workflow: three
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
| `RAPID_MLX_AGENT_MATRIX_FAMILY` | (all) | Restrict to `qwen36` / `gemma4` / `gptoss` |
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
| gpt-oss | `gpt-oss-20b-mxfp4-q8` | Smallest gpt-oss; ~11 GB |

## Current cell status (2026-07-06 · 0.10.2)

Populated as tests land. Empty (🔲) cells will be filled by the 0.10.6 Phase
4 plumbing per `0.10-TODO.md`.

### Agent × Family matrix (8 × 3 = 24)

| Agent | Qwen 3.6 | Gemma 4 | gpt-oss |
|---|---|---|---|
| codex-cli | 🔲 | 🔲 | 🔲 |
| claude-code | 🔲 | 🔲 | 🔲 |
| opencode | 🔲 | 🔲 | 🔲 |
| qwen-code | 🔲 | 🔲 | 🔲 |
| openhands | 🔲 | 🔲 | 🔲 |
| hermes-agent | 🔲 | 🔲 | 🔲 |
| aider | 🔲 | 🔲 | 🔲 |
| kilo-code | 🔲 | 🔲 | 🔲 |

### Framework × Family matrix (3 × 3 = 9)

| Framework | Qwen 3.6 | Gemma 4 | gpt-oss |
|---|---|---|---|
| LangChain (+ LangGraph) | 🔲 | 🔲 | 🔲 |
| PydanticAI | 🔲 | 🔲 | 🔲 |
| smolagents | 🔲 | 🔲 | 🔲 |

Legend: ✅ passing · ⚠️ skipped (known cause) · 🔲 pending

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
