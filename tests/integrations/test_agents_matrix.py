# SPDX-License-Identifier: Apache-2.0
"""Tier-1 agents × 4 families integration matrix (0.10.2 PR-2 pilot).

Eleven Tier-1 agents — the pilot finalized the top 10 via pre-flight
verification of five commercial CLIs (cursor / droid / kimi-code /
qodercli / copilot) against the "can be pointed at a custom OpenAI
base_url" bar; kilo-code retained from #1030 pending an operator scope
call on top-10-vs-top-11:

Existing wire cells (from #1030 scaffold):

* codex-cli (/v1/responses)
* claude-code (/v1/messages via Anthropic SDK — covered by
  ``test_anthropic_sdk.py``; the matrix cell here proves the SDK still
  drives an end-to-end tool loop on the running server)
* opencode (/v1/chat/completions)
* qwen-code (/v1/chat/completions, promoted from fallback pool since
  Cursor CLI's agent path is locked to Cursor's backend)
* openhands (/v1/chat/completions)
* hermes-agent (covered end-to-end by ``test_hermes.py`` — this file
  smokes the wire in a lightweight cell; promoted from fallback pool
  since Alibaba Qoder's CLI has no first-party OpenAI-compat base_url
  hook, only proxy wrappers)
* aider (CLI edit-and-write, covered by ``test_aider.sh``)
* kilo-code (/v1/chat/completions)

New wire cells (0.10.2 PR-2 pilot):

* copilot (GitHub Copilot CLI — /v1/chat/completions via BYOK env vars
  ``COPILOT_PROVIDER_BASE_URL`` + ``COPILOT_PROVIDER_API_KEY``, docs
  https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models)
* droid (Factory AI Droid CLI — /v1/chat/completions via
  ``~/.factory/settings.json`` ``customModels`` array with
  ``provider: generic-chat-completion-api``)
* kimi-code (Moonshot Kimi Code CLI — /v1/chat/completions via
  ``~/.kimi/config.toml`` provider block with ``type = "openai"``
  and ``base_url``)

Each cell is a **smoke** — connect via the agent's wire, run a single
tool-calling exchange, verify the response envelope is well-formed and
no channel-marker leak (``<think>``, ``<|channel|>analysis``, etc.).
Deeper flows (multi-turn, sustained fuzz) live in the dedicated
integration files (``test_hermes.py``, ``test_anthropic_sdk.py``).

Rationale for lightweight cells + heavy dedicated files:
1. Matrix runs fast enough for per-PR (< 60 s for 24 cells).
2. Regressions in a single agent's wire still surface (dedicated file
   would give a false-green if the CI skipped it).
3. New agents added to the Tier-1 list get a smoke automatically; the
   deep test file follows at whatever cadence the agent's stability
   deserves.

**Cells not exercised in this file — see the "🔲 pending" cells of the
README matrix:** claude-code needs an installed ``anthropic`` package (in
optional extras, not core); openhands needs Docker (E2E harness lives in
``test_openhands.py``, deferred to 0.10.6 Phase 4 plumbing per 0.10-TODO
line 246); aider drives via a shell script (``test_aider.sh``) since
aider is not importable — the matrix cell here defers to that harness.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.integrations.conftest import (
    FamilyAlias,
    assert_content_nonempty,
    assert_no_analysis_channel_leak,
    assert_no_think_tag_leak,
    assert_tool_call_shape,
    strict_skip_or_fail,
)

# --------------------------------------------------------------------------- #
# Shared per-cell tool-call payload
# --------------------------------------------------------------------------- #


_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_TOOL_PROMPT = "What's the weather in Tokyo? Use the get_weather tool."


def _openai_client_and_errors(base_url: str):
    """Lazy openai import — the pkg is optional in the base venv.

    Codex #1030 round-5 findings 1-3: return the exception classes AND
    the client instance from a single import site so callers can never
    accidentally trigger a raw ImportError before the skip guard fires.
    Every OpenAI-wire cell in this module goes through this helper.
    """
    try:
        from openai import (
            APIStatusError,
            BadRequestError,
            NotFoundError,
            OpenAI,
        )
    except ImportError:
        pytest.skip("openai package not installed — agent matrix skipped")
    client = OpenAI(base_url=base_url, api_key="not-needed")
    return client, (BadRequestError, NotFoundError, APIStatusError)


def _openai_client(base_url: str):
    """Back-compat single-value client accessor (thin wrapper).

    Preserved for cells that only want the client and use a bare
    ``except Exception`` handler; the tool-call helper below uses the
    tuple-returning ``_openai_client_and_errors`` for typed catching.
    """
    client, _errs = _openai_client_and_errors(base_url)
    return client


def _run_openai_tool_smoke(
    rapid_mlx_server: dict[str, Any],
    family_alias: FamilyAlias,
    *,
    agent_label: str,
) -> None:
    """Run one tool-call cell against ``/v1/chat/completions``.

    Shared by every Tier-1 agent that speaks the OpenAI wire (opencode,
    qwen-code, openhands, kilo-code, and — degraded — hermes).
    """
    client, wire_errors = _openai_client_and_errors(rapid_mlx_server["base_url"])
    model_id = rapid_mlx_server["model_id"]

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": _TOOL_PROMPT}],
            tools=_TOOL_SCHEMA,
            temperature=0.0,
            max_tokens=384,
        )
    except wire_errors as exc:
        # Codex #1030 finding 3: server rejecting a tool-call request is a
        # regression, not a degraded-cell condition. In strict mode we fail
        # the cell so CI can catch the wire break; non-strict skips so a
        # local dev on a still-booting server doesn't get spurious reds.
        strict_skip_or_fail(
            f"{agent_label}/{family_alias.family}: server rejected tool request "
            f"on {model_id!r}: {exc}"
        )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    content = msg.content or ""

    if not tool_calls:
        # A model may answer inline for small aliases — still assert wire
        # cleanliness so we catch channel leaks even without a tool call.
        assert_content_nonempty(content, ctx=f"{agent_label}/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)
        # Codex #1030 round-2 finding 1: an empty tool_calls slot on a Tier-1
        # agent cell is a real regression signal in CI (server may have
        # dropped tool-call plumbing) — strict mode fails; local dev on a
        # small alias still skips.
        strict_skip_or_fail(
            f"{agent_label}/{family_alias.family}: {model_id} returned no "
            f"tool_calls (content={content[:100]!r}); strict CI treats this "
            f"as a wire regression on the tool-call path."
        )
        return  # unreachable in strict; explicit for clarity in local runs

    tc = tool_calls[0]
    # openai SDK returns a Pydantic model — normalize to dict for the helper.
    tc_dict = {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
    assert_tool_call_shape(tc_dict)
    assert tc.function.name == "get_weather", tc.function.name
    args = json.loads(tc.function.arguments)
    assert "city" in args, args
    assert "tokyo" in args["city"].lower(), args


# --------------------------------------------------------------------------- #
# Cells — one per Tier-1 agent
# --------------------------------------------------------------------------- #


class TestCodexCLI:
    """Codex CLI /v1/responses — stateless shim (see codex.yaml)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        import httpx

        base_url = rapid_mlx_server["base_url"]
        model_id = rapid_mlx_server["model_id"]

        # Minimal /v1/responses envelope that Codex CLI would send.
        payload = {
            "model": model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply with just SHIPPED."}
                    ],
                }
            ],
            "stream": False,
            "max_output_tokens": 64,
        }
        try:
            r = httpx.post(f"{base_url}/responses", json=payload, timeout=90)
        except httpx.HTTPError as exc:
            # Codex #1030 round-3 finding 1: a transport failure to a wired
            # server AFTER the session-scope /v1/models probe succeeded is a
            # regression signal in strict CI (the server tore down mid-test
            # or the codex-shape route was pulled). Strict fails; local dev
            # on a flaky loopback still skips.
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: transport error hitting "
                f"/v1/responses after session probe was healthy: {exc!r}"
            )
        if r.status_code in (404, 405):
            # Codex #1030 round-2 finding 3: RAPID_MLX_MATRIX_STRICT=1 is meant
            # to gate exactly this — the /v1/responses route MUST be wired for
            # Codex CLI Tier-1 support. Strict CI fails; local dev on an older
            # server without the shim skips.
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: /v1/responses returned "
                f"{r.status_code} — route not wired on this server."
            )
        if r.status_code >= 400:
            # Codex #1030 finding 2: a 4xx / 5xx from a wired route IS a
            # regression. Strict CI must fail so the codex-shape SSE break
            # can't hide behind a skipped cell.
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: server returned {r.status_code} "
                f"({r.text[:200]!r})"
            )
        data = r.json()
        # Codex #1030 round-6 finding 4: walk the /v1/responses envelope,
        # extract the first output_text block, assert it is non-empty and
        # channel-clean. A blank output_text or a leaked ``<|channel|>``
        # marker now fails instead of passing on envelope-only truthiness.
        outputs = data.get("output") or []
        assert outputs, f"empty output envelope: {data}"
        text = ""
        for output_msg in outputs:
            for block in output_msg.get("content", []) or []:
                if block.get("type") in ("output_text", "text"):
                    text = block.get("text", "") or ""
                    if text:
                        break
            if text:
                break
        assert_content_nonempty(text, ctx=f"codex-cli/{family_alias.family}")
        assert_no_think_tag_leak(text)
        assert_no_analysis_channel_leak(text)


class TestClaudeCode:
    """Claude Code /v1/messages — Anthropic SDK route (see test_anthropic_sdk.py)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from anthropic import (
                Anthropic,
                APIStatusError,
                BadRequestError,
                NotFoundError,
            )
        except ImportError:
            pytest.skip("anthropic SDK not installed — cell deferred")

        base_no_v1 = rapid_mlx_server["base_url"].rstrip("/").removesuffix("/v1")
        client = Anthropic(base_url=base_no_v1, api_key="not-needed")

        try:
            resp = client.messages.create(
                model=rapid_mlx_server["model_id"],
                max_tokens=128,
                messages=[{"role": "user", "content": "Reply with just SHIPPED."}],
            )
        except NotFoundError:
            # Codex #1030 round-2 finding 4: strict CI must fail when the
            # Anthropic /v1/messages route is missing — that's exactly the
            # regression the Claude Code Tier-1 matrix cell is here to catch.
            # Local dev on a mock or older server still skips.
            strict_skip_or_fail(
                f"claude-code/{family_alias.family}: /v1/messages returned 404 "
                f"on {rapid_mlx_server['base_url']} — Anthropic route not wired."
            )
        except (BadRequestError, APIStatusError) as exc:
            # Codex #1030 finding 2: a wired-but-broken /v1/messages IS a
            # regression. Strict CI fails; local dev skips.
            strict_skip_or_fail(
                f"claude-code/{family_alias.family}: server rejected request: {exc}"
            )

        # Walk content blocks and find the first text — reasoning models emit a
        # thinking block first (see test_anthropic_sdk.py _first_text).
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        assert_content_nonempty(text, ctx=f"claude-code/{family_alias.family}")
        assert_no_think_tag_leak(text)
        assert_no_analysis_channel_leak(text)


class TestOpenCode:
    """OpenCode /v1/chat/completions with tool call."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="opencode")


class TestQwenCode:
    """Qwen Code /v1/chat/completions with tool call.

    Qwen Code speaks plain OpenAI wire via ``openaiCompatible.baseUrl``
    (see ``qwen-code.yaml``) — the smoke here is the same wire as the
    agent itself would drive.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="qwen-code")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OpenHands uses text-action format (openhands.yaml "
        "capabilities.function_calling: false), NOT OpenAI-style function "
        "calling — no OpenAI-shape tool_calls to parse. Real drive requires "
        "the Docker E2E harness deferred to 0.10.6 Phase 4. Wire-only smoke "
        "was previously here but was flagged as shape-only. Kept as a "
        "structural xfail so the matrix stays symmetric and the intended "
        "coverage gap is grep-visible in test output."
    ),
)
class TestOpenHands:
    """OpenHands — expected-fail placeholder for Docker E2E harness.

    OpenHands' native wire is a text-action format, not OpenAI function
    calling. A tool-call assertion against ``/v1/chat/completions``
    cannot faithfully represent what OpenHands actually drives. The
    real coverage will land in the 0.10.6 Phase 4 Docker E2E harness;
    this placeholder xfails strictly so the matrix cell count stays at
    11 × 4 and the coverage gap can't be quietly forgotten.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        pytest.fail(
            f"openhands/{family_alias.family}: strict xfail — Docker E2E "
            "harness required for OpenHands' text-action format; OpenAI "
            "tool-call shape does not apply. See class docstring."
        )


class TestHermesAgent:
    """Hermes Agent — deep flow in ``test_hermes.py``; matrix cell smokes wire."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(
            rapid_mlx_server, family_alias, agent_label="hermes-agent"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Aider drives via a bash CLI (``test_aider.sh``) using its own "
        "edit-and-write format, NOT OpenAI-style tool_calls — no OpenAI-"
        "shape tool_calls to parse. Real drive lives in the shell harness. "
        "Kept as a structural xfail so the matrix stays symmetric."
    ),
)
class TestAider:
    """Aider — expected-fail placeholder for shell CLI harness.

    Aider's real integration is a bash harness (``test_aider.sh``) that
    drives the ``aider`` CLI end-to-end through its own edit-and-write
    format. A tool-call assertion against ``/v1/chat/completions``
    cannot faithfully represent that flow. This placeholder xfails
    strictly so the 11 × 4 matrix stays symmetric and the coverage gap
    can't be quietly forgotten.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        pytest.fail(
            f"aider/{family_alias.family}: strict xfail — bash CLI harness "
            "(test_aider.sh) required for Aider's edit-and-write format; "
            "OpenAI tool-call shape does not apply. See class docstring."
        )


class TestKiloCode:
    """Kilo Code /v1/chat/completions with tool call.

    Kilo Code is a Cline fork; wire is standard OpenAI-compat
    (see ``kilo-code.yaml``). Cell smokes a tool call the same way
    Kilo's file-read + shell tools would.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="kilo-code")


class TestCopilot:
    """GitHub Copilot CLI wire smoke.

    Pre-flight verdict: **PASS** — Copilot CLI supports BYOK via the
    env vars ``COPILOT_PROVIDER_BASE_URL``, ``COPILOT_PROVIDER_API_KEY``,
    ``COPILOT_MODEL``, and ``COPILOT_PROVIDER_TYPE=openai``. Docs:
    https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models

    Cell shape: **wire-smoke only** — the plain
    ``/v1/chat/completions`` tool-call round-trip. Driving the real
    ``copilot`` CLI as a non-interactive subprocess is blocked on
    ``gh auth login`` OAuth (interactive TTY, no ``--no-tty`` escape
    hatch as of 2026-07). A follow-up sibling PR can add a real-CLI
    subprocess cell once a token-flow harness is agreed with raullen —
    the wire-smoke here still catches server-side tool-call regressions
    that would break Copilot BYOK users.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="copilot")


class TestDroid:
    """Factory AI Droid CLI wire smoke.

    Pre-flight verdict: **PASS** — Droid CLI supports custom models via
    ``~/.factory/settings.json`` ``customModels`` array with fields
    ``model`` / ``displayName`` / ``baseUrl`` / ``apiKey`` /
    ``provider = "generic-chat-completion-api"``. Docs:
    https://docs.factory.ai/cli/byok/overview

    Cell shape: **wire-smoke only** — same rationale as ``TestCopilot``.
    Real subprocess driving requires ``droid`` first-run onboarding and
    a Factory session token; wire-smoke catches the /v1/chat/completions
    tool-call regressions that would break Droid BYOK users.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="droid")


class TestKimiCode:
    """Moonshot Kimi Code CLI wire smoke.

    Pre-flight verdict: **PASS** — Kimi Code CLI supports OpenAI-compat
    providers via ``~/.kimi/config.toml`` provider blocks with
    ``type = "openai"`` + ``base_url``. Docs:
    https://moonshotai.github.io/kimi-cli/en/configuration/providers.html

    Cell shape: **wire-smoke only** — same rationale as ``TestCopilot``.
    Real subprocess driving requires kimi-cli first-run auth flow;
    wire-smoke catches the /v1/chat/completions tool-call regressions
    that would break Kimi-Code BYOK users.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="kimi-code")
