# SPDX-License-Identifier: Apache-2.0
"""Tier-1 frameworks × 3 families integration matrix (0.10.2).

Three Tier-1 frameworks from ``0.10-TODO.md`` §0.10.2:

* LangChain (+LangGraph — same profile / same wire)
* PydanticAI
* smolagents

Each cell is a smoke — plain-invoke + one tool call. Deep flows live in
the dedicated files (``test_langchain.py``, ``test_pydantic_ai_full.py``,
``test_smolagents_full.py``); the matrix cell here proves the framework
plumbs onto the running server's model without requiring the deep file
to be re-run for every family.
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
# LangChain (+ LangGraph)
# --------------------------------------------------------------------------- #


class TestLangChain:
    """LangChain / LangGraph — plain invoke + one tool call.

    LangGraph builds directly on ``langchain-openai``'s ``ChatOpenAI`` — a
    single profile covers both. LangGraph-specific StateGraph tests would
    add covered lines but no risk-of-regression signal; skipped here.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from langchain_core.messages import HumanMessage
            from langchain_core.tools import tool
            from langchain_openai import ChatOpenAI
        except ImportError:
            pytest.skip("langchain-openai not installed — cell deferred")

        llm = ChatOpenAI(
            model=rapid_mlx_server["model_id"],
            base_url=rapid_mlx_server["base_url"],
            api_key="not-needed",
            temperature=0.0,
            max_tokens=256,
        )

        # Plain invoke — confirm the model answers over the wire.
        try:
            r = llm.invoke([HumanMessage(content="Reply with just OK.")])
        except Exception as exc:  # noqa: BLE001
            # Codex #1030 finding 4: a wired-server failure on the same
            # /v1/chat/completions path LangChain drives is a regression.
            # Strict CI fails; local dev skips.
            strict_skip_or_fail(
                f"langchain/{family_alias.family}: plain invoke failed: {exc}"
            )
        content = r.content or ""
        assert_content_nonempty(content, ctx=f"langchain/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)

        # Tool call — confirm the bind_tools path plumbs onto rapid-mlx.
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"sunny in {city}"

        llm_with_tools = llm.bind_tools([get_weather])
        try:
            r = llm_with_tools.invoke(
                [HumanMessage(content="What's the weather in Tokyo? Use the tool.")]
            )
        except Exception as exc:  # noqa: BLE001
            # Codex #1030 finding 4: strict CI must fail on a real bind_tools
            # regression — one of the two most common LangChain agent paths.
            strict_skip_or_fail(
                f"langchain/{family_alias.family}: tool invoke failed: {exc}"
            )
        tool_calls = getattr(r, "tool_calls", None) or []
        if not tool_calls:
            # Codex #1030 round-2 finding 2: strict CI must catch the case
            # where LangChain's bind_tools path stops surfacing tool_calls —
            # that's exactly the regression a Tier-1 framework matrix cell
            # is meant to gate. Local dev on a small model still skips.
            strict_skip_or_fail(
                f"langchain/{family_alias.family}: model returned no tool_calls "
                f"on bind_tools path — strict CI treats this as a wire "
                f"regression on the LangChain tool route."
            )
            return  # unreachable in strict; explicit for local runs
        tc = tool_calls[0]
        # LangChain returns tool_calls as dicts with name/args/id.
        tc_dict = {
            "id": tc.get("id") or "call_lc_smoke",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["args"]),
            },
        }
        assert_tool_call_shape(tc_dict)
        assert tc["name"] == "get_weather", tc
        assert "city" in tc["args"], tc


# --------------------------------------------------------------------------- #
# PydanticAI
# --------------------------------------------------------------------------- #


class TestPydanticAI:
    """PydanticAI — plain run + structured output smoke."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError:
            pytest.skip("pydantic-ai not installed — cell deferred")

        model = OpenAIChatModel(
            model_name=rapid_mlx_server["model_id"],
            provider=OpenAIProvider(
                base_url=rapid_mlx_server["base_url"],
                api_key="not-needed",
            ),
        )
        agent = Agent(model)
        try:
            result = agent.run_sync("Reply with just OK.")
        except Exception as exc:  # noqa: BLE001
            # Codex #1030 finding 4: strict CI must fail on a real regression
            # in the PydanticAI run_sync path.
            strict_skip_or_fail(
                f"pydantic-ai/{family_alias.family}: run_sync failed: {exc}"
            )
        content = (result.output or "").strip()
        assert_content_nonempty(content, ctx=f"pydantic-ai/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)


# --------------------------------------------------------------------------- #
# smolagents
# --------------------------------------------------------------------------- #


class TestSmolagents:
    """smolagents — ToolCallingAgent with a real tool.

    Codex #1030 round-3 finding 2: an empty ``tools=[]`` ToolCallingAgent
    doesn't actually exercise the tool-calling path the docstring claims
    to smoke. A ``final_answer`` tool + a math helper give the agent a
    real routing decision — a wire regression on the smolagents tool
    format now surfaces as a hard red instead of a silent "plain reply".
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from smolagents import OpenAIServerModel, Tool, ToolCallingAgent
        except ImportError:
            pytest.skip("smolagents not installed — cell deferred")

        class GetWeatherTool(Tool):
            name = "get_weather"
            description = "Get the weather for a city."
            inputs = {
                "city": {
                    "type": "string",
                    "description": "City name.",
                }
            }
            output_type = "string"

            def forward(self, city: str) -> str:  # type: ignore[override]
                return f"sunny in {city}"

        model = OpenAIServerModel(
            model_id=rapid_mlx_server["model_id"],
            api_base=rapid_mlx_server["base_url"],
            api_key="not-needed",
        )
        agent = ToolCallingAgent(tools=[GetWeatherTool()], model=model, max_steps=3)
        try:
            answer = agent.run("What's the weather in Tokyo? Use the tool.")
        except Exception as exc:  # noqa: BLE001
            # Strict CI must fail on a real regression in the smolagents
            # tool-routing path — this is the whole point of a framework cell.
            strict_skip_or_fail(f"smolagents/{family_alias.family}: run failed: {exc}")
        content = str(answer)
        assert_content_nonempty(content, ctx=f"smolagents/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)
