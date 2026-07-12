# SPDX-License-Identifier: Apache-2.0
"""r8-C: streaming reasoning split + think-leak regressions.

Drives the full streaming postprocessor (``StreamingPostProcessor``)
with the prompts identified in the r8 Mira + Sven evidence and asserts
the SSE-level routing matches the documented non-stream behaviour.

* **R8-M6** — UI-TARS-native ``Thought: I should answer.\n\nAnswer: 4``
  used to stream the entire prompt (including the answer ``"4"``) on
  ``delta.reasoning``. The non-stream path correctly splits at the
  blank-line boundary per ``a16d8c8`` (shape #4) — the streaming
  state machine in ``vllm_mlx/reasoning/ui_tars_parser.py`` did not
  mirror that exit predicate. Mirror also added for ``</think>``
  (shape #5) and ``Answer:`` (defensive UI-TARS native form).

* **R8-M2** — With ``enable_thinking=False`` and ``tool_choice="auto"``
  Qwen3-thinking sometimes ignores the off-flag and still emits an
  explicit ``<think>...</think>`` wrapper. The pre-fix bypass routed
  the literal wrapper bytes to ``delta.content`` BEFORE the tool-call
  chunk. The postprocessor now detects the explicit wrapper (including
  its split-SSE leading edge) and re-enters the reasoning lane so the
  gate splits BEFORE content emit.

Tests deliberately exercise the postprocessor end-to-end (the parser
in isolation is covered separately in ``test_ui_tars_parser.py`` /
``test_reasoning_parsers.py``) so the assertions reflect the
on-wire SSE shape, not the parser's intermediate ``DeltaMessage``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm_mlx.service.postprocessor import StreamingPostProcessor


def _make_cfg(**overrides):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = False
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_output(text: str = "", finished: bool = False):
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = None
    out.finish_reason = "stop" if finished else None
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


def _drive(pp: StreamingPostProcessor, deltas: list[str]) -> dict:
    """Drive a sequence of deltas through the postprocessor and return
    the concatenated SSE-level reasoning and content streams.

    Concatenates content from BOTH ``type="content"`` events AND the
    terminal ``type="finish"`` event's ``content`` field — the
    postprocessor folds the last delta's content into the finish event
    when ``finished=True`` arrives in the same chunk."""
    all_events = []
    for i, d in enumerate(deltas):
        finished = i == len(deltas) - 1
        all_events.extend(pp.process_chunk(_make_output(d, finished=finished)))
    reasoning = "".join(
        getattr(e, "reasoning", "") or "" for e in all_events if e.type == "reasoning"
    )
    content = "".join(
        getattr(e, "content", "") or ""
        for e in all_events
        if e.type in ("content", "finish")
    )
    return {"reasoning": reasoning, "content": content, "events": all_events}


class TestR8M2ToolChoiceAutoThinkLeak:
    """When the model emits an explicit ``<think>...</think>`` wrapper
    despite ``enable_thinking=False``, the streaming postprocessor must
    re-enter the reasoning lane so the wrapper splits at the gate
    BEFORE content emit. Pre-fix the wrapper leaked into
    ``delta.content`` before the tool-call chunk.
    """

    def _pp(self, enable_thinking=False):
        cfg = _make_cfg(
            reasoning_parser_name="qwen3",
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
        )
        pp = StreamingPostProcessor(
            cfg, tools_requested=True, enable_thinking=enable_thinking
        )
        pp.reset()
        return pp

    def test_explicit_think_wrapper_routed_to_reasoning(self):
        """Whole-chunk wrapper: ``<think>body</think>`` before the
        tool_call chunk. Body must go to reasoning, NOT content."""
        pp = self._pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                "<think>",
                "I should call get_weather.",
                "</think>",
                '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>',
            ],
        )
        assert "should call get_weather" in result["reasoning"]
        # The wrapper bytes must NOT have leaked into content.
        assert "<think>" not in result["content"]
        assert "</think>" not in result["content"]
        assert "should call" not in result["content"]

    def test_split_think_tag_no_leak(self):
        """SSE-split opener tag: ``<th`` then ``ink>`` then body. The
        leading edge of the tag must be held until the full opener
        resolves so neither half leaks to content."""
        pp = self._pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                "<th",
                "ink>",
                "thinking...",
                "</think>",
                '<tool_call>{"name":"foo","arguments":{}}</tool_call>',
            ],
        )
        assert "thinking..." in result["reasoning"]
        assert "<th" not in result["content"]
        assert "ink>" not in result["content"]
        assert "thinking" not in result["content"]

    def test_no_think_wrapper_still_bypasses(self):
        """Sanity: with ``enable_thinking=False`` AND no ``<think>``
        opener in the output, the bypass must still apply so a plain
        direct answer flows to ``delta.content`` (this is the
        original purpose of the bypass — PR #208 closed the empty-
        content bug). The R8-M2 fix must NOT regress it."""
        pp = self._pp(enable_thinking=False)
        result = _drive(pp, ["The answer is ", "Paris."])
        assert result["content"] == "The answer is Paris."
        assert result["reasoning"] == ""

    def test_false_positive_tag_lookalike_does_not_lock_promotion(self):
        """A non-``<think>`` payload that happens to start with ``<``
        (e.g. ``<thanks for asking!``) must NOT permanently promote
        the bypass to reasoning lane. Once the full prefix is in the
        accumulator and clearly not ``<think>``, the bypass resumes."""
        pp = self._pp(enable_thinking=False)
        result = _drive(pp, ["<thanks for asking!"])
        # The accumulated buffer shows it's not a <think> opener; bypass
        # routes it as content.
        assert "<thanks for asking!" in result["content"]
        assert result["reasoning"] == ""

    def test_default_enable_thinking_unaffected(self):
        """Regression guard: ``enable_thinking=None`` (default) keeps
        going through the reasoning parser as before — the R8-M2 fix
        is scoped to the ``False`` bypass override."""
        pp = self._pp(enable_thinking=None)
        result = _drive(
            pp,
            [
                "<think>",
                "thinking.",
                "</think>",
                '<tool_call>{"name":"foo","arguments":{}}</tool_call>',
            ],
        )
        assert "thinking." in result["reasoning"]
        assert "<think>" not in result["content"]

    def test_literal_think_token_mid_content_does_not_latch(self):
        """Codex r8-C round-2 MED: a plain direct answer that MENTIONS
        ``<think>`` mid-content (e.g. explaining HTML tags or
        documenting a reasoning wrapper) must NOT latch the bypass
        promotion. Pre-fix, ``self._THINK_OPEN_TOKEN in probe``
        triggered anywhere in the buffer, so the moment the model
        emitted ``... use the <think> tag ...`` everything after that
        chunk routed through the reasoning parser and the answer body
        was hidden. Post-fix, the complete-token branch anchors at the
        first non-whitespace bytes (mirror of the split-prefix branch
        below it)."""
        pp = self._pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                "To wrap reasoning, ",
                "use the <think> tag. ",
                "It is a reserved token.",
            ],
        )
        # Entire answer stays on content; no chunk routed to reasoning.
        assert result["reasoning"] == ""
        assert "use the <think> tag" in result["content"]
        assert "It is a reserved token." in result["content"]
        # Latch must NOT have promoted (so a subsequent reset+request
        # in the singleton path doesn't drag a stale latch).
        assert pp._explicit_think_seen is False

    def test_r10c7_plain_then_standalone_think_chunk_does_not_latch(self):
        """R10-C7 (Mira r10-R1, 2026-06-23): the r8-C anchor used
        ``self.accumulated_text + delta_text`` and required ``<think>``
        at the buffer HEAD to latch. But ``_process_standard`` (the
        path used while the bypass is active) does NOT mutate
        ``accumulated_text`` — only ``_process_with_reasoning`` does
        (postprocessor.py:2158/2163). So a plain answer that emits
        chunk #1 of clean content followed by chunk #2 starting with
        a literal ``<think>`` token would see ``probe = "" +
        "<think>..."``, match ``head.startswith("<think>")``, and latch
        ``_explicit_think_seen`` — rerouting the rest of the answer to
        ``delta.reasoning`` and hiding the body.

        Post-fix the ``_standard_content_observed`` latch refuses any
        mid-content ``<think>`` promotion once the bypass has emitted
        plain content for this request. The historic happy paths
        (first-token ``<think>`` latches; split-SSE ``<th``/``ink>``
        held back; ``<thanks for asking!`` does NOT latch) are
        preserved by the other tests in this file.
        """
        pp = self._pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                # Chunk #1: clean plain content. No <think>.
                "Reply with a snippet that says ",
                # Chunk #2: literal <think> at the head of THIS chunk
                # but mid-content for the response. Pre-fix this
                # latched the reasoning router.
                "<think> as code. ",
                # Chunk #3: continuation of the plain answer.
                "It is a reserved token.",
            ],
        )
        # Entire answer stays on content; no chunk routed to reasoning.
        assert result["reasoning"] == ""
        assert "Reply with a snippet" in result["content"]
        assert "<think> as code." in result["content"]
        assert "It is a reserved token." in result["content"]
        # R10-C7 latch was set on the first plain-content emission.
        assert pp._standard_content_observed is True
        # And the r8-C bypass latch must NOT have promoted.
        assert pp._explicit_think_seen is False

    def test_r10c7_whitespace_only_prefix_does_not_block_think_promotion(self):
        """Codex r10-F HIGH (review of this PR): the router's head
        anchor uses ``probe.lstrip()`` so it deliberately tolerates
        leading whitespace before the ``<think>`` opener (Sven r8-M2
        evidence: Qwen3-thinking sometimes prefixes its wrapper with
        ``\\n`` / spaces). The R10-C7 latch must therefore gate on a
        NON-WHITESPACE byte — emitting a leading whitespace-only
        chunk via ``_process_standard`` must NOT poison subsequent
        explicit-``<think>`` promotion.

        Pre-fix (codex round-1 finding): chunks ``" \\n"`` then
        ``"<think>"`` then body produced content containing
        ``"<think>private reasoningfinal answer"`` and empty
        reasoning — the whitespace chunk latched
        ``_standard_content_observed`` so the head-match was
        short-circuited.
        """
        pp = self._pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                # Whitespace-only leading chunk.
                " \n",
                # Explicit wrapper opener.
                "<think>",
                "private reasoning",
                "</think>",
                "final answer",
            ],
        )
        # Reasoning lane saw the wrapper body.
        assert "private reasoning" in result["reasoning"]
        # Content lane saw the answer body, NOT the wrapper.
        assert "<think>" not in result["content"]
        assert "</think>" not in result["content"]
        assert "private reasoning" not in result["content"]
        assert "final answer" in result["content"]
        # The R8-M2 latch did promote (correctly).
        assert pp._explicit_think_seen is True

    def test_r10c7_first_token_think_still_latches(self):
        """R10-C7 must NOT regress the original r8-C happy path. When
        the FIRST chunk is the ``<think>`` opener (Qwen3-thinking
        emitting an explicit wrapper despite ``enable_thinking=False``,
        before any plain content has been streamed), the bypass still
        promotes to the reasoning lane so the wrapper body splits
        correctly.

        Distinct from
        ``test_explicit_think_wrapper_routed_to_reasoning`` above: this
        version targets the latch state directly to lock in that the
        R10-C7 short-circuit only fires AFTER ``_process_standard``
        has emitted plain content for this request.
        """
        pp = self._pp(enable_thinking=False)
        # ``_process_standard`` has NOT emitted any plain content yet
        # at request start.
        assert pp._standard_content_observed is False
        result = _drive(
            pp,
            ["<think>", "I should reply.", "</think>", "The answer is 42."],
        )
        # Reasoning lane saw the wrapper body.
        assert "I should reply." in result["reasoning"]
        # Content lane saw the answer body, NOT the wrapper.
        assert "<think>" not in result["content"]
        assert "</think>" not in result["content"]
        assert "The answer is 42." in result["content"]
        # The r8-C latch did promote (correctly).
        assert pp._explicit_think_seen is True


# =====================================================================
# Reset / lifecycle — the latch must clear between requests
# =====================================================================


class TestR8M2ResetClearsLatch:
    """``reset()`` must clear the ``_explicit_think_seen`` latch so a
    re-used postprocessor instance (legacy singleton path) doesn't
    carry the prior request's promotion into the next request."""

    def test_latch_cleared_on_reset(self):
        cfg = _make_cfg(
            reasoning_parser_name="qwen3",
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
        )
        pp = StreamingPostProcessor(cfg, tools_requested=True, enable_thinking=False)
        pp.reset()
        # First request: explicit <think> triggers promotion.
        pp.process_chunk(_make_output("<think>"))
        pp.process_chunk(_make_output("thinking."))
        assert pp._explicit_think_seen is True
        # Reset for next request.
        pp.reset()
        assert pp._explicit_think_seen is False
        # Second request: plain answer; bypass should apply again.
        result_events = []
        for d in ["The answer is ", "Paris."]:
            result_events.extend(pp.process_chunk(_make_output(d)))
        content = "".join(
            getattr(e, "content", "") or ""
            for e in result_events
            if e.type == "content"
        )
        reasoning = "".join(
            getattr(e, "reasoning", "") or ""
            for e in result_events
            if e.type == "reasoning"
        )
        assert content == "The answer is Paris."
        assert reasoning == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
