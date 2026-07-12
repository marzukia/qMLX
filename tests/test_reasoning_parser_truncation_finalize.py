# SPDX-License-Identifier: Apache-2.0
"""r5-D — reasoning-parser finalize-on-truncation contract tests.

Pins the parser-side fix for F-DGF-V080-B-7 (gemma4 dup-into-both-fields)
and F-DGF-V080-B-9 (glm4 leak-into-content) on the non-streaming
``/v1/chat/completions`` aggregator's finalize-on-truncation path.

Bug shape:

* When ``finish_reason="length"`` truncates the model mid-think (the
  closing ``</think>`` / ``<channel|>`` sentinel never arrives), the
  pre-r5-D non-streaming aggregator misclassified the unclosed buffer:
  - gemma4 duplicated the raw scratchpad into BOTH ``content`` and
    ``reasoning_content`` (when the engine's token-level OutputRouter
    also populated ``engine_reasoning_text``).
  - glm4 / minimax leaked the raw scratchpad into ``content`` with
    ``reasoning_content=null``.

Fix (shared finalize-on-truncation):

* Each parser implements ``is_open_in_think(text)`` (default False) so
  the route knows whether the unclosed buffer should be classified
  as reasoning.
* A shared ``finalize_truncation(open_in_think, buffer)`` helper in
  ``vllm_mlx/reasoning/base.py`` routes the buffer parser-agnostically.
* The non-streaming aggregator
  ``vllm_mlx/service/helpers.py::_finalize_content_and_reasoning``
  invokes the helper when ``finish_reason="length"`` is reported AND
  the parser's first pass returned the leak shape.
* Each parser's own ``extract_reasoning`` ALSO routes correctly on the
  parser-side so unit-test callers that don't go through the helper
  see the right behaviour.

The no-regression contract: ``finish_reason="stop"`` (or
``finish_reason="length"`` with the buffer already closed —
``</think>answer``) splits cleanly, byte-identical pre/post.
"""

from __future__ import annotations

import pytest

from vllm_mlx.api.utils import clean_output_text, strip_thinking_tags
from vllm_mlx.reasoning import finalize_truncation
from vllm_mlx.reasoning.deepseek_r1_parser import (
    DeepSeekR1ReasoningParser,
    VibeThinkerReasoningParser,
)
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from vllm_mlx.service.helpers import (
    _finalize_content_and_reasoning,
    _rescue_silent_drop_from_reasoning,
)


def _route_end_to_end(parser, raw, finish_reason):
    """Mini-route harness: ``_finalize_content_and_reasoning`` →
    ``clean_output_text`` + ``strip_thinking_tags`` →
    ``_rescue_silent_drop_from_reasoning``. Matches the chat-route
    finalize flow (chat.py:~2007–2115) but does not exercise tool
    parsing or response_format. Returns the user-facing
    ``(content, reasoning_content)`` pair the route would ship.
    """
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=parser,
        engine_reasoning_text="",
        finish_reason=finish_reason,
    )
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
    rescued = _rescue_silent_drop_from_reasoning(
        final_content,
        reasoning_text,
        tool_calls=[],
        finish_reason=finish_reason,
        raw_text=raw,
        reasoning_is_case4=False,
    )
    return rescued, reasoning_text


# ---------------------------------------------------------------------
# Shared helper contract
# ---------------------------------------------------------------------


class TestFinalizeTruncationHelper:
    """``finalize_truncation`` is the single source of truth for the
    parser-agnostic open-in-think → (reasoning, content) routing."""

    def test_open_in_think_routes_to_reasoning(self):
        reasoning, content = finalize_truncation(True, "step 1 of my thought")
        assert reasoning == "step 1 of my thought"
        assert content is None

    def test_not_open_in_think_routes_to_content(self):
        reasoning, content = finalize_truncation(False, "final answer body")
        assert reasoning is None
        assert content == "final answer body"

    def test_empty_buffer_routes_to_none(self):
        assert finalize_truncation(True, "") == (None, None)
        assert finalize_truncation(False, "") == (None, None)
        assert finalize_truncation(True, None) == (None, None)
        assert finalize_truncation(False, None) == (None, None)


# ---------------------------------------------------------------------
# Per-parser ``is_open_in_think`` contract
# ---------------------------------------------------------------------


class TestIsOpenInThink:
    """Each thinking parser correctly identifies its own unclosed-think
    state from accumulated text. Non-think parsers default to False."""

    def test_qwen3_open_in_think(self):
        p = Qwen3ReasoningParser()
        assert p.is_open_in_think("<think>Reasoning so far") is True

    def test_qwen3_closed(self):
        p = Qwen3ReasoningParser()
        assert p.is_open_in_think("<think>R</think>answer") is False

    def test_deepseek_r1_open_in_think(self):
        p = DeepSeekR1ReasoningParser()
        assert p.is_open_in_think("<think>R") is True


# ---------------------------------------------------------------------
# Per-parser ``extract_reasoning`` finalize-on-truncation contract
# ---------------------------------------------------------------------


class TestExtractReasoningMidThink:
    """The parser-side fix: ``extract_reasoning`` on a buffer that
    ends inside an unclosed think tag must return
    ``(reasoning=buffer, content=None)``. Pre-fix gemma4 / minimax
    leaked the buffer into ``content``."""

    def test_qwen3_mid_thought_with_think_opener(self):
        p = Qwen3ReasoningParser()
        reasoning, content = p.extract_reasoning("<think>Reasoning so far")
        assert reasoning == "Reasoning so far"
        assert content is None

    def test_deepseek_r1_mid_thought_with_think_opener(self):
        p = DeepSeekR1ReasoningParser()
        reasoning, content = p.extract_reasoning("<think>Reasoning so far")
        assert reasoning == "Reasoning so far"
        assert content is None


class TestExtractReasoningClosedStop:
    """No-regression: ``finish_reason="stop"`` happy-path (think
    properly closed with answer following) must split cleanly,
    byte-identical pre/post."""

    def test_qwen3_clean_split(self):
        p = Qwen3ReasoningParser()
        reasoning, content = p.extract_reasoning(
            "<think>Reasoning</think>The answer is 42."
        )
        assert reasoning == "Reasoning"
        assert content == "The answer is 42."

    def test_deepseek_r1_clean_split(self):
        p = DeepSeekR1ReasoningParser()
        reasoning, content = p.extract_reasoning(
            "<think>Reasoning</think>The answer is 42."
        )
        assert reasoning == "Reasoning"
        assert content == "The answer is 42."


# ---------------------------------------------------------------------
# Route-level finalize plug: ``_finalize_content_and_reasoning``
# integration with ``finish_reason``
# ---------------------------------------------------------------------


class TestRouteFinalizeOnTruncation:
    """The shared finalize-on-truncation plug in
    ``_finalize_content_and_reasoning`` covers the gaps each parser's
    ``extract_reasoning`` cannot detect from text alone (notably
    glm4 autonomous mode — no ``<think>`` tag emitted)."""

    def test_qwen3_length_truncation_routes_to_reasoning(self):
        """Cross-parser sweep: qwen3 mid-think on ``finish_reason="length"``
        must route to reasoning_content. Existing Case-3 fallback
        already handled this — pin it for no-regression."""
        raw = "<think>Reasoning mid-thought"
        parser = Qwen3ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text is not None
        assert "Reasoning mid-thought" in reasoning_text
        assert "<think>" not in (cleaned_text or "")

    def test_deepseek_r1_length_truncation_routes_to_reasoning(self):
        """Cross-parser sweep: deepseek-r1 mid-think — existing Case-3
        fallback already handled this — pin it for no-regression."""
        raw = "<think>Reasoning mid-thought"
        parser = DeepSeekR1ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text is not None
        assert "Reasoning mid-thought" in reasoning_text
        assert "<think>" not in (cleaned_text or "")


class TestRouteFinishReasonStopNoRegression:
    """``finish_reason="stop"`` (clean close) MUST split cleanly,
    byte-identical pre/post-r5-D. This is the most important
    regression-prevention check — desktop clients have been seeing
    correct behaviour on the happy path and we must not disturb it."""

    def test_qwen3_clean_split_finish_stop(self):
        raw = "<think>Reasoning</think>The answer is 42."
        parser = Qwen3ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="stop",
        )
        assert reasoning_text == "Reasoning"
        assert cleaned_text == "The answer is 42."


class TestRouteFinishReasonLengthAfterClose:
    """``finish_reason="length"`` AFTER ``</think>`` closed and content
    started: the buffer is partial CONTENT, not reasoning. Must NOT
    re-route the content into reasoning."""

    def test_qwen3_length_after_close(self):
        raw = "<think>Reasoning</think>Partial answer that was cut"
        parser = Qwen3ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text == "Reasoning"
        assert cleaned_text == "Partial answer that was cut"


class TestNoDuplicationOfBuffer:
    """The original B-7 bug: gemma4 dup'd the same bytes into BOTH
    fields. The post-fix contract is that ``content`` and
    ``reasoning_content`` MUST NEVER be byte-identical when both are
    non-empty (modulo the edge case where reasoning ends with the
    same suffix the content starts with, which is structurally
    distinct from a full-buffer dup)."""

    @pytest.mark.parametrize(
        "parser_cls,raw",
        [
            (
                Qwen3ReasoningParser,
                "<think>Reasoning that was truncated mid-flight",
            ),
            (
                DeepSeekR1ReasoningParser,
                "<think>Reasoning that was truncated mid-flight",
            ),
            (
                VibeThinkerReasoningParser,
                "<think>Reasoning that was truncated mid-flight",
            ),
        ],
    )
    def test_no_dup_on_length_truncation(self, parser_cls, raw):
        parser = parser_cls()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        # If both are non-empty, they MUST NOT be byte-identical.
        if cleaned_text and reasoning_text:
            assert cleaned_text != reasoning_text, (
                f"{parser_cls.__name__} duplicated buffer into both "
                f"content and reasoning_content: {cleaned_text[:80]!r}"
            )


class TestVibeThinkerLengthTruncation:
    """VibeThinker is a DeepSeek-R1 variant — its truncated-think
    behaviour was already pinned by the live-test plug
    (``first_parse_was_truncated_think``). Pin it again here under
    the r5-D contract so the plug ordering doesn't drift."""

    def test_vibethinker_mid_think(self):
        raw = "<think>Reasoning so far"
        parser = VibeThinkerReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text is not None
        assert "Reasoning so far" in reasoning_text
        assert "<think>" not in (cleaned_text or "")


class TestEndToEndRouteContract:
    """End-to-end mini-route harness — exercises
    ``_finalize_content_and_reasoning`` → ``clean_output_text`` /
    ``strip_thinking_tags`` → ``_rescue_silent_drop_from_reasoning``.

    The bug repro at the route layer: even when the parser-side fix
    produces ``reasoning_text=<thought>`` and ``cleaned_text=""``, the
    silent-drop rescue can re-surface the reasoning bytes as
    ``content`` and re-introduce the dup-into-both-fields shape (the
    B-7 132/128/512-char identical-dup repro). The rescue's
    truncated-``<think>`` gate handles the qwen3/glm4 family but
    misses gemma4's channel format — the r5-D plug adds the gemma4
    analog.

    Contract: ``finish_reason="length"`` mid-think → ``content=None``,
    ``reasoning_content=<thought>``, NEVER byte-identical.
    """

    def test_qwen3_end_to_end_no_dup(self):
        raw = "<think>Mid-thought reasoning that was cut short"
        content, reasoning = _route_end_to_end(Qwen3ReasoningParser(), raw, "length")
        assert content is None
        assert reasoning is not None
        assert "Mid-thought reasoning" in reasoning
        assert content != reasoning

    def test_deepseek_r1_end_to_end_no_dup(self):
        raw = "<think>Mid-thought reasoning that was cut short"
        content, reasoning = _route_end_to_end(
            DeepSeekR1ReasoningParser(), raw, "length"
        )
        assert content is None
        assert reasoning is not None
        assert "Mid-thought reasoning" in reasoning
        assert content != reasoning

    def test_qwen3_end_to_end_clean_split_finish_stop(self):
        raw = "<think>Reasoning</think>The answer is 42."
        content, reasoning = _route_end_to_end(Qwen3ReasoningParser(), raw, "stop")
        assert content == "The answer is 42."
        assert reasoning == "Reasoning"


# ---------------------------------------------------------------------
# r5-D codex r1 BLOCKING follow-up: legitimate content containing a
# LITERAL ``<|channel>thought`` / ``<think>`` substring must NOT be
# reclassified as reasoning by the new finalize-on-truncation gates.
#
# Mirrors the contract pinned by
# ``test_literal_closed_think_in_answer_preserved_non_streaming`` in
# ``tests/test_reasoning_parsers.py`` (PR #722 codex r3) for the two
# parsers whose new finalize branches missed the substring-vs-structural
# distinction.
# ---------------------------------------------------------------------
