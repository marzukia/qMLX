"""Round-trip stability for the enable_thinking=False generation scaffold.

Regression guard for the KV-cache-break bug (F-thinkscaffold): the chat template
appends an empty ``<think>\\n\\n</think>\\n\\n`` to the GENERATION prompt when
``enable_thinking`` is False, but does NOT reproduce it when the same assistant
turn is later rendered as history in a multi-turn tool conversation. The
checkpoint (with scaffold) then diverges from the next turn's prefix (without),
forcing a cold re-prefill on every tool turn. The fix strips the scaffold from
the generation prompt so generation matches the scaffold-free history render.
"""

from vllm_mlx.utils.chat_template import (
    _GEN_THINK_SCAFFOLD,
    _strip_gen_think_scaffold,
    apply_chat_template,
)

SCAFFOLD = _GEN_THINK_SCAFFOLD  # "<think>\n\n</think>\n\n"


def test_strip_helper_removes_trailing_scaffold():
    p = "<|im_start|>assistant\n" + SCAFFOLD
    assert _strip_gen_think_scaffold(p) == "<|im_start|>assistant\n"


def test_strip_helper_noop_without_scaffold():
    # thinking-enabled generation ends with an OPEN think tag, must be untouched
    p = "<|im_start|>assistant\n<think>\n"
    assert _strip_gen_think_scaffold(p) == p
    # plain text, no think at all
    q = "<|im_start|>assistant\nhello"
    assert _strip_gen_think_scaffold(q) == q


def test_strip_helper_only_strips_suffix():
    # a scaffold mid-string (real prior reasoning-less turn) must stay
    p = "<|im_start|>assistant\n" + SCAFFOLD + "<tool_call>...</tool_call><|im_end|>\n"
    assert _strip_gen_think_scaffold(p) == p


class _FakeQwenApplicator:
    """Mimics the Qwen3.5 asymmetry: the generation prompt gets the empty
    scaffold when thinking is disabled, but completed assistant turns in history
    do not (this is the exact behaviour observed on real captured payloads)."""

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
        tools=None,
        **kw,
    ):
        out = []
        for m in messages:
            content = m.get("content") or ""
            tcs = m.get("tool_calls") or []
            tc_txt = "".join(
                "<tool_call>{}</tool_call>".format(tc["function"]["name"]) for tc in tcs
            )
            out.append(
                "<|im_start|>{}\n{}{}<|im_end|>\n".format(m["role"], content, tc_txt)
            )
        s = "".join(out)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if enable_thinking is False:
                s += "<think>\n\n</think>\n\n"  # the asymmetric scaffold
        return s


def test_generation_prompt_is_scaffold_free_when_thinking_disabled():
    msgs = [{"role": "user", "content": "fix the bug"}]
    gen = apply_chat_template(
        _FakeQwenApplicator(), msgs, enable_thinking=False, model_name="qwen"
    )
    assert not gen.endswith(SCAFFOLD), (
        "generation prompt still carries the empty think scaffold"
    )
    assert gen.endswith("<|im_start|>assistant\n")


def test_roundtrip_generation_matches_history_render():
    """The KV checkpoint is the generation prompt + generated tokens. The next
    turn re-renders that turn as history. Both must agree byte-for-byte at the
    turn boundary or the cache diverges. With the fix, they do."""
    fake = _FakeQwenApplicator()
    base = [{"role": "user", "content": "fix the bug"}]
    # what the model saw at generation time (checkpoint prefix), fix applied:
    gen_prompt = apply_chat_template(
        fake, base, enable_thinking=False, model_name="qwen"
    )
    # the model then generates a tool call; the completed turn as the client stores it:
    asst = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "bash"}}],
    }
    # next turn's render of the SAME history:
    hist = apply_chat_template(
        fake, base + [asst], enable_thinking=False, model_name="qwen"
    )
    # the history render up to the assistant turn must start with the exact
    # generation-prompt prefix (no scaffold on either side).
    assert hist.startswith(gen_prompt), (
        "history render diverges from the generation prompt at the turn boundary:\n"
        f"gen : {gen_prompt[-40:]!r}\nhist: {hist[len(gen_prompt) - 40 : len(gen_prompt) + 20]!r}"
    )
