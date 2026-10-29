"""Round-trip stability for the enable_thinking=False generation scaffold.

Regression guard for the KV-cache-break bug (F-thinkscaffold): the chat template
appends an empty ``<think>\\n\\n</think>\\n\\n`` to the GENERATION prompt when
``enable_thinking`` is False, but does NOT reproduce it when the same assistant
turn is later rendered as history in a multi-turn tool conversation. The
checkpoint (with scaffold) then diverges from the next turn's prefix (without),
forcing a cold re-prefill on every tool turn.

The identical empty scaffold can also appear MID-history, wrapped around a
completed think-less assistant tool-call turn (observed live as
``kv_restore_divergence deep miss`` where the cached render carried the
scaffold and the incoming render did not). A tail-only strip never touched
those, so the #45 tail-only strip never resolved it. Since #50 the
treatment is SYMMETRIC: ``_strip_think_scaffold`` is neutralised (no-op)
and ``_symmetrize_think_scaffold`` keeps the scaffold on both sides of
every completed assistant turn boundary (the terminal primer included)—
checkpoint-time and restore-time renders normalise to byte-identical
prefixes through every turn. Real ``<think>...content...</think>`` blocks are a different
string and must never be touched.
"""

from vllm_mlx.utils.chat_template import (
    _GEN_THINK_SCAFFOLD,
    _normalize_think_scaffold,
    _strip_think_scaffold,
    _symmetrize_think_scaffold,
    apply_chat_template,
)

SCAFFOLD = _GEN_THINK_SCAFFOLD  # "<think>\n\n</think>\n\n"


def test_strip_helper_neutralised_keeps_trailing_scaffold():
    """#50: the strip pass is neutralised (early ``return prompt``)— the
    trailing generation-tail primer is no longer removed, and the
    symmetriser leaves an already-emitted scaffold untouched (no double
    scaffold)."""
    p = "<|im_start|>assistant\n" + SCAFFOLD
    assert _strip_think_scaffold(p) == p
    assert _strip_think_scaffold(p, preserve_trailing=True) == p
    assert _normalize_think_scaffold(p, assume_primer=True) == p


def test_strip_helper_noop_without_scaffold():
    # thinking-enabled generation ends with an OPEN think tag, must be untouched
    p = "<|im_start|>assistant\n<think>\n"
    assert _strip_think_scaffold(p) == p
    # plain text, no think at all
    q = "<|im_start|>assistant\nhello"
    assert _strip_think_scaffold(q) == q


def test_symmetrize_keeps_emitted_mid_history_scaffold_and_inserts_missing():
    """The exact shape of the observed live divergence (cached_next carried
    ``<think>\\n\\n</think>\\n\\n<tool_call>...``, incoming_next started at
    ``<tool_call>...``): the scaffold the template already emitted around a
    completed think-less assistant tool-call turn is KEPT (no double
    scaffold), while a completed assistant turn the template omitted it from
    (the ``preserve_thinking=False`` else branch— the #50 bug) gets it
    inserted at the turn start, byte-matching the generation primer."""
    p = (
        "<|im_start|>user\nfix the bug<|im_end|>\n"
        "<|im_start|>assistant\n" + SCAFFOLD + "<tool_call>\n"
        "<function=todowrite>...</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>tool\nok<|im_end|>\n"
        "<|im_start|>assistant\ndone<|im_end|>\n"
    )
    expected = (
        "<|im_start|>user\nfix the bug<|im_end|>\n"
        "<|im_start|>assistant\n" + SCAFFOLD + "<tool_call>\n"
        "<function=todowrite>...</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>tool\nok<|im_end|>\n"
        "<|im_start|>assistant\n" + SCAFFOLD + "done<|im_end|>\n"
    )
    assert _symmetrize_think_scaffold(p, assume_primer=True) == expected


def test_symmetrize_all_boundaries_carry_scaffold_and_idempotent():
    """Every completed assistant boundary ends up carrying exactly one
    emitted/inserted scaffold, and normalising twice yields the same bytes
    as normalising once (the stateless idempotence the KV restore relies
    on, #45/#50)."""
    at = "<|im_start|>assistant\n"
    p = at + SCAFFOLD + "<tool_call>a</tool_call>\n" + at + SCAFFOLD
    once = _normalize_think_scaffold(p, assume_primer=True)
    assert once == p
    assert once.count(SCAFFOLD) == 2
    assert _normalize_think_scaffold(once, assume_primer=True) == once


def test_strip_helper_preserves_real_reasoning_block():
    """A non-empty think block is a different string and must NOT be stripped."""
    p = (
        "<|im_start|>assistant\n"
        "<think>\nreal reasoning about the bug\n</think>\n\n"
        "<tool_call>bash</tool_call><|im_end|>\n"
    )
    assert _strip_think_scaffold(p) == p


def test_renders_differing_only_by_scaffold_normalize_identically():
    """The divergence-prevention property: a checkpoint-time render that
    contains the empty scaffold and a restore-time render that omits it must
    normalise to the SAME final string, so the token prefix cannot diverge.
    Since #50 the normalisation is symmetric— the scaffold is kept on the
    one side and inserted on the other, at every completed assistant
    boundary (not stripped from one side)."""
    prefix = "<|im_start|>user\nfix the bug<|im_end|>\n<|im_start|>assistant\n"
    suffix = (
        "<tool_call>\n<function=todowrite>...</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>tool\nok<|im_end|>\n<|im_start|>assistant\n"
    )
    with_scaffold = prefix + SCAFFOLD + suffix
    without_scaffold = prefix + suffix
    norm_with = _normalize_think_scaffold(with_scaffold, assume_primer=True)
    norm_without = _normalize_think_scaffold(without_scaffold, assume_primer=True)
    assert norm_with == norm_without
    # converged on the symmetric form: the scaffold sits at the assistant
    # boundary in BOTH renders, and the terminal generation-tail primer is
    # present (the runaway-``think`` guard survives, #45).
    assert norm_with.count(SCAFFOLD) == 2
    assert norm_with.endswith("<|im_start|>assistant\n" + SCAFFOLD)


class _FakeQwenApplicator:
    """Mimics the Qwen3.5 asymmetry: the generation prompt gets the empty
    scaffold when thinking is disabled, but completed assistant turns in history
    do not (this is the exact behaviour observed on real captured payloads)."""

    # When set, completed assistant tool-call turns in history ALSO carry the
    # empty scaffold — the mid-history variant of the same asymmetry.
    scaffold_in_history = False

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
            scaffold = ""
            if self.scaffold_in_history and m["role"] == "assistant" and tcs:
                scaffold = "<think>\n\n</think>\n\n"
            out.append(
                "<|im_start|>{}\n{}{}{}<|im_end|>\n".format(
                    m["role"], scaffold, content, tc_txt
                )
            )
        s = "".join(out)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if enable_thinking is False:
                s += "<think>\n\n</think>\n\n"  # the asymmetric scaffold
        return s


class _FakeQwenApplicatorScaffoldedHistory(_FakeQwenApplicator):
    scaffold_in_history = True


def test_generation_prompt_keeps_primer_when_thinking_disabled():
    """Runaway-reasoning fix: with enable_thinking=False +
    add_generation_prompt=True the Qwen3.5 template ends the prompt with the
    empty-think primer. That primer tells the model thinking is already done, so
    it emits the answer / tool_call immediately. The blind global strip (#30/#31)
    deleted it, leaving the prompt bare at ``<|im_start|>assistant\n``; the model
    then opened its own <think> and ran to max_tokens (~23 min). The primer must
    survive on the generation tail."""
    msgs = [{"role": "user", "content": "fix the bug"}]
    gen = apply_chat_template(
        _FakeQwenApplicator(), msgs, enable_thinking=False, model_name="qwen"
    )
    assert gen.endswith(SCAFFOLD), (
        "generation-tail empty-think primer was stripped (runaway-reasoning bug)"
    )
    # exactly one scaffold: the trailing primer, nothing left mid-prompt
    assert gen.count(SCAFFOLD) == 1


def test_generation_prompt_thinking_enabled_has_no_primer():
    """No-op case: thinking enabled never emits the empty primer (the real
    template ends with an OPEN ``<think>``), so nothing is preserved and nothing
    runs away for a different reason."""
    msgs = [{"role": "user", "content": "hi"}]
    gen = apply_chat_template(
        _FakeQwenApplicator(), msgs, enable_thinking=True, model_name="qwen"
    )
    assert SCAFFOLD not in gen
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
    # The generation tail carries the empty-think primer (runaway fix) and,
    # since #50, the completed assistant turn re-rendered as history carries
    # the SAME scaffold (symmetrised into the preserve_thinking=False else
    # branch). The two renders therefore agree byte-for-byte at the primer
    # boundary— no cold re-prefill on the follow-up turn (#50).
    assert gen_prompt.endswith(SCAFFOLD)
    base_prefix = gen_prompt[: -len(SCAFFOLD)]
    assert hist.startswith(base_prefix), (
        "history render diverges from the generation prompt BEFORE the primer:\n"
        f"gen : {base_prefix[-40:]!r}\nhist: {hist[len(base_prefix) - 40 : len(base_prefix) + 20]!r}"
    )
    # the boundary now AGREES: the history render carries the same
    # empty-think scaffold the checkpoint cached (symmetric treatment, #50).
    assert hist[len(base_prefix) :].startswith(SCAFFOLD)


def test_scaffolded_and_scaffold_free_history_renders_converge():
    """Two applicators that differ ONLY in whether completed assistant
    tool-call turns carry the empty scaffold mid-history must produce the
    SAME normalized prompt through apply_chat_template. This is the property
    that makes checkpoint renders and restore renders always match."""
    base = [{"role": "user", "content": "fix the bug"}]
    asst = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "todowrite"}}],
    }
    tool = {"role": "tool", "content": "ok"}
    msgs = base + [asst, tool]
    with_scaffold = apply_chat_template(
        _FakeQwenApplicatorScaffoldedHistory(),
        msgs,
        enable_thinking=False,
        model_name="qwen",
    )
    without_scaffold = apply_chat_template(
        _FakeQwenApplicator(), msgs, enable_thinking=False, model_name="qwen"
    )
    assert with_scaffold == without_scaffold, (
        "renders differing only by the mid-history empty scaffold did not "
        f"normalize identically:\nwith   : {with_scaffold!r}\nwithout: {without_scaffold!r}"
    )
    # symmetric treatment (#50): both renders carry the scaffold at the
    # completed assistant boundary AND the trailing generation primer—
    # byte-identical through every turn, so the KV restore can match the
    # checkpoint prefix (the runaway-``think`` guard survives, #45).
    assert with_scaffold.count(SCAFFOLD) == 2
    assert with_scaffold.endswith(SCAFFOLD)
