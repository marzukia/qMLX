"""Round-trip stability for symmetric empty-think scaffold treatment (#50).

Regression guard for the multi-turn re-prefill bug: the Qwen3.5 chat template
is asymmetric about the empty think scaffold ``\nthink\n\n/think\n\n``. With
``enable_thinking=False`` + ``add_generation_prompt=True`` the GENERATION prompt
ends with the scaffold (the generation primer: "thinking is already done, emit
the answer / tool_call"), but the same assistant turn re-rendered as HISTORY
omits it (the template's history guard is gated on ``preserve_thinking``, which
is False for ``enable_thinking=False``). The cached checkpoint prefix and the
next turn's prefix therefore diverged at the assistant boundary, and every
follow-up turn cold-prefilled the entire conversation.

The fix (qMLX-side prompt normalisation, not a model-file patch):

* ``_strip_think_scaffold`` is neutralised (the verified issue #50
  workaround: it no longer deletes scaffolds, so the runaway-reasoning primer
  on the generation tail survives);
* ``_symmetrize_think_scaffold`` re-inserts the exact empty scaffold at every
  completed assistant turn the template omitted it from, deriving the
  turn-start marker from the rendered prompt itself.

Net effect: for a template that emits the generation primer, every completed
assistant turn AND the terminal primer carry the scaffold, so a checkpoint-time
render and the next turn's render are byte-identical through every completed
assistant turn. The normalisation is stateless and idempotent; real (non-empty)
think blocks and templates that emit no primer are untouched (no spurious
scaffold is added or removed).

These tests run in the no-MLX Linux CI job (template rendering is mlx-free).
"""

from vllm_mlx.utils.chat_template import (
    _GEN_THINK_SCAFFOLD,
    _normalize_think_scaffold,
    _strip_think_scaffold,
    _symmetrize_think_scaffold,
    apply_chat_template,
)

SCAFFOLD = _GEN_THINK_SCAFFOLD  # "\nthink\n\n/think\n\n"


class _FakeQwenApplicator:
    """Mimics the Qwen3.5 asymmetry: the generation prompt gets the empty
    scaffold when thinking is disabled, but completed assistant turns in
    history do not (the exact behaviour observed on real captured payloads).
    With ``scaffold_in_history`` the template is "patched": completed turns
    carry the scaffold too (the symmetric template the model file ideally
    would have)."""

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
            tc_txt = "".join(tc["function"]["name"] for tc in tcs)
            scaffold = ""
            if self.scaffold_in_history and m["role"] == "assistant" and tcs:
                scaffold = SCAFFOLD
            out.append(
                "\n<|im_start|>{}\n{}{}{}\n".format(
                    m["role"], scaffold, content, tc_txt
                )
            )
        s = "".join(out)
        if add_generation_prompt:
            s += "\n<|im_start|>assistant\n"
            if enable_thinking is False:
                s += SCAFFOLD  # the asymmetric generation primer
        return s


class _FakeQwenApplicatorScaffoldedHistory(_FakeQwenApplicator):
    """The "patched" template: completed assistant turns carry the scaffold
    too (symmetric). Normalisation must make the unpatched template converge
    to exactly these bytes."""

    scaffold_in_history = True


class _FakeNonThinkingApplicator:
    """A template that emits no primer at all (non-thinking model): no
    scaffold on the generation tail. Normalisation must be a no-op - no
    spurious scaffold may be added."""

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kw,
    ):
        out = []
        for m in messages:
            content = m.get("content") or ""
            tcs = m.get("tool_calls") or []
            tc_txt = "".join(tc["function"]["name"] for tc in tcs)
            out.append("\n{}\n{}{}\n".format(m["role"], content, tc_txt))
        s = "".join(out)
        if add_generation_prompt:
            s += "\n<|im_start|>assistant\n"
        return s


def _tool_call(name):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name}}],
    }


def test_strip_helper_neutralized_is_noop():
    """The verified issue #50 workaround: _strip_think_scaffold no longer
    deletes scaffolds (trailing primer, mid-history, or none)."""
    p_trailing = "\nassistant\n" + SCAFFOLD
    assert _strip_think_scaffold(p_trailing) == p_trailing
    p_mid = "\nassistant\n" + SCAFFOLD + "A\n" + "\nassistant\n" + SCAFFOLD + "B"
    assert _strip_think_scaffold(p_mid) == p_mid
    p_plain = "\nassistant\nplain"
    assert _strip_think_scaffold(p_plain) == p_plain
    assert _strip_think_scaffold(123) == 123  # non-str passes through


def test_single_turn_generation_prompt_primer_intact():
    """No completed assistant turns: the generation primer survives and the
    normalisation is a no-op (nothing to symmetrise)."""
    gen = apply_chat_template(
        _FakeQwenApplicator(),
        [{"role": "user", "content": "fix the bug"}],
        enable_thinking=False,
        model_name="qwen",
    )
    assert gen.endswith(SCAFFOLD)
    assert gen.count(SCAFFOLD) == 1
    assert _normalize_think_scaffold(gen) == gen


def test_multi_turn_thinking_off_cached_prefix_is_reused():
    """The #50 bug case: over a multi-turn thinking-off conversation the
    cached generation prompt must be a byte-exact PREFIX of the next turn's
    render (no re-prefill of the whole conversation)."""
    fake = _FakeQwenApplicator()
    turn1 = [{"role": "user", "content": "fix the bug"}]
    cached_gen = apply_chat_template(
        fake, turn1, enable_thinking=False, model_name="qwen"
    )
    assert cached_gen.endswith(SCAFFOLD)
    # the model emitted a tool call; turn 2 re-renders the full history
    turn2 = turn1 + [
        _tool_call("bash"),
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "and now deploy"},
    ]
    next_gen = apply_chat_template(
        fake, turn2, enable_thinking=False, model_name="qwen"
    )
    # THE invariant: the checkpoint prefix is reused byte-for-byte.
    assert next_gen.startswith(cached_gen)
    # every completed assistant turn carries the scaffold (symmetry)
    body = next_gen[: -len(SCAFFOLD)]  # drop the terminal primer
    assert body.count(SCAFFOLD) == 1  # the completed turn, and only that
    # idempotent: a second normalisation pass changes nothing
    assert _normalize_think_scaffold(next_gen) == next_gen


def test_multi_turn_three_turns_stable_through_every_turn():
    """Extend the invariant across three turns: each turn's generation
    prompt is a prefix of the next turn's render."""
    fake = _FakeQwenApplicator()
    msgs = [{"role": "user", "content": "fix the bug"}]
    prev = apply_chat_template(fake, msgs, enable_thinking=False, model_name="qwen")
    for tool_name, tool_result in [
        ("bash", {"role": "tool", "content": "ok"}),
        ("todowrite", {"role": "tool", "content": "done"}),
    ]:
        msgs = msgs + [
            _tool_call(tool_name),
            tool_result,
            {"role": "user", "content": "step " + tool_name},
        ]
        cur = apply_chat_template(fake, msgs, enable_thinking=False, model_name="qwen")
        assert cur.startswith(prev), "divergence for " + tool_name
        prev = cur
    # all completed turns scaffolded + terminal primer
    assert cur.count(SCAFFOLD) == 3


def test_thinking_enabled_no_scaffold_and_no_regression():
    """With enable_thinking=True the template never emits the empty scaffold
    (the real one ends with an OPEN think tag): the normalisation must be a
    no-op (no regression for the thinking path)."""
    fake = _FakeQwenApplicator()
    msgs = [
        {"role": "user", "content": "hi"},
        _tool_call("bash"),
        {"role": "tool", "content": "ok"},
    ]
    gen = apply_chat_template(fake, msgs, enable_thinking=True, model_name="qwen")
    assert SCAFFOLD not in gen
    assert _normalize_think_scaffold(gen) == gen
    assert _symmetrize_think_scaffold(gen) == gen


def test_non_thinking_template_no_spurious_scaffold():
    """A template that emits no primer (non-thinking model) must not gain a
    scaffold: the gate is "the prompt ends with the scaffold", so no primer
    means no-op."""
    fake = _FakeNonThinkingApplicator()
    msgs = [
        {"role": "user", "content": "hi"},
        _tool_call("bash"),
        {"role": "tool", "content": "ok"},
    ]
    gen = apply_chat_template(fake, msgs, model_name="gpt")
    assert SCAFFOLD not in gen
    assert _normalize_think_scaffold(gen) == gen
    assert _symmetrize_think_scaffold(gen) == gen


def test_real_think_block_untouched():
    """A completed turn carrying a REAL (non-empty) think block is a
    different string from the empty scaffold and must be left untouched."""
    p = (
        "\nassistant\n"
        + "<think>\nreal reasoning about the bug\n</think>\n\n"
        + "bash\n"
    )
    assert _symmetrize_think_scaffold(p) == p
    # with the primer gate satisfied, the real block still wins
    p2 = p + "\nassistant\n" + SCAFFOLD
    assert _symmetrize_think_scaffold(p2) == p2


def test_patched_and_unpatched_templates_converge():
    """The "patched" template (scaffold in history) and the unpatched one
    must normalise to the SAME bytes through apply_chat_template - this is
    the property that makes checkpoint renders and restore renders match
    regardless of which template version rendered the checkpoint."""
    msgs = [
        {"role": "user", "content": "fix the bug"},
        _tool_call("bash"),
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "deploy"},
        _tool_call("todowrite"),
        {"role": "tool", "content": "saved"},
    ]
    raw_patched = _FakeQwenApplicatorScaffoldedHistory().apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    raw_unpatched = _FakeQwenApplicator().apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert raw_patched != raw_unpatched  # raw renders still differ (asymmetry)
    patched = apply_chat_template(
        _FakeQwenApplicatorScaffoldedHistory(),
        msgs,
        enable_thinking=False,
        model_name="qwen",
    )
    unpatched = apply_chat_template(
        _FakeQwenApplicator(), msgs, enable_thinking=False, model_name="qwen"
    )
    assert patched == unpatched  # converged through the wrapper (the fix)
    assert _normalize_think_scaffold(unpatched) == _normalize_think_scaffold(patched)
    # the converged bytes: every completed turn + the terminal primer
    assert patched.endswith(SCAFFOLD)
    assert patched.count(SCAFFOLD) == 3
    # idempotent
    assert _normalize_think_scaffold(patched) == patched


def test_symmetrize_idempotent_across_shapes():
    """Normalising an already-symmetrised prompt is a no-op for every
    shape: scaffolded, un-scaffolded, mixed, real-think, no-primer."""
    shapes = [
        "\nassistant\n" + SCAFFOLD,
        "\nassistant\n" + SCAFFOLD + "A\n" + "\nassistant\n" + SCAFFOLD,
        "\nassistant\n" + "A\n" + "\nassistant\n" + SCAFFOLD,
        "\nassistant\n"
        + "<think>\nreal\n</think>\n\n"
        + "A"
        + "\n"
        + "\nassistant\n"
        + SCAFFOLD,
        "\nassistant\n" + "A\n",  # no primer -> no-op
    ]
    for s in shapes:
        once = _symmetrize_think_scaffold(s)
        assert _symmetrize_think_scaffold(once) == once, f"not idempotent for {s!r}"


def test_normalize_equals_symmetrize_since_strip_is_noop():
    """_normalize_think_scaffold is strip (now a no-op) + symmetrise, so it
    must equal _symmetrize_think_scaffold on every input."""
    shapes = [
        "\nassistant\n" + SCAFFOLD + "A" + SCAFFOLD,
        "\nassistant\n" + "<think>\nreal\n</think>\n\n" + "A\n",
        123,
        "\nassistant\nplain",
    ]
    for s in shapes:
        assert _normalize_think_scaffold(s) == _symmetrize_think_scaffold(s)
