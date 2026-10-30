# SPDX-License-Identifier: Apache-2.0
"""XTC env-gate wiring for the MTP generate step (#110).

XTC (typical-set filtering) was implemented in the sampler chain but
**no config path ever fed it** in the serve path — the call site
passed nothing, so ``xtc_probability`` was always 0. A/B verdict at
temp 1: **negative** — the absolute floor fights flat vocab
distributions (prob=1 thr=0.01 collapsed output to 11 tokens;
min_p-style relative floor is the working narrowing mechanism).
Kept in-tree, default off; the env form is a stopgap for A/B (if
XTC is kept, it belongs in the speculative config JSON).

Pinned here:

1. ``xtc_probability > 0`` flows from the env gate to the
   ``mtp_generate_step`` call site (end-to-end through
   ``_install_mtp_vendored`` + the first-call path of the installed
   ``_mtp_step``), and the boot log announces the A/B state only
   when the gate is on.
2. Default off = byte-identical behaviour: env unset ->
   ``xtc_probability 0.0`` -> the shared sampler chain skips XTC
   entirely (``xtc_cell is None``, no xtc step).
3. The shared draw keeps draft/verify exact: with a fixed logits
   pair and a fixed (seeded) draw, draft and verify reach the SAME
   apply/skip decision — the chain order is top_p -> min_p -> xtc
   -> top_k.

No-MLX: numpy-backed stubs for the small ``mx`` surface;
``sys.modules`` is save/restored so a real MLX install is untouched.
"""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

# ----------------------------------------------------------------------
# mlx / mlx_lm stubs (same pattern as test_mtp_prefix_length_metric.py)
# ----------------------------------------------------------------------


class _AnyBase:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _AnyBase()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _AnyBase()


@pytest.fixture()
def _mlx_stubs(monkeypatch):
    saved = {k: v for k, v in sys.modules.items() if k == "mlx" or k.startswith("mlx.")}
    # Cached imports of ours must be dropped too: if a preceding test
    # already imported the generator/scheduler with a REAL mlx, their
    # module-level ``mx`` binding is the real one and the stubs below
    # would be dead weight. The WHOLE package is dropped (not just
    # the two submodules): ``from vllm_mlx import scheduler`` /
    # ``from vllm_mlx.spec_decode.mtp import generator`` fall back to
    # the PARENT package's attribute when the submodule is missing
    # from sys.modules, and the stale attribute still holds the
    # real-mlx module object. Re-import under the stubs so every code
    # path under test sees the numpy-backed mx.
    saved_ours = {
        k: v
        for k, v in sys.modules.items()
        if k
        in (
            "vllm_mlx",
            "vllm_mlx.scheduler",
            "vllm_mlx.spec_decode",
            "vllm_mlx.spec_decode.mtp",
            "vllm_mlx.spec_decode.mtp.generator",
        )
    }
    for k in list(saved_ours):
        del sys.modules[k]

    def stub(name):
        m = _StubModule(name)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, name, m)
        return m

    mx = stub("mlx")
    mx.core = stub("mlx.core")
    mx.nn = stub("mlx.nn")
    mx.core.uint32 = "uint32"
    mx.core.array = lambda *a, **k: np.asarray(a[0] if a else [])
    mx.core.inf = float("inf")

    def _softmax(a, axis):
        a = np.asarray(a, dtype=float)
        m = a.max(axis=axis, keepdims=True)
        e = np.exp(a - m)
        return e / e.sum(axis=axis, keepdims=True)

    mx.core.softmax = _softmax
    mx.core.where = np.where
    mx.core.zeros_like = np.zeros_like
    mlm = stub("mlx_lm")
    for sub in (
        "generate",
        "sample_utils",
        "tokenizer_utils",
        "utils",
        "models",
        "quantize",
        "convert",
        "models.base",
        "models.cache",
        "models.gated_delta",
        "models.switch_layers",
    ):
        m = stub(f"mlx_lm.{sub}")
        setattr(mlm, sub.split(".")[-1], m)
    yield
    for k in [k for k in sys.modules if k == "mlx" or k.startswith("mlx.")]:
        del sys.modules[k]
    sys.modules.update(saved)
    sys.modules.update(saved_ours)


# ----------------------------------------------------------------------
# 1. End-to-end: env gate -> mtp_generate_step kwargs + boot log
# ---------------------------------------------------------------------------


def _install_and_first_step(monkeypatch, caplog, env):
    """Install the vendored MTP hot loop on a fake BatchGenerator and
    drive the first-call path of the installed ``_mtp_step``. Returns
    the captured ``mtp_generate_step`` kwargs."""
    from vllm_mlx import scheduler
    from vllm_mlx.spec_decode.mtp import generator as gen_mod

    captured: dict = {}

    def _capture(*a, **kw):
        captured.update(kw)

        def _gen():
            yield 1, np.array(0.0), False

        return _gen()

    monkeypatch.setattr(gen_mod, "mtp_generate_step", _capture)

    class _TokArr:
        """Dual-interface token array: ``[0].item()`` like an mx
        array, ``.astype(anything)`` identity (np's astype can't
        interpret a real-mx dtype class)."""

        def __init__(self, vals) -> None:
            self._v = np.asarray(vals)

        def __getitem__(self, i):
            return self._v[i]

        def astype(self, *a, **k):
            return self

    class _GB:
        def __init__(self) -> None:
            self.uids = [1]
            self.logits_processors = None
            self._next_tokens = _TokArr([7])
            self._next_logprobs = [np.array(0.0)]
            self.max_tokens = [80]
            self.tokens = [[]]
            self.prompt_cache = None

            def _step(self):
                return [0], [np.array(0.0)]

            self._step = _step.__get__(self)

    class _BG:
        _generation_batch = _GB()
        stop_tokens = None

    batch_gen = _BG()
    model = SimpleNamespace(mtp_forward=None, make_mtp_cache=None, mtp=None)

    for key in ("QMLX_XTC_PROB", "QMLX_XTC_THR"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    with caplog.at_level(logging.INFO, logger="vllm_mlx.scheduler"):
        assert scheduler._install_mtp_vendored(batch_gen, model) is True
        batch_gen._generation_batch._step()
    return captured, caplog


def test_xtc_default_off_zero_probability(_mlx_stubs, monkeypatch, caplog):
    """Env unset -> xtc_probability 0.0 (the chain skips XTC —
    byte-identical to pre-#110) and the boot log has NO XTC line."""
    captured, caplog = _install_and_first_step(monkeypatch, caplog, env={})
    assert captured["xtc_probability"] == 0.0
    # Default threshold is 0.01 (dormant — prob 0 skips the step).
    assert captured["xtc_threshold"] == 0.01
    assert "XTC active" not in caplog.text


def test_xtc_env_gate_flows_to_step(_mlx_stubs, monkeypatch, caplog):
    """QMLX_XTC_PROB / QMLX_XTC_THR land on the step's kwargs and the
    boot log announces the A/B state."""
    captured, caplog = _install_and_first_step(
        monkeypatch,
        caplog,
        env={"QMLX_XTC_PROB": "1", "QMLX_XTC_THR": "0.001"},
    )
    assert captured["xtc_probability"] == 1.0
    assert captured["xtc_threshold"] == 0.001
    assert "[MTP-vendored] XTC active: prob=1 thr=0.001" in caplog.text


def test_xtc_empty_env_string_is_off(_mlx_stubs, monkeypatch, caplog):
    """QMLX_XTC_PROB='' (set but empty) -> 0.0 (the ``or 0.0`` guard)."""
    captured, _ = _install_and_first_step(
        monkeypatch, caplog, env={"QMLX_XTC_PROB": ""}
    )
    assert captured["xtc_probability"] == 0.0


# ----------------------------------------------------------------------
# 2. Default off = chain skips XTC entirely
# ---------------------------------------------------------------------------
def test_default_off_chain_skips_xtc(_mlx_stubs):
    from vllm_mlx.spec_decode.mtp.generator import _make_sampler_chain

    # No params: empty chain, no xtc cell.
    chain, cell = _make_sampler_chain()
    assert chain == []
    assert cell is None

    # Sampling params set but xtc_probability=0 (the default): the
    # chain has top_p + top_k only — NO xtc step, no xtc cell.
    chain, cell = _make_sampler_chain(top_p=0.95, top_k=20)
    assert len(chain) == 2
    assert cell is None

    # prob > 0: the xtc step is present, between min_p and top_k.
    chain, cell = _make_sampler_chain(
        top_p=0.95, min_p=0.01, top_k=20, xtc_probability=1.0, xtc_threshold=0.01
    )
    assert len(chain) == 4
    assert cell is not None


def test_chain_order_is_top_p_then_min_p_then_xtc_then_top_k(_mlx_stubs):
    """The shared sampler chain order is top_p -> min_p -> xtc -> top_k
    (the draw shared across draft and verify keeps acceptance exact)."""
    from pathlib import Path

    src = (
        Path("vllm_mlx/spec_decode/mtp/generator.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    stripped = [line.strip() for line in src]
    order = [
        "chain.append(lambda x: apply_top_p(x, top_p))",
        "chain.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))",
        "chain.append(_xtc)",
        "chain.append(lambda x: apply_top_k(x, top_k))",
    ]
    idx = [stripped.index(line) for line in order]
    assert idx == sorted(idx), f"chain order broke: {idx}"


# ----------------------------------------------------------------------
# 3. Shared draw keeps draft/verify exact (fixed logits, seeded)
# ---------------------------------------------------------------------------
def _fixed_logit_pair():
    """Fixed (seeded) 3-token logits. softmax([0, -0.5, -3]) ≈
    [0.58, 0.42, 0.01]: at thr=0.2 only token 2 falls below threshold
    while tokens 0/1 stay above (n_above = 2 > 1, so the all-or-nothing
    guard does NOT zero the mask)."""
    draft_logits = np.array([0.0, -0.5, -3.0])
    verify_logits = np.array([0.5, -0.2, -3.0])
    return draft_logits, verify_logits


def test_shared_draw_same_apply_decision_draft_verify(_mlx_stubs):
    from vllm_mlx.spec_decode.mtp.generator import _apply_xtc_with_shared_draw

    draft_logits, verify_logits = _fixed_logit_pair()
    special: list[int] = []

    # Seeded draw BELOW the probability: the filter APPLIES on BOTH
    # the draft and the verify pass (same draw -> same decision).
    # Vendored XTC semantics: mask = probs > threshold; masked
    # (above-threshold) tokens are set to -inf, low-prob tokens
    # survive. p(2) ~ 0.01 < 0.2 -> token 2 survives on both.
    p_draw = np.array(0.02)
    thr = 0.2
    out_draft = _apply_xtc_with_shared_draw(draft_logits, 0.5, thr, special, p_draw)
    out_verify = _apply_xtc_with_shared_draw(verify_logits, 0.5, thr, special, p_draw)
    assert out_draft[0] == float("-inf")
    assert out_draft[1] == float("-inf")
    assert out_draft[2] == draft_logits[2]
    assert out_verify[0] == float("-inf")
    assert out_verify[1] == float("-inf")
    assert out_verify[2] == verify_logits[2]

    # Seeded draw AT/ABOVE the probability: the filter SKIPS on both —
    # outputs are the unmodified logits (exact, not just equal-in-mask).
    p_draw_hi = np.array(0.9)
    out_draft = _apply_xtc_with_shared_draw(draft_logits, 0.5, thr, special, p_draw_hi)
    out_verify = _apply_xtc_with_shared_draw(
        verify_logits, 0.5, thr, special, p_draw_hi
    )
    np.testing.assert_array_equal(out_draft, draft_logits)
    np.testing.assert_array_equal(out_verify, verify_logits)


def test_shared_draw_special_tokens_unmasked(_mlx_stubs):
    from vllm_mlx.spec_decode.mtp.generator import _apply_xtc_with_shared_draw

    logits = np.array([0.0, -0.5, -6.0])  # p(0) ~ 0.62 > thr 0.02
    p_draw = np.array(0.01)  # below prob 0.5 -> filter applies
    out = _apply_xtc_with_shared_draw(logits, 0.5, 0.02, [0], p_draw)
    # Token 0 is above threshold (would be -inf) BUT special ->
    # force-unmasked -> survives at its original logit.
    assert out[0] == logits[0]
    assert out[1] == float("-inf")
    assert out[2] == logits[2]


def test_threshold_outside_range_raises(_mlx_stubs):
    from vllm_mlx.spec_decode.mtp.generator import _apply_xtc_with_shared_draw

    with pytest.raises(ValueError, match="xtc_threshold must be in"):
        _apply_xtc_with_shared_draw(np.array([0.0]), 0.5, 0.9, [], np.array(0.1))


def test_p_draw_none_delegates_to_upstream(_mlx_stubs, monkeypatch):
    """p_draw=None -> upstream's bare apply_xtc (fresh internal draw)."""
    from vllm_mlx.spec_decode.mtp.generator import _apply_xtc_with_shared_draw

    calls: list = []

    class _Upstream:
        @staticmethod
        def apply_xtc(logits, prob, thr, special):
            calls.append((prob, thr))
            return logits

    import types

    m = types.ModuleType("mlx_lm.sample_utils")
    m.apply_xtc = _Upstream.apply_xtc
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", m)

    out = _apply_xtc_with_shared_draw(np.array([0.0]), 0.5, 0.01, [], None)
    assert calls == [(0.5, 0.01)]
    np.testing.assert_array_equal(out, np.array([0.0]))
