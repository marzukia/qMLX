# SPDX-License-Identifier: Apache-2.0
"""QMLX_MTP_FIXED_K env gate for the MTP serve path (#111).

``fixed_k`` already exists in ``mtp_generate_step`` (harness-only
per #53 — the byte-exact d=3 gate pins depth through it) but the
serve call site never forwarded it, so it was unreachable in
production. This exposes it through an env gate for controller A/B.

A/B verdict (workload-matched, 27B oQ4e): slightly WORSE than
auto-K at both workloads — the EV controller is near-optimal
(the K=0 parks are correct EV on hard-text segments); fixed depth
is not the lever. The gate stays for future draft-quality changes
(if p1 rises, fixed K=2/K=3 becomes worth re-testing).

Pinned here:

1. Unset env = no ``fixed_k`` kwarg on the step call (behaviour
   identical to pre-#111); set env forwards the value.
2. ``0 <= QMLX_MTP_FIXED_K <= max_k`` is validated at BOOT
   (install time) with an actionable error — a bad value fails at
   construction, not mid-stream (#53 contract).
3. The #53 generator construction-time contract still holds
   (fixed_k outside [0, max_k] raises at construction).

No-MLX: numpy-backed stubs; ``sys.modules`` is save/restored so a
real MLX install is untouched.
"""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

# ----------------------------------------------------------------------
# mlx / mlx_lm stubs (same pattern as test_mtp_xtc_env_gate.py)
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

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _AnyBase()


@pytest.fixture()
def _mlx_stubs(monkeypatch):
    saved = {k: v for k, v in sys.modules.items() if k == "mlx" or k.startswith("mlx.")}
    # Cached imports of ours must be dropped too: if a preceding test
    # already imported the scheduler with a REAL mlx, its module-level
    # ``mx`` binding is the real one and the stubs below would be
    # dead weight. The WHOLE package + parent subpackages are dropped:
    # ``from vllm_mlx import scheduler`` / ``from
    # vllm_mlx.spec_decode.mtp import generator`` fall back to the
    # PARENT package's attribute when the submodule is missing from
    # sys.modules, and the stale attribute still holds the real-mlx
    # module object.
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
    # Restore-or-DROP: a module we re-imported under the stubs (and
    # that was NOT in sys.modules at setup) must be dropped, not
    # kept — otherwise a later real-mx test's
    # ``from vllm_mlx.spec_decode.mtp import generator`` finds the
    # stub-bound module in sys.modules and inherits the numpy-backed
    # mx.
    for k in (
        "vllm_mlx",
        "vllm_mlx.scheduler",
        "vllm_mlx.spec_decode",
        "vllm_mlx.spec_decode.mtp",
        "vllm_mlx.spec_decode.mtp.generator",
    ):
        if k in saved_ours:
            sys.modules[k] = saved_ours[k]
        else:
            sys.modules.pop(k, None)


# ----------------------------------------------------------------------
# Shared install + first-step driver (same shape as #110's)
# ---------------------------------------------------------------------------


def _install_and_first_step(monkeypatch, caplog, env, max_k=3, *, step=True):
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

    for key in ("QMLX_MTP_FIXED_K", "QMLX_XTC_PROB", "QMLX_XTC_THR"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    with caplog.at_level(logging.INFO):
        assert scheduler._install_mtp_vendored(batch_gen, model, max_k=max_k) is True
        if step:
            batch_gen._generation_batch._step()
    return captured, caplog, batch_gen


# ----------------------------------------------------------------------
# 1. Call site: env gate -> fixed_k kwarg (absent when unset)
# ---------------------------------------------------------------------------
def test_unset_env_has_no_fixed_k_kwarg(_mlx_stubs, monkeypatch, caplog):
    """Unset QMLX_MTP_FIXED_K -> the kwarg is ABSENT from the step
    call (behaviour identical to pre-#111, not merely None)."""
    captured, _, _ = _install_and_first_step(monkeypatch, caplog, env={})
    assert "fixed_k" not in captured


def test_set_env_forwards_fixed_k(_mlx_stubs, monkeypatch, caplog):
    captured, caplog, _ = _install_and_first_step(
        monkeypatch, caplog, env={"QMLX_MTP_FIXED_K": "2"}
    )
    assert captured["fixed_k"] == 2
    assert "fixed depth pinned: QMLX_MTP_FIXED_K=2" in caplog.text


def test_padded_env_value_is_stripped(_mlx_stubs, monkeypatch, caplog):
    captured, _, _ = _install_and_first_step(
        monkeypatch, caplog, env={"QMLX_MTP_FIXED_K": "  3  "}
    )
    assert captured["fixed_k"] == 3


def test_empty_env_string_means_unset(_mlx_stubs, monkeypatch, caplog):
    captured, _, _ = _install_and_first_step(
        monkeypatch, caplog, env={"QMLX_MTP_FIXED_K": ""}
    )
    assert "fixed_k" not in captured


def _install_only(monkeypatch, caplog, env, max_k=3):
    """Boot-path variant of the driver: install, no step."""
    from vllm_mlx import scheduler

    class _GB2:
        def _step(self):
            return [0], [np.array(0.0)]

    class _BG2:
        _generation_batch = _GB2()
        stop_tokens = None

    bg = _BG2()
    model = SimpleNamespace(mtp_forward=None, make_mtp_cache=None, mtp=None)

    for key in ("QMLX_MTP_FIXED_K", "QMLX_XTC_PROB", "QMLX_XTC_THR"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    with caplog.at_level(logging.INFO):
        ok = scheduler._install_mtp_vendored(bg, model, max_k=max_k)
    return ok, caplog


# ----------------------------------------------------------------------
# 2. Boot validation: 0 <= QMLX_MTP_FIXED_K <= max_k
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "env_val, max_k, match",
    [
        ("4", 3, r"QMLX_MTP_FIXED_K=4 must lie in \[0, max_k=3\]"),
        ("3", 2, r"QMLX_MTP_FIXED_K=3 must lie in \[0, max_k=2\]"),
        ("-1", 3, r"QMLX_MTP_FIXED_K=-1 must lie in \[0, max_k=3\]"),
        ("abc", 3, r"QMLX_MTP_FIXED_K must be an integer, got 'abc'"),
        ("1.5", 3, r"QMLX_MTP_FIXED_K must be an integer, got '1.5'"),
    ],
)
def test_boot_validation_rejects_bad_values(
    _mlx_stubs, monkeypatch, caplog, env_val, max_k, match
):
    """A bad value fails at BOOT (install time), not mid-stream —
    the #53 construction-time contract, promoted to the env gate."""
    import pytest as _pt

    with _pt.raises(ValueError, match=match):
        _install_only(monkeypatch, caplog, {"QMLX_MTP_FIXED_K": env_val}, max_k)


@pytest.mark.parametrize("env_val, max_k", [("0", 3), ("2", 3), ("3", 3), ("1", 5)])
def test_boot_validation_accepts_in_range(
    _mlx_stubs, monkeypatch, caplog, env_val, max_k
):
    ok, caplog = _install_only(
        monkeypatch, caplog, {"QMLX_MTP_FIXED_K": env_val}, max_k
    )
    assert ok is True
    assert "fixed depth pinned" in caplog.text


def test_unset_env_boot_is_quiet(_mlx_stubs, monkeypatch, caplog):
    """No env -> no fixed-depth boot line (byte-identical to
    pre-#111 boot output)."""
    ok, caplog = _install_only(monkeypatch, caplog, {})
    assert ok is True
    assert "fixed depth pinned" not in caplog.text


# ----------------------------------------------------------------------
# 2b. Step-time env re-parse: the per-step fallback (config pin
#     absent) raises the SAME actionable message as the boot gate.
#     The first-call construction try/except catches it, disables
#     MTP for the uid, and falls back to plain decode — so the
#     message surfaces in the construction-failed warning.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_val", ["abc", "1.5"])
def test_step_parse_non_integer_env_is_actionable(
    _mlx_stubs, monkeypatch, caplog, bad_val
):
    """A value that becomes non-integer between boot and the first
    step fails that step's env re-parse with the actionable 'must be
    an integer' message (not a bare ``int()`` ValueError), then
    falls back to plain decode for the request."""
    _, _, batch_gen = _install_and_first_step(
        monkeypatch, caplog, env={"QMLX_MTP_FIXED_K": "2"}, step=False
    )
    monkeypatch.setenv("QMLX_MTP_FIXED_K", bad_val)
    y, lps = batch_gen._generation_batch._step()
    assert f"QMLX_MTP_FIXED_K must be an integer, got {bad_val!r}" in caplog.text
    assert y == [0]


# ----------------------------------------------------------------------
# 3. The #53 generator construction-time contract (fixed_k outside
#    [0, max_k] fails at construction, not mid-stream)
# ---------------------------------------------------------------------------
def test_generator_fixed_k_outside_range_raises_at_construction(_mlx_stubs):
    """The #53 construction-time contract: fixed_k outside [0, max_k]
    raises at construction (first ``next()``), not mid-stream."""
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    class _FakeModel:
        layers = []

        def make_mtp_cache(self):
            return []

        def mtp_forward(self, *a, **k):
            return np.zeros((1, 1))

        def __call__(self, *a, **k):
            return np.zeros((1, 1)), np.zeros((1, 1, 8))

    gen = mtp_generate_step(
        np.array([1], dtype="uint32"),
        _FakeModel(),
        max_tokens=8,
        max_k=3,
        fixed_k=5,
        prompt_cache=[],
    )
    with pytest.raises(ValueError, match=r"fixed_k=5 must lie in \[0, max_k=3\]"):
        next(gen)
