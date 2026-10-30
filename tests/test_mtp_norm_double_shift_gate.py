# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #106 — the MTP norm double-shift fixup is
gated on the ACTUAL mlx-lm sanitize precondition, not run unconditionally.

mlx-lm >= 0.32 ``qwen3_5.sanitize`` strips all ``mtp.*`` keys BEFORE the
norm shift and shifts only when at least one conv1d is stored raw-HF
(``shape[-1] != 1``). A checkpoint whose conv1d is already MLX-layout
(``shape[-1] == 1``) was never double-shifted — running the
``mean > 1.5`` fixup there corrupted the target model's legitimately
trained q/k_norms (mean ~1.5-1.7 → ~0.5-0.7) and the final ``norm``,
breaking MTP draft/target alignment (acceptance ~0, MTP slower than
off). The fixup must fire ONLY when a raw-HF conv1d is present.

Pinned here:

1. ``_has_unsanitized_conv1d`` — the gate predicate (MLX-layout →
   ``False``; raw-HF → ``True``; non-conv / non-3-D / missing weight
   entries ignored).
2. ``inject_mtp_support`` end-to-end on a fake model: MLX-layout conv1d
   + high-mean norm → the norm is left untouched and the boot log says
   the fixup was skipped; raw-HF conv1d → the fixup still applies
   (weight minus 1.0) and the "Fixed N" log fires.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

# ----------------------------------------------------------------------
# Minimal mlx stubs (no-mlx host; the inject path imports ``mlx.core`` /
# ``mlx.nn`` inside the function). Save/restore ``sys.modules``.
# ----------------------------------------------------------------------


class _Scalar:
    def __init__(self, value: float) -> None:
        self._v = value

    def item(self) -> float:
        return self._v


class _FakeArray:
    """1-D array stub carrying the ``astype`` / ``ndim`` / ``-`` surface
    ``_fix_mtp_norm_double_shift`` uses."""

    def __init__(self, values) -> None:
        self._values = tuple(float(v) for v in values)

    @property
    def ndim(self) -> int:
        return 1

    def astype(self, _dtype: object) -> _FakeArray:
        return self

    def __sub__(self, other: object) -> _FakeArray:
        return _FakeArray([v - other for v in self._values])


class _Conv1dStub:
    """Conv module stand-in — the gate reads ``type(mod).__name__`` and
    ``mod.weight.shape`` only."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.weight = SimpleNamespace(shape=shape)


class _RMSNormStub:
    """1-D norm stand-in the fixup's mean test can read / mutate."""

    def __init__(self, values) -> None:
        self.weight = _FakeArray(values)


class _FakeMTP:
    def parameters(self) -> dict:
        return {}


class _FakeInnerTextModel:
    """The shape ``_resolve_inner_text_model`` accepts in case 2
    (``model`` + ``args``) with the two iteration surfaces the inject
    uses: ``modules()`` for the gate, ``named_modules()`` for the fixup."""

    def __init__(self, args, named, module_list) -> None:
        self.args = args
        self.model = SimpleNamespace(layers=[])
        self._named = list(named)
        self._module_list = list(module_list)

    def modules(self):
        return self._module_list

    def named_modules(self):
        return self._named


def _args_ns(**overrides) -> SimpleNamespace:
    defaults = {
        "mtp_num_hidden_layers": 1,
        "hidden_size": 64,
        "_use_gated_attention": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture()
def mlx_stub(monkeypatch):
    """Stub ``mlx`` / ``mlx.core`` / ``mlx.nn`` for the inject path and
    restore the previous ``sys.modules`` state afterwards."""
    saved = {
        name: sys.modules.get(name)
        for name in ("mlx", "mlx.core", "mlx.nn", "mlx.utils")
    }

    mlx = ModuleType("mlx")
    mx = ModuleType("mlx.core")
    mx.float32 = "float32"
    mx.bfloat16 = "bfloat16"
    mx.uint32 = "uint32"
    mx.eval = lambda *a, **k: None
    mx.mean = lambda a: _Scalar(sum(a._values) / len(a._values))
    nn = ModuleType("mlx.nn")
    mlx.core = mx
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = mx
    sys.modules["mlx.nn"] = nn
    yield
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


@pytest.fixture()
def inject_scaffolding(monkeypatch):
    """Monkeypatch the heavy post-gate machinery of
    ``inject_mtp_support`` (MTP module build + process-global cache
    patches) so a fake model can drive the Step 0 gate end-to-end."""
    from vllm_mlx.spec_decode.mtp import cache_patch, head

    monkeypatch.setattr(head, "build_mtp_module", lambda *a, **k: _FakeMTP())
    monkeypatch.setattr(
        cache_patch, "patch_arrays_cache_rollback_state", lambda *a, **k: None
    )
    monkeypatch.setattr(
        cache_patch, "patch_gated_delta_net_for_mtp", lambda *a, **k: None
    )


# ----------------------------------------------------------------------
# _has_unsanitized_conv1d — the gate predicate
# ----------------------------------------------------------------------


def test_has_unsanitized_conv1d_false_for_mlx_layout(monkeypatch, mlx_stub):
    """conv1d already MLX-layout (shape[-1] == 1) → no raw-HF conv1d →
    sanitize applied no shift → gate is False."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _has_unsanitized_conv1d,
    )

    model = SimpleNamespace(modules=lambda: [_Conv1dStub((10240, 4, 1))])
    assert _has_unsanitized_conv1d(model) is False


def test_has_unsanitized_conv1d_true_for_raw_hf(monkeypatch, mlx_stub):
    """conv1d stored raw-HF (shape[-1] != 1) → sanitize shifted → gate
    is True and the fixup may fire."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _has_unsanitized_conv1d,
    )

    model = SimpleNamespace(modules=lambda: [_Conv1dStub((1, 10240, 4))])
    assert _has_unsanitized_conv1d(model) is True


def test_has_unsanitized_conv1d_ignores_non_conv_and_malformed(monkeypatch, mlx_stub):
    """Non-Conv modules, 2-D weights, and missing weights are ignored —
    only a 3-D conv1d with ``shape[-1] != 1`` flips the gate."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _has_unsanitized_conv1d,
    )

    class _Linear:
        weight = SimpleNamespace(shape=(64, 64))  # 2-D — not a conv1d

    model = SimpleNamespace(
        modules=lambda: [
            _Linear(),
            _Conv1dStub((64, 4)),  # 2-D conv — skipped
            SimpleNamespace(weight=None),  # no weight — skipped
            _RMSNormStub([1.0]),  # not a Conv — skipped
        ]
    )
    assert _has_unsanitized_conv1d(model) is False


def test_has_unsanitized_conv1d_false_with_no_modules(monkeypatch, mlx_stub):
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _has_unsanitized_conv1d,
    )

    assert _has_unsanitized_conv1d(SimpleNamespace(modules=lambda: [])) is False


# ----------------------------------------------------------------------
# inject_mtp_support — the Step 0 gate end-to-end
# ----------------------------------------------------------------------


_HIGH_MEAN_NORM_VALUES = [1.6, 1.6, 1.6]  # mean 1.6 > 1.5 threshold


def test_inject_mlx_layout_conv1d_skips_fixup(
    monkeypatch, mlx_stub, inject_scaffolding, caplog
):
    """A fake model with MLX-layout conv1d + a high-mean norm: the
    gate is False so ``_fix_mtp_norm_double_shift`` never runs — the
    norm is left untouched (no -1.0 corruption) and the boot log says
    the fixup was skipped. Regression for the 19 corrupted target
    norms (q/k_norm + final model.norm) behind the ~0 accept ratio."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    norm = _RMSNormStub(list(_HIGH_MEAN_NORM_VALUES))
    model = _FakeInnerTextModel(
        _args_ns(),
        [("model.layers.57.self_attn.q_norm", norm)],
        [_Conv1dStub((10240, 4, 1))],  # MLX layout — shape[-1] == 1
    )
    with caplog.at_level(
        logging.INFO, logger="vllm_mlx.spec_decode.mtp.qwen3_5_inject"
    ):
        assert inject_mtp_support(model, allow_random_init=True) is True

    assert list(norm.weight._values) == [1.6, 1.6, 1.6]
    assert "conv1d already MLX-layout" in caplog.text
    assert "double-shifted norm weights" not in caplog.text


def test_inject_raw_hf_conv1d_still_applies_fixup(
    monkeypatch, mlx_stub, inject_scaffolding, caplog
):
    """A fake model with raw-HF conv1d + a high-mean norm: the gate is
    True, the fixup still runs, the norm is shifted by -1.0, and the
    boot log reports the fixed count. Pins that the gate did not
    disable the original #87 fixup for checkpoints that DID shift."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    norm = _RMSNormStub(list(_HIGH_MEAN_NORM_VALUES))
    model = _FakeInnerTextModel(
        _args_ns(),
        [("model.layers.57.self_attn.q_norm", norm)],
        [_Conv1dStub((1, 10240, 4))],  # raw-HF — shape[-1] != 1
    )
    with caplog.at_level(
        logging.INFO, logger="vllm_mlx.spec_decode.mtp.qwen3_5_inject"
    ):
        assert inject_mtp_support(model, allow_random_init=True) is True

    assert all(abs(v - 0.6) < 1e-9 for v in norm.weight._values)
    assert "Fixed 1 double-shifted norm weights" in caplog.text
    assert "conv1d already MLX-layout" not in caplog.text
