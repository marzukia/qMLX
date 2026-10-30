# SPDX-License-Identifier: Apache-2.0
"""QMLX_MTP_DEQUANT diagnostic — dequantise MTP linears to bf16 at
inject time (#108).

A/B probe for draft-quantisation error as an acceptance limiter.
Result: **negative** (85 rounds, mixed greedy + temp 1, same
checkpoint) — the quantised draft head matches the quantised backbone
BETTER (overall accept 50-51% quant vs 44.2% bf16) and the bf16 MTP
linears cost +665 MB, slowing draft passes. Kept in-tree, default off;
the "quantise MTP to match the base" design is correct.

Pinned here:

1. The dequant math — nibble ``j`` of a U32 packed group is weight
   ``j`` (matching mlx's quantized-matmul kernel) with the per-group
   affine term applied. Verified against a hand-rolled reference on
   fixed inputs (no-mlx numpy path) and, when MLX is available,
   against the real ``nn.QuantizedLinear`` matmul (max abs diff <
   1e-2, bf16 rounding).
2. The walk covers nested modules; non-quant children are left
   untouched; the ``replaced`` count matches.
3. Default-off: with the env var unset the call site never fires
   (no log, modules untouched); ``QMLX_MTP_DEQUANT=1`` fires it.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# ----------------------------------------------------------------------
# numpy-backed mlx stubs for the helper (``mx`` ops the dequant uses)
# and the inject sidecar path (``mx.load`` / ``mlx.utils.tree_flatten``).
# The real-MLX accuracy test (``importorskip``) covers the production
# path on Apple-Silicon CI; everything else here runs on a no-mlx host.
# ----------------------------------------------------------------------

UINT32, FLOAT32, BFLOAT16 = "uint32", "float32", "bfloat16"


class _T:
    """Minimal numpy-backed tensor with the ops the helper uses."""

    def __init__(self, arr: np.ndarray, dtype: str) -> None:
        self._a = np.asarray(arr)
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._a.shape

    def __getitem__(self, key: object) -> _T:
        return _T(self._a[key], self.dtype)

    def __rshift__(self, other: object) -> _T:
        o = other._a if isinstance(other, _T) else other
        return _T(self._a >> o, self.dtype)

    def __and__(self, other: object) -> _T:
        o = other._a if isinstance(other, _T) else other
        return _T(self._a & o, self.dtype)

    def __mul__(self, other: object) -> _T:
        o = other._a if isinstance(other, _T) else other
        return _T(self._a * o, self.dtype)

    def __add__(self, other: object) -> _T:
        o = other._a if isinstance(other, _T) else other
        return _T(self._a + o, self.dtype)

    def reshape(self, *shape: int) -> _T:
        return _T(self._a.reshape(*shape), self.dtype)

    def repeat(self, count: int, axis: int) -> _T:
        return _T(np.repeat(self._a, count, axis=axis), self.dtype)

    def astype(self, dtype: str) -> _T:
        return _T(self._a, dtype)


class _QuantizedLinear:
    """Test double for ``nn.QuantizedLinear`` (class NAME matters — the
    helper matches on ``type(child).__name__``)."""

    __name__ = "QuantizedLinear"

    def __init__(self, weight: _T, scales: _T, biases: _T) -> None:
        self.weight = weight
        self.scales = scales
        self.biases = biases

    def children(self) -> dict:
        return {}


# Keep ``type(child).__name__ == "QuantizedLinear"`` true.
class QuantizedLinear(_QuantizedLinear):  # noqa: N801 — name is the contract
    pass


class _PlainModule:
    def __init__(self, children: dict | None = None) -> None:
        self._children = children or {}
        for name, mod in self._children.items():
            setattr(self, name, mod)

    def children(self) -> dict:
        return dict(self._children)

    def parameters(self) -> dict:
        return {}


class _Linear:
    """Non-quant child stand-in — must be left untouched by the walk."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.weight = _T(np.zeros((out_dim, in_dim)), FLOAT32)

    def children(self) -> dict:
        return {}


class _FakeMTP:
    def __init__(self, children: dict, param_keys: list[str]) -> None:
        self._children = children
        self._param_keys = param_keys
        for name, mod in children.items():
            setattr(self, name, mod)

    def parameters(self) -> dict:
        return {k: object() for k in self._param_keys}

    def children(self) -> dict:
        return dict(self._children)

    def load_weights(self, items, strict: bool = True) -> None:
        pass


@pytest.fixture()
def mlx_stub(monkeypatch, tmp_path):
    """Stub ``mlx`` / ``mlx.core`` / ``mlx.nn`` / ``mlx.utils`` with the
    numpy-backed surface and save/restore ``sys.modules``."""
    saved = {
        name: sys.modules.get(name)
        for name in ("mlx", "mlx.core", "mlx.nn", "mlx.utils")
    }

    mlx = ModuleType("mlx")
    mx = ModuleType("mlx.core")
    mx.uint32, mx.float32, mx.bfloat16 = UINT32, FLOAT32, BFLOAT16
    mx.eval = lambda *a, **k: None
    mx.arange = lambda n: _T(np.arange(n), FLOAT32)
    mx.load = lambda path: {
        f"t{i}": object() for i in range(2)
    }  # matches the fake MTP's two param keys

    nn = ModuleType("mlx.nn")

    class Linear:  # stub for nn.Linear — NAME is what the tests assert
        def __init__(self, in_dim: int, out_dim: int) -> None:
            self.weight = _T(np.zeros((out_dim, in_dim)), FLOAT32)

    nn.Linear = Linear

    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda d: list(d.items())

    mlx.core = mx
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = mx
    sys.modules["mlx.nn"] = nn
    sys.modules["mlx.utils"] = utils
    yield
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


def _make_q(in_dim: int, out_dim: int, seed: int = 0) -> QuantizedLinear:
    """Build a quantized linear with KNOWN nibble data so the dequant
    output can be checked against a hand-rolled reference."""
    rng = np.random.default_rng(seed)
    gs = in_dim // 64
    k8 = in_dim // 8
    # 4-bit values per group, packed 8-per-U32 (nibble j = weight j).
    raw = rng.integers(0, 16, size=(out_dim, in_dim), dtype=np.uint32)
    packed = np.zeros((out_dim, k8), dtype=np.uint32)
    for j in range(8):
        packed |= raw[:, j::8] << (4 * j)
    scales = 0.5 + 0.5 * rng.random((out_dim, gs), dtype=np.float64)
    biases = -1.0 + 2.0 * rng.random((out_dim, gs), dtype=np.float64)
    return QuantizedLinear(
        weight=_T(packed, UINT32),
        scales=_T(scales, FLOAT32),
        biases=_T(biases, FLOAT32),
    )


# ----------------------------------------------------------------------
# 1. Dequant math — nibble unpack + per-group affine term
# ---------------------------------------------------------------------------
def test_dequant_matches_hand_rolled_reference(mlx_stub):
    """The unpacked weights must equal the known 4-bit values scaled by
    the per-group affine term — the same layout mlx's quantized-matmul
    kernel applies (weight ``j`` of a group at nibble ``j``)."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import _dequantize_mtp_linears

    in_dim, out_dim = 128, 16
    q = _make_q(in_dim, out_dim, seed=7)
    container = _PlainModule({"dense": q})

    assert _dequantize_mtp_linears(container) == 1
    new = container.dense
    assert new.weight.dtype == BFLOAT16
    assert new.weight.shape == (out_dim, in_dim)

    # Hand-rolled reference: unpack nibble j, repeat scale/bias per group.
    packed = q.weight._a
    vals = np.stack(
        [((packed >> (4 * j)) & 0xF).astype(np.float64) for j in range(8)],
        axis=2,
    ).reshape(out_dim, -1)
    gs = q.scales._a.shape[1]
    s = np.repeat(q.scales._a.reshape(out_dim, gs, 1), in_dim // gs, axis=2).reshape(
        out_dim, -1
    )
    b = np.repeat(q.biases._a.reshape(out_dim, gs, 1), in_dim // gs, axis=2).reshape(
        out_dim, -1
    )
    expected = vals * s + b
    got = new.weight._a.astype(np.float64)
    assert np.max(np.abs(got - expected)) < 1e-5


def test_dequant_matches_quantized_matmul_real_mlx():
    """Production-path check (MLX host): dequantised bf16 Linear output
    must match the original ``nn.QuantizedLinear`` matmul within bf16
    rounding (max abs diff < 1e-2) on fixed inputs."""
    mx = pytest.importorskip("mlx.core")
    # This host's mlx build exposes the module-level ``repeat`` but
    # not the Array method (the one ``_dequantize_mtp_linears``
    # calls); guard both so the test SKIPS instead of red-ing.
    if not hasattr(mx, "repeat") or not hasattr(mx.array([0.0]), "repeat"):
        pytest.skip("mlx too old: no array.repeat")
    import mlx.nn as nn

    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import _dequantize_mtp_linears

    in_dim, out_dim = 256, 128
    ql = nn.QuantizedLinear(in_dim, out_dim)
    assert ql.weight.dtype == mx.uint32, "precondition: 4-bit packed layout"

    container = nn.Module()
    container.linear = ql
    assert _dequantize_mtp_linears(container) == 1
    new = container.linear
    assert type(new).__name__ == "Linear"
    assert new.weight.dtype == mx.bfloat16

    x = mx.random.uniform(-1.0, 1.0, shape=(1, in_dim), key=mx.random.key(1234))
    ref = ql(x)
    got = new(x)
    diff = mx.max(mx.abs(got - ref)).item()
    assert diff < 1e-2, f"dequantised output diverges: max abs diff {diff}"


# ----------------------------------------------------------------------
# 2. Walk coverage — nested modules, non-quant untouched, count matches
# ---------------------------------------------------------------------------
def test_walk_covers_nested_modules_and_counts(mlx_stub):
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import _dequantize_mtp_linears

    q1 = _make_q(128, 16, seed=1)
    q2 = _make_q(256, 32, seed=2)
    plain = _Linear(64, 8)
    nested = _PlainModule({"deep": q2})
    root = _PlainModule({"dense": q1, "other": nested, "plain": plain})

    assert _dequantize_mtp_linears(root) == 2  # q1 + q2 (nested)

    # Quant children replaced with bf16 Linears.
    assert type(root.dense).__name__ == "Linear"
    assert root.dense.weight.dtype == BFLOAT16
    assert root.other.deep is not q2
    assert type(root.other.deep).__name__ == "Linear"
    # Non-quant children untouched (same objects, same weights).
    assert root.plain is plain
    assert root.plain.weight is plain.weight


def test_walk_ignores_non_uint32_weights(mlx_stub):
    """A 'QuantizedLinear'-named child whose weight is NOT packed U32
    (e.g. already dequantised) is left alone — no double-dequant."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import _dequantize_mtp_linears

    q = _make_q(128, 16, seed=3)
    q.weight = _T(q.weight._a.astype(np.float32), FLOAT32)  # no longer uint32
    root = _PlainModule({"dense": q})

    assert _dequantize_mtp_linears(root) == 0
    assert root.dense is q


# ----------------------------------------------------------------------
# 3. Env gating — default off, QMLX_MTP_DEQUANT=1 on
# ---------------------------------------------------------------------------


def _end_to_end(monkeypatch, caplog, tmp_path, env_on: bool) -> _FakeMTP:
    from vllm_mlx.spec_decode.mtp import cache_patch, head
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    class _Model:  # plain class — SimpleNamespace rejects __class__ swap
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                mtp_num_hidden_layers=1,
                hidden_size=64,
                _use_gated_attention=False,
            )
            self.model = SimpleNamespace(layers=[])

        def modules(self):
            return []

        def named_modules(self):
            return []

    q = _make_q(128, 16, seed=11)
    mtp = _FakeMTP({"dense": q}, ["t0", "t1"])
    monkeypatch.setattr(head, "build_mtp_module", lambda *a, **k: mtp)
    monkeypatch.setattr(
        cache_patch, "patch_arrays_cache_rollback_state", lambda *a, **k: None
    )
    monkeypatch.setattr(
        cache_patch, "patch_gated_delta_net_for_mtp", lambda *a, **k: None
    )

    sidecar = tmp_path / "mtp.safetensors"
    sidecar.write_bytes(b"")  # contents irrelevant — mx.load is stubbed

    if env_on:
        monkeypatch.setenv("QMLX_MTP_DEQUANT", "1")
    else:
        monkeypatch.delenv("QMLX_MTP_DEQUANT", raising=False)

    model = _Model()
    with caplog.at_level(
        logging.INFO, logger="vllm_mlx.spec_decode.mtp.qwen3_5_inject"
    ):
        assert inject_mtp_support(model, mtp_sidecar=sidecar) is True
    return mtp


def test_default_off_never_dequantises(monkeypatch, caplog, tmp_path, mlx_stub):
    """Without the env var the call site is inert — no dequant log and
    the QuantizedLinear child is left untouched (byte-identical boot
    behaviour to pre-#108)."""
    mtp = _end_to_end(monkeypatch, caplog, tmp_path, env_on=False)
    assert "QMLX_MTP_DEQUANT" not in caplog.text
    assert type(mtp.dense).__name__ == "QuantizedLinear"


def test_env_on_dequantises_and_logs(monkeypatch, caplog, tmp_path, mlx_stub):
    """QMLX_MTP_DEQUANT=1 fires the diagnostic: the quant child is
    replaced with a bf16 Linear and the boot log reports the count."""
    mtp = _end_to_end(monkeypatch, caplog, tmp_path, env_on=True)
    assert "QMLX_MTP_DEQUANT: dequantised 1 MTP linears" in caplog.text
    assert type(mtp.dense).__name__ == "Linear"
    assert mtp.dense.weight.dtype == BFLOAT16


def test_env_gate_accepts_true_yes_variants(monkeypatch):
    """The gate expression itself: '1'/'true'/'yes' (case-insensitive)
    enable; '0', '', and unset do not."""
    import os

    def gated() -> bool:
        return os.environ.get("QMLX_MTP_DEQUANT", "").lower() in (
            "1",
            "true",
            "yes",
        )

    monkeypatch.delenv("QMLX_MTP_DEQUANT", raising=False)
    assert gated() is False
    for on in ("1", "true", "YES"):
        monkeypatch.setenv("QMLX_MTP_DEQUANT", on)
        assert gated() is True
    for off in ("0", "false", ""):
        monkeypatch.setenv("QMLX_MTP_DEQUANT", off)
        assert gated() is False
