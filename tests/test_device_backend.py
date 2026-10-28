# SPDX-License-Identifier: Apache-2.0
"""
Metal backend parity for the device/backend seam (item 2 of 9 — the
conformance seed for the item-7 backend conformance suite).

The Metal backend must return EXACTLY the values the current call
sites get from ``mlx`` directly. ``mlx`` is not installed on this CI
(Linux), so parity is proven against a fake ``mlx.core`` with the same
API surface: every assertion compares the seam's answer to the
*literal* expression the call sites use today (plus the literal
expected value, so the test cannot degenerate into a tautology).

On an Apple-Silicon machine ``test_parity_against_real_mlx`` runs the
same comparison against the real ``mlx.core`` — the fake is only a
stand-in for the direct-call side.
"""

from __future__ import annotations

import ast
import importlib.resources

import pytest

from vllm_mlx.device import (
    Backend,
    BackendKind,
    EnforcementMode,
    MetalBackend,
    StreamModel,
    get_backend,
    register_backend,
    reset_registry,
)


def _real_mx_or_none():
    try:
        import mlx.core as mx

        return mx
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fake mlx.core — same API surface the seam uses
# ---------------------------------------------------------------------------


class _FakeArray:
    def __init__(self, values):
        self._values = list(values)

    def __add__(self, other):
        return _FakeArray([a + b for a, b in zip(self._values, other._values)])

    def item(self):
        return self._values[0]


class _FakeMetal:
    def __init__(self, available=True):
        self._available = available

    def is_available(self):
        return self._available


class _FakeStreamContext:
    def __init__(self, stream, enter_error=None):
        self._stream = stream
        self._enter_error = enter_error

    def __enter__(self):
        if self._enter_error is not None:
            raise self._enter_error
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMx:
    """Stand-in for ``mlx.core``.

    Defaults model a full-featured M1–M4-class build: multi-stream,
    fast kernels, flash attention, quantize/dequantize, compile,
    async_eval.
    """

    def __init__(
        self,
        *,
        metal_available=True,
        device_info=None,
        active_bytes=100 * 2**20,
        peak_bytes=200 * 2**20,
        cache_bytes=50 * 2**20,
        single_stream=False,
        stream_error=None,
        new_thread_local_stream_api=True,
        fast_kernels=True,
        flash_attention=True,
        quantized_kv=True,
        compile_through=True,
        async_eval=True,
    ):
        self.metal = _FakeMetal(metal_available)
        self._device_info = (
            dict(device_info)
            if device_info is not None
            else {
                "max_recommended_working_set_size": 16 * 2**30,
                "memory_size": 32 * 2**30,
                "device_name": "Apple M4 Max",
            }
        )
        self._device = "gpu0"
        self._active = active_bytes
        self._peak = peak_bytes
        self._cache = cache_bytes
        if stream_error is None and single_stream:
            stream_error = RuntimeError("There is no Stream(gpu, 1) in current thread.")
        self._stream_error = stream_error
        self._tls_api = new_thread_local_stream_api
        self._memory_limit = None
        self._cache_limit = None
        if new_thread_local_stream_api:
            self.new_thread_local_stream = self._new_thread_local_stream
        if fast_kernels:
            fast = type("_Fast", (), {"rms_norm": staticmethod(lambda *a, **k: None)})
            if flash_attention:
                fast.scaled_dot_product_attention = staticmethod(lambda *a, **k: None)
            self.fast = fast()
        if quantized_kv:
            self.quantize = lambda *a, **k: None
            self.dequantize = lambda *a, **k: None
        if compile_through:
            self.compile = lambda f: f
        if async_eval:
            self.async_eval = lambda *a, **k: None

    def _new_thread_local_stream(self, device):
        return ("thread_local_stream", device)

    def device_info(self):
        return dict(self._device_info)

    def default_device(self):
        return self._device

    def default_stream(self, device):
        return ("default_stream", device)

    def stream(self, stream):
        return _FakeStreamContext(stream, self._stream_error)

    def array(self, values):
        return _FakeArray(values)

    def get_active_memory(self):
        return self._active

    def get_peak_memory(self):
        return self._peak

    def get_cache_memory(self):
        return self._cache

    def set_memory_limit(self, n):
        self._memory_limit = n

    def set_cache_limit(self, n):
        self._cache_limit = n


# ---------------------------------------------------------------------------
# Import contract
# ---------------------------------------------------------------------------


def test_package_imports_without_mlx():
    """``import vllm_mlx.device`` must work on a system without MLX
    (metadata-only access, Linux CI) — i.e. no module in the package
    may import ``mlx`` at module level. AST audit of top-level imports
    (same convention as ``test_mlx_compat``)."""
    import vllm_mlx.device  # noqa: F401  (this machine has no mlx)

    pkg = importlib.resources.files("vllm_mlx.device")
    for name in ("__init__.py", "backend.py", "metal.py"):
        tree = ast.parse(pkg.joinpath(name).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                assert not (mod == "mlx" or mod.startswith("mlx.")), (
                    f"module-level mlx import in vllm_mlx/device/{name}: {mod}"
                )


def test_metal_backend_satisfies_protocol():
    assert isinstance(MetalBackend(mx=FakeMx()), Backend)


# ---------------------------------------------------------------------------
# Parity: seam answers == literal call-site expressions
# ---------------------------------------------------------------------------


def test_identity_parity():
    mx = FakeMx()
    b = MetalBackend(mx=mx)
    assert b.kind == BackendKind.METAL
    assert b.kind == "metal"
    assert b.device() == mx.default_device()
    assert b.device_name() == mx.device_info().get("device_name", "")
    assert b.device_name() == "Apple M4 Max"
    assert b.total_memory_bytes() == mx.device_info().get("memory_size", 0)
    assert b.total_memory_bytes() == 32 * 2**30


def test_is_available_parity():
    mx = FakeMx()
    assert MetalBackend(mx=mx).is_available() == mx.metal.is_available()
    mx_off = FakeMx(metal_available=False)
    assert MetalBackend(mx=mx_off).is_available() == mx_off.metal.is_available()
    # Ported build: no mx.metal attribute → False, never raises.
    mx_no_metal = FakeMx()
    del mx_no_metal.metal
    assert MetalBackend(mx=mx_no_metal).is_available() is False


def test_recommended_working_set_parity():
    # Both keys present — call sites take max_recommended.
    mx = FakeMx()
    b = MetalBackend(mx=mx)
    direct = mx.device_info().get(
        "max_recommended_working_set_size",
        mx.device_info().get("memory_size", 0),
    )
    assert b.recommended_working_set_bytes() == direct == 16 * 2**30
    # max_recommended missing → memory_size fallback (exact call-site
    # expression, batched.py / scheduler.py / engine_core.py).
    mx_fb = FakeMx(device_info={"memory_size": 64 * 2**30})
    b_fb = MetalBackend(mx=mx_fb)
    direct_fb = mx_fb.device_info().get(
        "max_recommended_working_set_size",
        mx_fb.device_info().get("memory_size", 0),
    )
    assert b_fb.recommended_working_set_bytes() == direct_fb == 64 * 2**30
    # Neither key → 0 (call sites then apply their own policy).
    mx_empty = FakeMx(device_info={})
    assert MetalBackend(mx=mx_empty).recommended_working_set_bytes() == 0


def test_memory_stats_parity():
    mx = FakeMx(active_bytes=111, peak_bytes=222, cache_bytes=333)
    b = MetalBackend(mx=mx)
    stats = b.memory_stats()
    assert stats.active_bytes == mx.get_active_memory() == 111
    assert stats.peak_bytes == mx.get_peak_memory() == 222
    assert stats.cache_bytes == mx.get_cache_memory() == 333


def test_memory_limit_parity():
    mx = FakeMx()
    b = MetalBackend(mx=mx)
    b.set_memory_limit(1234)
    b.set_cache_limit(5678)
    assert mx._memory_limit == 1234
    assert mx._cache_limit == 5678


def test_enforcement_mode_declared():
    assert MetalBackend(mx=FakeMx()).enforcement_mode == EnforcementMode.HINT
    # str-enum: JSON-serializable, comparable to the bare string.
    assert EnforcementMode.HINT == "hint"
    assert EnforcementMode.HARD == "hard"


def test_default_stream_parity():
    mx = FakeMx()
    b = MetalBackend(mx=mx)
    assert b.default_stream() == mx.default_stream(mx.default_device())


def test_capability_flags_parity():
    mx = FakeMx()
    caps = MetalBackend(mx=mx).capabilities()
    assert caps.fast_kernels == hasattr(mx, "fast")
    assert caps.flash_attention == (
        hasattr(mx, "fast") and hasattr(mx.fast, "scaled_dot_product_attention")
    )
    assert caps.quantized_kv == (hasattr(mx, "quantize") and hasattr(mx, "dequantize"))
    assert caps.compile_through == hasattr(mx, "compile")
    assert caps.async_eval == hasattr(mx, "async_eval")
    # Full-featured build.
    assert (
        caps.fast_kernels,
        caps.flash_attention,
        caps.quantized_kv,
        caps.compile_through,
        caps.async_eval,
    ) == (True,) * 5
    assert caps.stream_model == StreamModel.MULTI_STREAM
    # Feature-absent build.
    mx_bare = FakeMx(
        fast_kernels=False,
        flash_attention=False,
        quantized_kv=False,
        compile_through=False,
        async_eval=False,
    )
    caps_bare = MetalBackend(mx=mx_bare).capabilities()
    assert (
        caps_bare.fast_kernels,
        caps_bare.flash_attention,
        caps_bare.quantized_kv,
        caps_bare.compile_through,
        caps_bare.async_eval,
    ) == (False,) * 5
    # fast namespace present but no SDPA → fast_kernels True, flash False
    # (the optimizations.py capability-report distinction).
    mx_no_sdpa = FakeMx(flash_attention=False)
    caps_no_sdpa = MetalBackend(mx=mx_no_sdpa).capabilities()
    assert caps_no_sdpa.fast_kernels is True
    assert caps_no_sdpa.flash_attention is False


# ---------------------------------------------------------------------------
# Stream contract — declared single-stream detection (no string-probing
# by callers)
# ---------------------------------------------------------------------------


def test_single_stream_detection_m5_shaped():
    """M5: the probe raises the specific RuntimeError → False (declared
    contract answers the question the #404 shim used to string-probe)."""
    mx = FakeMx(single_stream=True)
    b = MetalBackend(mx=mx)
    assert b.supports_secondary_streams() is False
    assert b.capabilities().stream_model == StreamModel.SINGLE_STREAM


def test_multi_stream_detection_m1_m4_shaped():
    mx = FakeMx()
    assert MetalBackend(mx=mx).supports_secondary_streams() is True
    assert MetalBackend(mx=mx).capabilities().stream_model == StreamModel.MULTI_STREAM


def test_unrelated_runtime_error_reraised():
    """An unrelated RuntimeError must surface, not be silently degraded
    (same contract as the #404 shim's probe)."""
    mx = FakeMx(stream_error=RuntimeError("Some completely unrelated MLX error"))
    with pytest.raises(RuntimeError, match="completely unrelated"):
        MetalBackend(mx=mx).supports_secondary_streams()


def test_pre_408_mlx_without_tls_stream_symbol():
    mx = FakeMx(new_thread_local_stream_api=False)
    assert not hasattr(mx, "new_thread_local_stream")
    assert MetalBackend(mx=mx).supports_secondary_streams() is False


def test_stream_probe_is_one_time_per_device():
    """The probe is cached per device — the one-time contract."""
    mx = FakeMx(single_stream=True)
    b = MetalBackend(mx=mx)
    calls: list = []
    original = mx.new_thread_local_stream

    def counting(device):
        calls.append(device)
        return original(device)

    mx.new_thread_local_stream = counting
    assert b.supports_secondary_streams() is False
    assert b.supports_secondary_streams() is False
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_get_backend_prefers_registered():
    reset_registry()
    fake = FakeMx()
    b = MetalBackend(mx=fake)
    register_backend(b)
    try:
        assert get_backend() is b
    finally:
        reset_registry()


def test_get_backend_unavailable_registered_falls_through():
    """A registered backend that reports unavailable must not shadow
    the built-in resolution."""

    class _Dead:
        kind = BackendKind.ROCM
        enforcement_mode = EnforcementMode.HARD

        def is_available(self):
            return False

    reset_registry()
    register_backend(_Dead())
    try:
        b = get_backend()
        if _real_mx_or_none() is not None:
            assert isinstance(b, MetalBackend)
        else:
            assert b is None
    finally:
        reset_registry()


def test_get_backend_resolution():
    reset_registry()
    try:
        b = get_backend()
        if _real_mx_or_none() is not None:
            assert isinstance(b, MetalBackend)
        else:
            assert b is None
    finally:
        reset_registry()


# ---------------------------------------------------------------------------
# Gold standard: parity against the real mlx.core (Apple-Silicon CI)
# ---------------------------------------------------------------------------


def test_parity_against_real_mlx():
    """Runs only where ``mlx`` is installed: the seam must return the
    exact values the call sites get from direct ``mlx`` calls."""
    mx = _real_mx_or_none()
    if mx is None:
        pytest.skip("mlx not installed (Linux CI)")
    b = MetalBackend(mx=mx)
    assert b.is_available() == mx.metal.is_available()
    if not mx.metal.is_available():
        pytest.skip("no Metal device on this machine")
    assert b.device() == mx.default_device()
    assert b.device_name() == mx.device_info().get("device_name", "")
    assert b.total_memory_bytes() == mx.device_info().get("memory_size", 0)
    direct = mx.device_info().get(
        "max_recommended_working_set_size",
        mx.device_info().get("memory_size", 0),
    )
    assert b.recommended_working_set_bytes() == direct
    assert b.memory_stats().active_bytes == mx.get_active_memory()
    assert b.memory_stats().peak_bytes == mx.get_peak_memory()
    assert b.memory_stats().cache_bytes == mx.get_cache_memory()
    assert b.default_stream() == mx.default_stream(mx.default_device())
    caps = b.capabilities()
    assert caps.fast_kernels == hasattr(mx, "fast")
    assert caps.flash_attention == (
        hasattr(mx, "fast") and hasattr(mx.fast, "scaled_dot_product_attention")
    )
    assert caps.quantized_kv == (hasattr(mx, "quantize") and hasattr(mx, "dequantize"))
    assert caps.compile_through == hasattr(mx, "compile")
    assert caps.async_eval == hasattr(mx, "async_eval")
