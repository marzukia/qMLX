# SPDX-License-Identifier: Apache-2.0
"""
The Metal backend for the device/backend seam (item 2 of 9).

Returns EXACTLY the values the current call sites get from ``mlx``
directly — parity is pinned by ``tests/test_device_backend.py`` (the
conformance seed for the item-7 backend conformance suite):

================================  ======================================
seam member                     current call-site expression
================================  ======================================
``is_available()``                ``mx.metal.is_available()``
``device()``                      ``mx.default_device()``
``device_name()``                 ``mx.device_info().get("device_name", "")``
``total_memory_bytes()``          ``mx.device_info().get("memory_size", 0)``
``recommended_working_set_bytes()``
                                ``mx.device_info().get(
                                    "max_recommended_working_set_size",
                                    mx.device_info().get("memory_size",
                                    0))``
``set_memory_limit(n)``           ``mx.set_memory_limit(n)``
``set_wired_limit(n)``            ``mx.set_wired_limit(n)``
                                  (fallback ``mx.metal.set_wired_limit(n)``
                                  on older MLX — the deprecated name,
                                  same fallback MTPLX uses)
``set_cache_limit(n)``            ``mx.set_cache_limit(n)``
``memory_stats()``                ``mx.get_active_memory()`` /
                                ``mx.get_peak_memory()`` /
                                ``mx.get_cache_memory()``
``default_stream()``              ``mx.default_stream(mx.default_device())``
``capabilities().flash_attention``
                                ``hasattr(mx, "fast") and hasattr(
                                    mx.fast,
                                    "scaled_dot_product_attention")``
``capabilities().os_memory_eviction``
                                ``sys.platform == "darwin"`` (evaluated
                                at call time — UBC is a macOS
                                property of the Metal build, not of
                                the GPU)
================================  ======================================

``enforcement_mode`` is declared ``EnforcementMode.HINT``: Metal treats
``mx.set_memory_limit`` as a *guideline* and grows past it under load
while system RAM is available (D-METAL-CAP root cause — the scheduler
polices the cap at admission).

``mlx.core`` is imported lazily: the constructor accepts an ``mx``
module for injection (tests); otherwise the first use imports
``mlx.core``.
"""

from __future__ import annotations

import sys
from typing import Any

from vllm_mlx.device.backend import (
    BackendKind,
    CapabilityFlags,
    EnforcementMode,
    MemoryStats,
    StreamModel,
)


class MetalBackend:
    """The Metal implementation of the ``Backend`` seam contract.

    Drop-in source of truth for the item-3 call-site migration: every
    member maps 1:1 onto the ``mlx`` expression the call site uses
    today (see the module docstring table).
    """

    kind = BackendKind.METAL
    enforcement_mode = EnforcementMode.HINT

    def __init__(self, mx: Any | None = None) -> None:
        self._mx = mx
        self._stream_probe_cache: dict[str, bool] = {}

    def _mx_(self) -> Any:
        if self._mx is None:
            import mlx.core as mx

            self._mx = mx
        return self._mx

    def is_available(self) -> bool:
        """Parity: ``mx.metal.is_available()`` — ``False`` (never
        raises) when ``mx.metal`` is absent (ported build)."""
        try:
            return bool(self._mx_().metal.is_available())
        except Exception:
            return False

    def device(self) -> Any:
        """Parity: ``mx.default_device()``."""
        return self._mx_().default_device()

    def device_name(self) -> str:
        """Parity: ``mx.device_info().get("device_name", "")``."""
        return self._mx_().device_info().get("device_name", "")

    def total_memory_bytes(self) -> int:
        """Parity: ``mx.device_info().get("memory_size", 0)``."""
        return int(self._mx_().device_info().get("memory_size", 0))

    def recommended_working_set_bytes(self) -> int:
        """Parity: the exact two-level fallback expression used by
        ``engine/batched.py`` (boot limits), ``scheduler.py``
        (D-METAL-CAP admission gate) and ``engine_core.py`` (idle-loop
        governor) today."""
        info = self._mx_().device_info()
        return int(
            info.get(
                "max_recommended_working_set_size",
                info.get("memory_size", 0),
            )
        )

    def set_memory_limit(self, limit_bytes: int) -> None:
        """Parity: ``mx.set_memory_limit(n)`` (HINT — see class
        ``enforcement_mode``)."""
        self._mx_().set_memory_limit(limit_bytes)

    def set_wired_limit(self, limit_bytes: int) -> None:
        """Per-process GPU wired (resident) cap — the separate axis
        behind ``--wired-limit-gb`` (carbon parity journal D23.1).

        Unlike ``set_memory_limit`` (an allocation budget the HINT
        allocator treats as a guideline and overgrows — the 16GB
        OOM failure), the wired limit bounds the RESIDENT (wired)
        GPU pool: above it, the OS pages weight memory out to
        RAM/swap and it re-establishes lazily on the next submit
        (the partial-weight-residency mechanism MTPLX uses to run a
        ~10GB model on a 16GB Mac). MLX refuses a limit above the
        system's ``max_recommended_working_set_size`` (a refused
        limit wires nothing), so callers clamp first.

        Parity: ``mx.set_wired_limit(n)`` — prefers the top-level
        API; falls back to the deprecated ``mx.metal.set_wired_limit``
        on older MLX builds (same fallback MTPLX uses).

        Raises:
            AttributeError: neither API is present (ported build
                without a wired-limit concept).
        """
        mx = self._mx_()
        setter = getattr(mx, "set_wired_limit", None)
        if not callable(setter):
            metal = getattr(mx, "metal", None)
            metal_setter = getattr(metal, "set_wired_limit", None)
            if not callable(metal_setter):
                raise AttributeError("mlx wired-limit API unavailable")
            metal_setter(limit_bytes)
            return
        setter(limit_bytes)

    def set_cache_limit(self, limit_bytes: int) -> None:
        """Parity: ``mx.set_cache_limit(n)``."""
        self._mx_().set_cache_limit(limit_bytes)

    def memory_stats(self) -> MemoryStats:
        """Parity: ``mx.get_active_memory()`` / ``mx.get_peak_memory()``
        / ``mx.get_cache_memory()``."""
        mx = self._mx_()
        return MemoryStats(
            active_bytes=int(mx.get_active_memory()),
            peak_bytes=int(mx.get_peak_memory()),
            cache_bytes=int(mx.get_cache_memory()),
        )

    def capabilities(self) -> CapabilityFlags:
        """Parity: the ``hasattr`` feature-detection the capability
        report uses today (``optimizations.py``), plus the declared
        stream model."""
        mx = self._mx_()
        return CapabilityFlags(
            fast_kernels=hasattr(mx, "fast"),
            flash_attention=hasattr(mx, "fast")
            and hasattr(mx.fast, "scaled_dot_product_attention"),
            quantized_kv=hasattr(mx, "quantize") and hasattr(mx, "dequantize"),
            compile_through=hasattr(mx, "compile"),
            async_eval=hasattr(mx, "async_eval"),
            stream_model=self._stream_model(),
            # Evaluated lazily (not at construction) so a test that
            # monkeypatches ``sys.platform`` sees the patched value;
            # UBC is a macOS property of the Metal build, not of the
            # GPU. Ported backends (ROCm/CUDA on Linux) declare their
            # own OS-side eviction story (page-cache / madvise).
            os_memory_eviction=sys.platform == "darwin",
        )

    def default_stream(self) -> Any:
        """Parity: ``mx.default_stream(mx.default_device())`` — the
        stream the owner thread uses (one-owner-thread invariant)."""
        mx = self._mx_()
        return mx.default_stream(mx.default_device())

    def supports_secondary_streams(self) -> bool:
        """Declared single-stream detection contract (see the
        ``vllm_mlx.device.backend`` module docstring).

        On Metal the stream model is a runtime property of the chip
        (M5 exposes a single stream slot; M1–M4 expose usable
        thread-local streams), so the declared answer is computed by a
        one-time probe per device — the same probe the #404 M5 shim
        uses, so behavior is unchanged — and cached. An unrelated
        ``RuntimeError`` is re-raised (unexpected failures surface, not
        silently degraded).
        """
        mx = self._mx_()
        device = mx.default_device()
        key = repr(device)
        cached = self._stream_probe_cache.get(key)
        if cached is not None:
            return cached
        result = self._probe_secondary_stream(mx, device)
        self._stream_probe_cache[key] = result
        return result

    def _probe_secondary_stream(self, mx: Any, device: Any) -> bool:
        if not hasattr(mx, "new_thread_local_stream"):
            # Pre-#408 mlx: no thread-local stream API at all.
            return False
        stream = mx.new_thread_local_stream(device)
        try:
            with mx.stream(stream):
                # Force evaluation so the stream is actually exercised.
                _ = (mx.array([0.0]) + mx.array([1.0])).item()
            return True
        except RuntimeError as e:
            msg = str(e)
            # Be permissive about the exact wording: upstream may change
            # it (same matching as the #404 shim).
            if "Stream" in msg and ("no Stream" in msg or "not exist" in msg):
                return False
            raise

    def _stream_model(self) -> StreamModel:
        if self.supports_secondary_streams():
            return StreamModel.MULTI_STREAM
        return StreamModel.SINGLE_STREAM
