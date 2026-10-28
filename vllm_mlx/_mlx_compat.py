# SPDX-License-Identifier: Apache-2.0
"""
MLX hardware-compatibility shims.

Currently handles one upstream issue:

**M5 single-stream GPU (#404)**: `mlx_lm/generate.py` does

    generation_stream = mx.new_thread_local_stream(mx.default_device())

at module-import time. On M1–M4, this returns a usable thread-local stream.
On M5, the call appears to succeed (returning a Stream handle), but later
``with mx.stream(generation_stream):`` raises

    RuntimeError: There is no Stream(gpu, 1) in current thread.

because the M5 GPU only exposes a single stream slot. Every pure-attention
model crashes at first prompt evaluation. Hybrid models (Qwen3.5/3.6) work
because their custom path doesn't import ``mlx_lm.generate``.

Fix: monkey-patch ``mx.new_thread_local_stream`` with a probe-and-cache
wrapper. On the first call we attempt a trivial op inside ``mx.stream(s)``;
if it raises, we cache that fact and return ``mx.default_stream(device)``
for all subsequent calls. Single-stream devices then run with the default
stream, losing parallel-issue throughput but staying functional. Hardware
that supports multiple streams gets the original behavior — the probe is
one-time per device.

Item 3 (device/backend seam): the single-stream *detection* is routed
through the backend's declared stream semantics —
``backend.supports_secondary_streams()`` — instead of this module
string-probing the ``RuntimeError`` itself. Ported backends (ROCm/
CUDA) declare their stream model directly; on Metal the declared
answer is the same one-time hardware probe (cached in the backend).
The monkeypatch itself stays: ``mlx_lm.generate`` captures
``mx.new_thread_local_stream`` at module-import time, so the symbol
must still be intercepted. The per-device probe below remains as the
fallback for non-default devices and for builds where no backend
resolves. A re-entrancy guard keeps the backend's own probe (which
calls the patched symbol) from recursing into the wrapper.

This patch must execute *before* any ``import mlx_lm.generate``, since
that module captures the returned stream at module level. The install
hook is called at the top of every consumer that imports
``mlx_lm.generate`` (currently ``vllm_mlx/scheduler.py``) — *not* from
``vllm_mlx/__init__.py``.
We deliberately keep ``import vllm_mlx`` free of any ``mlx.core`` import
so the package stays usable for metadata-only access on systems where
``mlx`` is installed but Metal is unavailable (``import mlx.core``
SIGABRTs there with an uncatchable NSException).

Upstream tracking: file mlx-lm bug + remove this shim when upstream lands
a device-capability check.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install() -> None:
    """Install M5-compat shim. Safe to call multiple times (idempotent).

    No-op when mlx.core can't be imported (non-Apple-Silicon CI). Logging
    is at debug level on success to keep startup quiet for the 99% of
    users on hardware where the original API works.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return  # Linux CI / no MLX

    if getattr(mx, "_qmlx_compat_installed", False):
        return

    # No-op on builds that predate ``mx.new_thread_local_stream`` (#408): the
    # M5 single-stream bug only manifests when ``mlx_lm.generate`` captures
    # this symbol at module import. Older mlx never had it, so neither
    # ``mlx_lm.generate`` nor the bug it triggers can be present here. We
    # intentionally do NOT set ``_qmlx_compat_installed`` here — if
    # the symbol later appears (importlib.reload, dynamic upgrade), the
    # next install() call should re-evaluate and apply the wrap.
    if not hasattr(mx, "new_thread_local_stream"):
        return

    # Item 3 (device/backend seam): stream semantics are declared by the
    # backend, not string-probed here. ``supports_secondary_streams()``
    # is the declared single-stream detection contract; on Metal it is
    # the same one-time hardware probe (cached in the backend), on
    # ported backends it is a direct declaration.
    from vllm_mlx.device import get_backend

    backend = get_backend()
    _default_device_key = None
    if backend is not None:
        try:
            _default_device_key = repr(backend.device())
        except Exception:
            _default_device_key = None

    original = mx.new_thread_local_stream

    # Tri-state cache per device:
    #   None  → not probed yet
    #   True  → original works on this device
    #   False → original is unusable, must fall back to default_stream
    _probe_cache: dict = {}
    # Re-entrancy guard: the backend's own probe calls
    # ``mx.new_thread_local_stream`` (this wrapper) internally; the
    # inner call must return the raw stream so the outer probe can
    # exercise it — without the guard the two probes recurse.
    _in_progress: set = set()

    def _probe(stream, device) -> bool:
        # Backend-declared answer for the default device (ported
        # backends declare stream semantics directly instead of
        # string-probing; on Metal the declared answer is the same
        # one-time hardware probe, cached in the backend).
        if _default_device_key is not None and repr(device) == _default_device_key:
            try:
                return bool(backend.supports_secondary_streams())
            except Exception:
                pass  # unexpected backend failure → local probe below
        try:
            with mx.stream(stream):
                # Force evaluation so the stream is actually exercised.
                _ = (mx.array([0.0]) + mx.array([1.0])).item()
            return True
        except RuntimeError as e:
            msg = str(e)
            # Be permissive about the exact wording: upstream may change it.
            if "Stream" in msg and ("no Stream" in msg or "not exist" in msg):
                logger.warning(
                    "MLX device %s rejects secondary streams (%s) — "
                    "falling back to default_stream. "
                    "Throughput on parallel ops may be reduced. "
                    "This is the #404 M5 single-stream workaround.",
                    device,
                    msg,
                )
                return False
            raise

    def patched_new_thread_local_stream(device):
        key = repr(device)
        cached = _probe_cache.get(key)
        if cached is False:
            return mx.default_stream(device)
        if cached is True:
            return original(device)
        if key in _in_progress:
            # Re-entrant call (the backend's own probe is running
            # through this wrapper) — return the raw stream; the outer
            # probe decides.
            return original(device)
        _in_progress.add(key)
        try:
            # First call for this device — probe. Log at INFO so that anyone
            # filing a hardware-shaped bug report has the device family in
            # their startup output. Future M5+/Apple chip families will land
            # here first; greppable on "qmlx compat".
            stream = original(device)
            if _probe(stream, device):
                _probe_cache[key] = True
                logger.info(
                    "qmlx compat: device %s supports thread-local streams (no shim).",
                    device,
                )
                return stream
            _probe_cache[key] = False
            logger.info(
                "qmlx compat: device %s uses default_stream fallback (#404 M5 path).",
                device,
            )
            return mx.default_stream(device)
        finally:
            _in_progress.discard(key)

    mx.new_thread_local_stream = patched_new_thread_local_stream
    mx._qmlx_compat_installed = True
    logger.debug("MLX compat shim installed (#404 M5 single-stream guard).")
