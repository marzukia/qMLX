# SPDX-License-Identifier: Apache-2.0
"""
The device/backend seam — the single place qMLX learns "the device".

qMLX is a multi-backend MLX engine: Apple Metal (this build), and
``mlx-rocm`` / ``mlx-cuda`` (ported builds). Before this seam, ~15 files
each reached into ``mlx`` with a different mental model of the device
(``mx.metal.is_available()`` gates, raw ``device_info()`` key-digging,
``sysctl`` probes, a ``RuntimeError`` string probe for single-stream
hardware). On a ported build those probes are silently wrong: the
``mx.metal.*`` gates disable the memory admission system, and unguarded
top-level memory queries raise.

Every module in this package is importable without ``mlx`` installed
(metadata-only usage, Linux CI): ``mlx.core`` is imported lazily inside
backend methods, never at module level.

The spec
--------

``Backend`` (Protocol — structural, ported backends need not inherit) —
one instance per backend. Members:

* ``kind``: ``BackendKind`` — ``"metal"`` / ``"rocm"`` / ``"cuda"``.
* ``enforcement_mode``: ``EnforcementMode`` — how the backend allocator
  treats ``set_memory_limit()``:

  - ``HINT`` — the cap is a guideline; the allocator may grow past it
    under load (Metal today — D-METAL-CAP: the scheduler polices the
    cap at admission).
  - ``HARD`` — the allocator enforces the cap; over-allocation fails.
    On a HARD backend the scheduler-side admission police become
    redundant; the admission gate's character changes with the declared
    mode.

* ``is_available() -> bool`` — replaces the ``mx.metal.is_available()``
  gate. Must return ``False`` (never raise) when the backend's device
  is absent (e.g. a ported build with no ``mx.metal``).
* ``device()`` — the device object (Metal: ``mx.default_device()``).
* ``device_name() -> str`` — chip name (Metal:
  ``device_info()["device_name"]``); replaces the four
  ``sysctl machdep.cpu.brand_string`` call sites.
* ``total_memory_bytes() -> int`` — total device memory (Metal:
  ``device_info()["memory_size"]``); replaces the
  ``sysctl hw.memsize`` call sites.
* ``recommended_working_set_bytes() -> int`` — the *recommended working
  set*: the denominator of every memory cap. Metal:
  ``device_info()["max_recommended_working_set_size"]`` with
  ``device_info()["memory_size"]`` fallback — exactly the expression the
  boot path (``engine/batched.py``), the admission gate
  (``scheduler.py`` D-METAL-CAP), and the idle-loop governor
  (``engine_core.py``) compute today. ROCm (unified HMM) and CUDA
  (discrete HBM) define their own local analogue; ``0`` when the
  backend has no such notion.
* ``set_memory_limit(n)`` / ``set_cache_limit(n)`` — allocator cap
  setters (Metal: ``mx.set_memory_limit`` / ``mx.set_cache_limit``).
  Interpret the effect through ``enforcement_mode``.
* ``memory_stats() -> MemoryStats`` — ``active_bytes`` / ``peak_bytes``
  / ``cache_bytes`` (Metal: ``mx.get_active_memory`` /
  ``mx.get_peak_memory`` / ``mx.get_cache_memory``).
* ``capabilities() -> CapabilityFlags`` — see below.
* ``default_stream()`` — the stream the owner thread uses (Metal:
  ``mx.default_stream(mx.default_device())``).
* ``supports_secondary_streams() -> bool`` — the declared single-stream
  detection contract (see the stream contract below).

``CapabilityFlags`` (frozen dataclass) — the Part-6 table items:

* ``fast_kernels`` — ``mx.fast`` namespace present (``mx.fast.rms_norm``
  and friends; P0 for the live MTP cache path).
* ``flash_attention`` — ``mx.fast.scaled_dot_product_attention``
  present.
* ``quantized_kv`` — ``mx.quantize`` AND ``mx.dequantize`` present
  (Metal-only in upstream MLX; a ported backend missing either one
  makes QuantizedKVCache checkpoints round-trip as *garbage* — the
  loader cannot tell, so this flag gates the load-time conformance
  check, not just an import check).
* ``compile_through`` — ``mx.compile`` present (the seeded-sampler
  reproducibility boundary; the fast path must degrade cleanly when
  absent).
* ``async_eval`` — ``mx.async_eval`` present (the multiplexer's eval
  pump; semantics differ per backend and the scheduler's backoff logic
  depends on them — see the stream contract).
* ``stream_model`` — ``StreamModel.SINGLE_STREAM`` / ``MULTI_STREAM``
  (the thread/stream semantics class).

``MemoryStats`` (frozen dataclass) — ``active_bytes``, ``peak_bytes``,
``cache_bytes`` (bytes).

Registry
-------

``get_backend() -> Backend | None`` — the active backend for this
process, or ``None`` when no MLX backend is available (``mlx`` not
installed, or no GPU backend detected). Resolution: explicitly
registered backends first (ported builds self-register at import time
via ``register_backend``), then the built-in Metal backend. Call sites
treat ``None`` as "no GPU backend" — the same semantics as today's
``if mx.metal.is_available():`` gate.

Stream contract — the "one owner thread" invariant
-------------------------------------------------

All MLX work on a device is submitted by a single owner thread (qMLX:
the ``mlx-step`` thread, ``engine_core._init_mlx_step_thread``). The
owner thread uses ``default_stream()``; no other thread touches MLX
objects or calls ``mx.set_default_stream``. This is a *design*
invariant — bandwidth-bound ⇒ interleave, don't overlap — not a
workaround for an mlx-lm 0.31.3 pathology.

``supports_secondary_streams()`` is the declared single-stream
detection contract: it answers "does this device expose a usable
secondary stream?" so callers stop parsing ``RuntimeError`` strings.
Today ``_mlx_compat.py`` probes the specific
``RuntimeError: There is no Stream(gpu, N) in current thread.`` string
to detect single-stream hardware (M5, #404). After this seam lands,
callers query the backend; on Metal the answer is still a one-time
hardware probe (cached per device — the same probe the #404 shim uses,
so behavior is unchanged), because the stream model is a runtime
property of Apple silicon. Ported backends (ROCm/CUDA) declare the
answer directly.

Landing scope (item 2 of 9)
---------------------------

Only the Metal backend is implemented here (``metal.py``); it returns
exactly the values the current call sites get (parity is pinned by
``tests/test_device_backend.py`` — the conformance seed for the item-7
backend conformance suite). The ~15 live call sites are NOT migrated in
this job (item 3); the Metal backend is the drop-in source of truth
for that migration. Ported backends land as
``vllm_mlx.device.rocm`` / ``vllm_mlx.device.cuda`` implementing this
Protocol and self-registering via ``register_backend``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class BackendKind(str, Enum):
    """Backend identity. Ported builds register with their own kind."""

    METAL = "metal"
    ROCM = "rocm"
    CUDA = "cuda"


class EnforcementMode(str, Enum):
    """How the backend allocator treats ``set_memory_limit``.

    ``HINT``: the cap is a guideline; the allocator may grow past it
    under load (Metal today — D-METAL-CAP: the scheduler polices the
    cap at admission). ``HARD``: the allocator enforces the cap;
    over-allocation fails (scheduler-side admission police become
    redundant).
    """

    HINT = "hint"
    HARD = "hard"


class StreamModel(str, Enum):
    """The thread/stream semantics class (Part 6, table I)."""

    SINGLE_STREAM = "single_stream"
    MULTI_STREAM = "multi_stream"


@dataclass(frozen=True)
class MemoryStats:
    """Allocator memory snapshot in bytes (Part 5, table B)."""

    active_bytes: int
    peak_bytes: int
    cache_bytes: int


@dataclass(frozen=True)
class CapabilityFlags:
    """Per-backend capability report (Part 6 table items)."""

    fast_kernels: bool
    flash_attention: bool
    quantized_kv: bool
    compile_through: bool
    async_eval: bool
    stream_model: StreamModel


@runtime_checkable
class Backend(Protocol):
    """The device/backend seam contract.

    Structural (``@runtime_checkable`` Protocol): a ported backend
    (``mlx-rocm`` / ``mlx-cuda``) satisfies it by implementing the
    members — no inheritance required. See the module docstring for the
    per-member spec.
    """

    kind: BackendKind
    enforcement_mode: EnforcementMode

    def is_available(self) -> bool:
        """Device present and usable. Never raises."""
        ...

    def device(self) -> Any:
        """The device object (Metal: ``mx.default_device()``)."""
        ...

    def device_name(self) -> str:
        """Chip name (replaces ``sysctl machdep.cpu.brand_string``)."""
        ...

    def total_memory_bytes(self) -> int:
        """Total device memory (replaces ``sysctl hw.memsize``)."""
        ...

    def recommended_working_set_bytes(self) -> int:
        """Denominator of every memory cap; 0 when the backend has no
        such notion."""
        ...

    def set_memory_limit(self, limit_bytes: int) -> None:
        """Allocator cap setter (interpret via ``enforcement_mode``)."""
        ...

    def set_cache_limit(self, limit_bytes: int) -> None:
        """Allocation-cache cap setter."""
        ...

    def memory_stats(self) -> MemoryStats:
        """Active / peak / cache bytes."""
        ...

    def capabilities(self) -> CapabilityFlags:
        """Part-6 table capability report."""
        ...

    def default_stream(self) -> Any:
        """The stream the owner thread uses."""
        ...

    def supports_secondary_streams(self) -> bool:
        """Declared single-stream detection contract (no string-probing
        by callers)."""
        ...


_UNSET = object()

_registered: list[Backend] = []
_builtin_cache: Backend | None | object = _UNSET


def _try_builtin_metal() -> Backend | None:
    """Built-in Metal backend, or ``None`` when mlx is missing or Metal
    is unavailable (ported builds register their own backend instead)."""
    try:
        import mlx.core as mx
    except Exception:
        return None
    try:
        if mx.metal.is_available():
            from vllm_mlx.device.metal import MetalBackend

            return MetalBackend()
    except Exception:
        return None
    return None


def get_backend() -> Backend | None:
    """The active backend for this process, or ``None`` when no MLX
    backend is available.

    Resolution: registered backends first (ported builds self-register
    at import time), then the built-in Metal backend. Call sites treat
    ``None`` as "no GPU backend" — the same semantics as today's
    ``if mx.metal.is_available():`` gate.
    """
    for backend in _registered:
        if backend.is_available():
            return backend
    global _builtin_cache
    if _builtin_cache is _UNSET:
        _builtin_cache = _try_builtin_metal()
    return None if _builtin_cache is _UNSET else _builtin_cache


def register_backend(backend: Backend) -> None:
    """Ported backends (``mlx-rocm`` / ``mlx-cuda``) self-register here
    at import time. Registered backends take priority over the
    built-in Metal backend."""
    _registered.append(backend)


def reset_registry() -> None:
    """Clear registered backends and the built-in cache (test helper)."""
    global _builtin_cache
    _registered.clear()
    _builtin_cache = _UNSET
