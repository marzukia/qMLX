# SPDX-License-Identifier: Apache-2.0
"""
The device/backend seam (qMLX multi-backend repositioning, item 2 of 9).

The single place qMLX learns what "the device" is: per-backend
identity, memory cap + allocator enforcement mode, memory stats,
capability flags, and the stream contract ("one owner thread"
invariant + declared single-stream detection). See
``vllm_mlx.device.backend`` for the full spec — item 3 (call-site
migration) and the ported backends (``mlx-rocm`` / ``mlx-cuda``) code
against it.

Only the Metal backend lands in this job; the ~15 live call sites are
migrated in item 3.

No module in this package imports ``mlx.core`` at import time — the
package stays importable on systems without MLX (metadata-only access,
Linux CI).
"""

from vllm_mlx.device.backend import (
    Backend,
    BackendKind,
    CapabilityFlags,
    EnforcementMode,
    MemoryStats,
    StreamModel,
    get_backend,
    register_backend,
    reset_registry,
)
from vllm_mlx.device.conformance import (
    CheckResult,
    ConformanceReport,
    documented_gap_names,
    run_conformance,
)
from vllm_mlx.device.metal import MetalBackend

__all__ = [
    "Backend",
    "BackendKind",
    "CapabilityFlags",
    "CheckResult",
    "ConformanceReport",
    "EnforcementMode",
    "MemoryStats",
    "MetalBackend",
    "StreamModel",
    "documented_gap_names",
    "get_backend",
    "register_backend",
    "reset_registry",
    "run_conformance",
]
