# SPDX-License-Identifier: Apache-2.0
"""
Backend conformance suite (qMLX repositioning, item 7 of 9).

The mlx-rocm / mlx-cuda acceptance gate. The suite runs against whatever
``mx.default_device()`` is — CPU on Linux CI, Metal on the Mac, the
ported build on the ROCm guest — and reports, per check:

* ``pass``               — the check ran and the backend behaves as declared
* ``fail``               — a restore bug (the API is present but the
                           behaviour contradicts the declaration, or a
                           round-trip is broken)
* ``capability-missing`` — a backend gap (the API is absent; the feature
                           degrades by design)

A backend is **ACCEPTED** when there are no ``fail`` lines and every
``capability-missing`` line is a documented gap (the built-in registry
below, or the port's gap register via ``--gaps FILE``).

Checks (the six repositioning items):

1. memory-cap enforcement — the allocator's observed behaviour
   (hint vs hard) must match the backend's declared ``enforcement_mode``
2. quantised KV round-trip — ``mx.quantize`` / ``mx.dequantize`` must be
   deterministic, shape-preserving, finite, and within the quantisation
   step bound; ``QuantizedKVCache`` checkpoints must round-trip
   byte-stably. A missing quantizer is LOUD (capability-missing), never
   silent: a quantised checkpoint would otherwise round-trip as garbage
   and the loader cannot tell.
3. prompt-cache format — ``save_prompt_cache`` / ``load_prompt_cache``
   must preserve class identity and values; the on-disk format must store
   the cache class name (the restore identity guard keys on it); an
   unknown class name must fail closed (the loader refuses, never
   returns the wrong class); the model identity guard must reject on any
   doubt (fail-closed, pure logic).
4. stream/thread ownership — the observed stream semantics (multi vs
   single) must match ``supports_secondary_streams()`` and
   ``capabilities().stream_model``; the default stream must be reachable
   from a non-owner thread; ``set_default_stream`` must work in the
   owner thread.
5. ``mx.compile`` / ``mx.async_eval`` — compile-through must be
   deterministic and match eager; ``async_eval`` must evaluate the
   pending chain in order. Both must match the capability flags.
6. ``device_info`` — per-key presence, type, and seam parity (the
   backend's accessors must agree with the raw ``device_info()`` keys
   they are specified to read).

Plus a seam-contract check: the resolved ``Backend`` must structurally
satisfy the protocol (``is_available()`` never raises; kind /
enforcement mode / capability flags are valid and consistent with the
raw API).

Usage:

    python -m vllm_mlx.device.conformance [--json PATH] [--gaps FILE]

``--gaps FILE`` is the port's gap register: a JSON list of check names
whose ``capability-missing`` status is a documented, expected gap for
this port (e.g. ``["stream_ownership"]`` for a single-stream ROCm build).

Exit code: 0 = accepted, 1 = not accepted, 2 = error.

The suite never mutates host state it does not own: the compile probe
runs in a ``TMPDIR``-isolated cache directory (the MLX compile cache
lives under ``$TMPDIR/mlx``); memory limits are reset after the probe.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any

from vllm_mlx.device.backend import (
    Backend,
    BackendKind,
    EnforcementMode,
    StreamModel,
    get_backend,
)

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_CAPABILITY_MISSING = "capability-missing"

_UNSET = object()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One conformance check line."""

    name: str
    status: str
    detail: str


@dataclass
class ConformanceReport:
    """The full conformance report.

    ``accepted`` is False when any check failed, when any
    capability-missing check is not in ``documented_gaps``, or when the
    probes did not run (metadata-only mode — not an acceptance verdict).
    """

    checks: list[CheckResult]
    environment: dict[str, Any] = field(default_factory=dict)
    documented_gaps: frozenset[str] = frozenset()

    @property
    def fails(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == STATUS_FAIL]

    @property
    def capability_missing(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == STATUS_CAPABILITY_MISSING]

    @property
    def undocumented_gaps(self) -> list[CheckResult]:
        return [
            c for c in self.capability_missing if c.name not in self.documented_gaps
        ]

    @property
    def accepted(self) -> bool:
        if not self.environment.get("probes_ran", False):
            return False
        return not self.fails and not self.undocumented_gaps

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable form (``--json``)."""
        return {
            "environment": dict(self.environment),
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
            "documented_gaps": sorted(self.documented_gaps),
            "fails": len(self.fails),
            "capability_missing": len(self.capability_missing),
            "undocumented_gaps": len(self.undocumented_gaps),
            "accepted": self.accepted,
        }

    def format_table(self) -> str:
        """Human-readable table (stdout)."""
        lines = [
            "qMLX backend conformance report",
            f"environment: {self._environment_summary()}",
            "",
        ]
        name_w = max(len("NAME"), max((len(c.name) for c in self.checks), default=0))
        lines.append(f"{'NAME':<{name_w}}  {'STATUS':<20}  DETAIL")
        for c in self.checks:
            lines.append(f"{c.name:<{name_w}}  {c.status:<20}  {c.detail}")
        lines.append("")
        if self.accepted:
            verdict = "ACCEPTED"
        elif not self.environment.get("probes_ran", False):
            verdict = (
                "NOT ACCEPTED (metadata-only: the device probes were "
                "skipped; not an acceptance verdict)"
            )
        else:
            verdict = "NOT ACCEPTED"
        lines.append(
            f"{verdict}: {len(self.fails)} fail(s), "
            f"{len(self.capability_missing)} capability-missing "
            f"({len(self.undocumented_gaps)} undocumented)"
        )
        return "\n".join(lines)

    def _environment_summary(self) -> str:
        e = self.environment
        parts = [f"python {e.get('python', '?')}"]
        if e.get("mlx") is not None:
            parts.append(f"mlx {e['mlx']}")
        if e.get("mlx_lm") is not None:
            parts.append(f"mlx_lm {e['mlx_lm']}")
        if e.get("device") is not None:
            parts.append(f"device {e['device']} ({e.get('device_class', '?')})")
        parts.append(f"backend {e.get('backend') or 'none'}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Documented gaps
# ---------------------------------------------------------------------------

# Every check name the suite can emit (the no-mlx environment documents
# all of them: every probe is skipped).
_ALL_CHECK_NAMES: tuple[str, ...] = (
    "environment.mlx",
    "environment.mlx_lm",
    "seam_contract",
    "memory_cap_enforcement",
    "quantized_kv_roundtrip",
    "prompt_cache_format",
    "stream_ownership",
    "compile_through",
    "async_eval",
    "device_info.device_name",
    "device_info.memory_size",
    "device_info.max_recommended_working_set_size",
)

# Check names whose capability-missing status is a documented, expected
# gap per device class. The CPU profile is not a port target: the
# metal-specific absences (no device allocator, no stream concept, no
# memory_size / max_recommended_working_set_size keys) are expected and
# documented — not fails. The port (ROCm / CUDA) documents its own gaps
# via the gap register (--gaps FILE).
_DOCUMENTED_GAPS: dict[str, set[str]] = {
    "cpu": {
        "memory_cap_enforcement",
        "stream_ownership",
        "device_info.memory_size",
        "device_info.max_recommended_working_set_size",
    },
    "gpu": set(),
    "unknown": set(),
}


def documented_gap_names(
    device_class: str | None,
    gap_register: list[str] | None = None,
) -> set[str]:
    """Check names whose capability-missing status is a documented gap.

    ``device_class=None`` (mlx not importable) documents every check:
    the whole suite is skipped in metadata-only mode.
    """
    if device_class is None:
        return set(_ALL_CHECK_NAMES)
    gaps = set(_DOCUMENTED_GAPS.get(device_class, set()))
    if gap_register:
        gaps.update(gap_register)
    return gaps


# ---------------------------------------------------------------------------
# Probe context
# ---------------------------------------------------------------------------


@dataclass
class _ProbeContext:
    mx: Any | None
    mlx_lm: bool
    backend: Any | None
    device_class: str | None
    environment: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _device_class(mx: Any) -> str:
    try:
        r = repr(mx.default_device())
    except Exception:
        return "unknown"
    return "cpu" if "cpu" in r.lower() else "gpu"


def _read_safetensors_metadata(path: str) -> dict[str, str]:
    """Read the ``__metadata__`` dict from a safetensors file.

    No mlx dependency: 8-byte little-endian header length, then a JSON
    header (the same layout ``memory_cache._safetensors_cache_classes``
    uses). Returns ``{}`` when the file has no metadata.
    """
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    meta = header.get("__metadata__")
    return dict(meta) if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _add(results: list[CheckResult], name: str, status: str, detail: str) -> None:
    results.append(CheckResult(name, status, detail))


def _check_environment(ctx: _ProbeContext, results: list[CheckResult]) -> None:
    if ctx.mx is None:
        _add(
            results,
            "environment.mlx",
            STATUS_CAPABILITY_MISSING,
            "mlx.core is not importable on this host; the device probes "
            "are skipped (metadata-only mode). Install the platform MLX "
            "build (pip mlx / mlx-cpu / the ported build) to run the "
            "suite.",
        )
        return
    _add(
        results,
        "environment.mlx",
        STATUS_PASS,
        f"mlx {ctx.environment.get('mlx', '?')} importable; default "
        f"device {ctx.environment.get('device')}",
    )
    if not ctx.mlx_lm:
        _add(
            results,
            "environment.mlx_lm",
            STATUS_CAPABILITY_MISSING,
            "mlx_lm is not importable; the prompt-cache format probes "
            "run against the raw mx API only (the save_prompt_cache / "
            "load_prompt_cache round-trip is not probed)",
        )
    else:
        _add(
            results,
            "environment.mlx_lm",
            STATUS_PASS,
            f"mlx_lm {ctx.environment.get('mlx_lm', '?')} importable",
        )


def _check_seam_contract(ctx: _ProbeContext, results: list[CheckResult]) -> None:
    if ctx.backend is None:
        _add(
            results,
            "seam_contract",
            STATUS_PASS,
            "no backend resolved on this device (expected when no GPU "
            "backend is present); the raw API probes run directly "
            "against mlx",
        )
        return
    mx = ctx.mx
    assert mx is not None  # run_conformance only calls probes with mlx present
    problems: list[str] = []
    if not isinstance(ctx.backend, Backend):
        missing = [
            m
            for m in (
                "kind",
                "enforcement_mode",
                "is_available",
                "capabilities",
                "supports_secondary_streams",
                "device",
                "device_name",
                "total_memory_bytes",
                "recommended_working_set_bytes",
                "set_memory_limit",
                "set_cache_limit",
                "memory_stats",
                "default_stream",
            )
            if not hasattr(ctx.backend, m)
        ]
        problems.append(
            "does not structurally satisfy the Backend protocol"
            + (f" (missing: {', '.join(missing)})" if missing else "")
        )
    try:
        avail = ctx.backend.is_available()
        if not isinstance(avail, bool):
            problems.append(
                f"is_available() returned {type(avail).__name__}; the "
                "contract is a bool that never raises"
            )
    except Exception as e:
        problems.append(
            f"is_available() raised {type(e).__name__} ({e}); the "
            "contract is never-raise"
        )
    kind = getattr(ctx.backend, "kind", None)
    if kind is not None and not (
        isinstance(kind, BackendKind) or kind in (b.value for b in BackendKind)
    ):
        problems.append(f"kind {kind!r} is not a BackendKind value")
    mode = getattr(ctx.backend, "enforcement_mode", None)
    if mode is not None and not (
        isinstance(mode, EnforcementMode) or mode in (m.value for m in EnforcementMode)
    ):
        problems.append(f"enforcement_mode {mode!r} is not an EnforcementMode value")
    try:
        stats = ctx.backend.memory_stats()
        for f in ("active_bytes", "peak_bytes", "cache_bytes"):
            v = getattr(stats, f, None)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                problems.append(
                    f"memory_stats().{f} must be a non-negative int (got {v!r})"
                )
    except Exception as e:
        problems.append(f"memory_stats() raised {type(e).__name__} ({e})")
    try:
        caps = ctx.backend.capabilities()
    except Exception as e:
        problems.append(f"capabilities() raised {type(e).__name__} ({e})")
    else:
        fa = hasattr(mx, "fast") and hasattr(mx.fast, "scaled_dot_product_attention")
        qk = hasattr(mx, "quantize") and hasattr(mx, "dequantize")
        for flag, raw, label in (
            (caps.fast_kernels, hasattr(mx, "fast"), "fast_kernels"),
            (caps.flash_attention, fa, "flash_attention"),
            (caps.quantized_kv, qk, "quantized_kv"),
            (caps.compile_through, hasattr(mx, "compile"), "compile_through"),
            (caps.async_eval, hasattr(mx, "async_eval"), "async_eval"),
        ):
            if bool(flag) != bool(raw):
                problems.append(
                    f"capabilities().{label}={flag} disagrees with the raw API ({raw})"
                )
    if problems:
        _add(
            results,
            "seam_contract",
            STATUS_FAIL,
            "backend non-conformant: " + "; ".join(problems),
        )
    else:
        _add(
            results,
            "seam_contract",
            STATUS_PASS,
            f"backend {getattr(kind, 'value', kind)} structurally "
            f"satisfies the Backend protocol; enforcement_mode="
            f"{getattr(mode, 'value', mode)}; capabilities: "
            f"fast_kernels={caps.fast_kernels}, "
            f"flash_attention={caps.flash_attention}, "
            f"quantized_kv={caps.quantized_kv}, "
            f"compile_through={caps.compile_through}, "
            f"async_eval={caps.async_eval}, "
            f"stream_model={getattr(caps.stream_model, 'value', caps.stream_model)}",
        )


def _check_memory_cap_enforcement(
    ctx: _ProbeContext, results: list[CheckResult]
) -> None:
    mx = ctx.mx
    assert mx is not None  # run_conformance only calls probes with mlx present
    if not hasattr(mx, "set_memory_limit"):
        _add(
            results,
            "memory_cap_enforcement",
            STATUS_CAPABILITY_MISSING,
            "backend gap: mx.set_memory_limit is absent; the allocator "
            "enforcement mode cannot be probed",
        )
        return
    cap = 4 * 2**20
    try:
        mx.set_memory_limit(cap)
    except Exception as e:
        _add(
            results,
            "memory_cap_enforcement",
            STATUS_CAPABILITY_MISSING,
            f"mx.set_memory_limit exists but raised on this device "
            f"({type(e).__name__}: {e}); the cap is not supported here",
        )
        return
    observed = "hint"
    try:
        big = mx.ones((8192, 1024), dtype=mx.float32)  # 32 MiB
        mx.eval(big)
        del big
    except Exception:
        observed = "hard"
    try:
        mx.set_memory_limit(0)
    except Exception:
        pass
    declared = ctx.backend.enforcement_mode.value if ctx.backend is not None else None
    if declared is not None and declared != observed:
        _add(
            results,
            "memory_cap_enforcement",
            STATUS_FAIL,
            f"the backend declares enforcement_mode={declared} but the "
            f"allocator behaved {observed} (set_memory_limit(4 MiB); a "
            f"32 MiB allocation {'grew past the cap' if observed == 'hint' else 'failed past the cap'}); "
            "the scheduler admission semantics must match the declared "
            "mode",
        )
        return
    detail = (
        f"observed allocator behaviour: {observed} (set_memory_limit(4 "
        f"MiB); a 32 MiB allocation "
        f"{'succeeded — the cap is a guideline' if observed == 'hint' else 'failed — the cap is enforced'})"
    )
    if declared is not None:
        detail += f"; declared enforcement_mode={declared} (consistent)"
    else:
        detail += (
            "; no backend resolved on this device; the declared mode is "
            "not cross-checked"
        )
    _add(results, "memory_cap_enforcement", STATUS_PASS, detail)


def _check_quantized_kv_roundtrip(
    ctx: _ProbeContext, results: list[CheckResult]
) -> None:
    mx = ctx.mx
    assert mx is not None  # run_conformance only calls probes with mlx present
    has_q = hasattr(mx, "quantize")
    has_dq = hasattr(mx, "dequantize")
    if not (has_q and has_dq):
        missing = [
            n
            for n, present in (("mx.quantize", has_q), ("mx.dequantize", has_dq))
            if not present
        ]
        _add(
            results,
            "quantized_kv_roundtrip",
            STATUS_CAPABILITY_MISSING,
            "backend gap: "
            + " and ".join(missing)
            + " are absent — LOUD: a QuantizedKVCache checkpoint would "
            "round-trip as garbage and the loader cannot tell; the "
            "load-time conformance check must fail closed",
        )
        return
    group_size = 32
    bits = 4
    problems: list[str] = []
    try:
        x = (mx.arange(64).reshape(2, 32).astype(mx.float32) * 0.25) - 4.0
        mx.eval(x)
        q1 = mx.quantize(x, group_size=group_size, bits=bits)
        q2 = mx.quantize(x, group_size=group_size, bits=bits)
        for i, (a, b) in enumerate(zip(q1, q2)):
            if list(a.flatten().tolist()) != list(b.flatten().tolist()):
                problems.append(
                    "mx.quantize is not deterministic (part "
                    f"{i} differs across identical calls)"
                )
        dq = mx.dequantize(*q1, group_size=group_size, bits=bits)
        mx.eval(dq)
        if tuple(dq.shape) != tuple(x.shape):
            problems.append(
                f"mx.dequantize returned shape {tuple(dq.shape)} for "
                f"input shape {tuple(x.shape)}"
            )
        else:
            if not bool(mx.all(mx.isfinite(dq))):
                problems.append("mx.dequantize returned non-finite values")
            x_list = x.tolist()
            bound = 0.0
            for row in x_list:
                for i in range(0, len(row), group_size):
                    g = row[i : i + group_size]
                    bound = max(bound, (max(g) - min(g)) / (2**bits - 1))
            dev = (dq - x).abs().max()
            mx.eval(dev)
            if float(dev) > bound + 1e-9:
                problems.append(
                    f"dequantized values deviate {float(dev):.6f} from "
                    f"the input, past the quantisation step bound "
                    f"{bound:.6f}; the round-trip is not faithful"
                )
            dq2 = mx.dequantize(*q2, group_size=group_size, bits=bits)
            mx.eval(dq2)
            dev2 = (dq - dq2).abs().max()
            mx.eval(dev2)
            if float(dev2) > 1e-6:
                problems.append(
                    "mx.dequantize is not deterministic across "
                    f"identical packed forms (max deviation "
                    f"{float(dev2):.6f})"
                )
    except Exception as e:
        problems.append(f"the quantize/dequantize probe raised {type(e).__name__}: {e}")
    # QuantizedKVCache checkpoint sub-probe (needs mlx_lm)
    if ctx.mlx_lm:
        try:
            from mlx_lm.models.cache import QuantizedKVCache

            with tempfile.TemporaryDirectory(prefix="qmlx_conformance_") as d:
                p = os.path.join(d, "qkv.safetensors")
                qc = QuantizedKVCache()
                qc.group_size = group_size
                qc.bits = bits
                qkeys = mx.quantize(
                    mx.ones((2, 4, 32)), group_size=group_size, bits=bits
                )
                qc.keys = qkeys
                qc.values = mx.quantize(
                    mx.ones((2, 4, 32)), group_size=group_size, bits=bits
                )
                qc.offset = 8
                from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

                save_prompt_cache(p, [qc])
                loaded, _ = load_prompt_cache(p, return_metadata=True)
            if type(loaded[0]).__name__ != "QuantizedKVCache":
                problems.append(
                    "QuantizedKVCache class identity did not "
                    f"round-trip (loaded {type(loaded[0]).__name__})"
                )
            else:
                d_orig = mx.dequantize(*qkeys, group_size=group_size, bits=bits)
                d_loaded = mx.dequantize(
                    *loaded[0].keys, group_size=group_size, bits=bits
                )
                mx.eval(d_orig, d_loaded)
                dev = (d_orig - d_loaded).abs().max()
                mx.eval(dev)
                if float(dev) > 1e-6:
                    problems.append(
                        "the QuantizedKVCache checkpoint round-trip is "
                        f"not byte-stable (max deviation {float(dev):.6f}); "
                        "a restored cache would differ from the original"
                    )
        except Exception as e:
            problems.append(
                f"the QuantizedKVCache checkpoint probe raised {type(e).__name__}: {e}"
            )
    if problems:
        _add(
            results,
            "quantized_kv_roundtrip",
            STATUS_FAIL,
            "restore bug: " + "; ".join(problems),
        )
    else:
        _add(
            results,
            "quantized_kv_roundtrip",
            STATUS_PASS,
            f"mx.quantize/mx.dequantize are deterministic, shape-"
            f"preserving, finite, and within the quantisation step "
            f"bound (group_size={group_size}, bits={bits})"
            + (
                "; QuantizedKVCache checkpoints round-trip byte-stably"
                if ctx.mlx_lm
                else ""
            ),
        )


def _check_prompt_cache_format(ctx: _ProbeContext, results: list[CheckResult]) -> None:
    """Prompt-cache format: round-trip, class-name storage, fail-closed
    guards (cache class + model identity)."""
    problems: list[str] = []
    if not ctx.mlx_lm:
        problems.append(
            "mlx_lm is not importable; the prompt-cache format "
            "(save_prompt_cache / load_prompt_cache) cannot be probed"
        )
    else:
        try:
            from mlx_lm.models.cache import (
                KVCache,
                load_prompt_cache,
                save_prompt_cache,
            )
        except ImportError as e:
            problems.append(
                f"mlx_lm.models.cache is not importable ({e}); the "
                "prompt-cache format cannot be probed"
            )
        else:
            mx = ctx.mx
            assert mx is not None  # run_conformance only calls probes with mlx present
            try:
                with tempfile.TemporaryDirectory(prefix="qmlx_conformance_") as d:
                    p = os.path.join(d, "kv.safetensors")
                    c = KVCache()
                    c.keys = [mx.ones((2, 4, 32), dtype=mx.bfloat16)]
                    c.values = [mx.ones((2, 4, 32), dtype=mx.bfloat16)]
                    c.offset = 8
                    save_prompt_cache(p, [c])
                    loaded, _ = load_prompt_cache(p, return_metadata=True)
                    if type(loaded[0]).__name__ != "KVCache":
                        problems.append(
                            "KVCache class identity did not round-trip "
                            f"(loaded {type(loaded[0]).__name__})"
                        )
                    else:
                        mx.eval(loaded[0].keys[0])
                        if not bool(mx.all(loaded[0].keys[0] == c.keys[0])):
                            problems.append(
                                "KVCache values were corrupted by the "
                                "save/load round-trip"
                            )
                    # (b) the class name must be stored in the on-disk
                    # format — the restore identity guard keys on it.
                    meta = _read_safetensors_metadata(p)
                    meta_values = [v for v in meta.values() if isinstance(v, str)]
                    if "KVCache" not in meta_values:
                        problems.append(
                            "the on-disk format does not store the "
                            "cache class name; the restore identity "
                            "guard (model + cache class) cannot vouch "
                            "for the checkpoint"
                        )
                    # (c) fail-closed: an unknown class name must make
                    # the loader refuse, never return the wrong class.
                    p2 = os.path.join(d, "unknown.safetensors")
                    mx.save_safetensors(
                        p2,
                        {
                            "0.0.0": mx.ones((2, 4, 32), dtype=mx.bfloat16),
                            "0.1.0": mx.ones((2, 4, 32), dtype=mx.bfloat16),
                            "0.2": mx.array(8),
                        },
                        {
                            "1.0": "NoSuchCacheClass",
                            "2.0": "",
                            "2.1.0": "0.2",
                            "2.1.1": "scalar",
                        },
                    )
                    try:
                        load_prompt_cache(p2, return_metadata=True)
                        problems.append(
                            "an unknown cache class name loaded without "
                            "error; the guard must fail closed"
                        )
                    except Exception:
                        pass  # fail-closed: the loader refused
            except Exception as e:
                problems.append(
                    f"the prompt-cache probe raised {type(e).__name__}: {e}"
                )
    # (d) the model identity guard is fail-closed (pure logic — runs
    # even when mlx_lm is absent, as long as the probe context exists).
    from vllm_mlx.runtime.disk_kv_checkpoint import model_identity_matches

    cases: list[tuple[bool, Any, Any]] = [
        (True, "A", "A"),
        (False, "A", "B"),
        (False, None, "A"),
        (False, "A", None),
        (False, "", "A"),
    ]
    bad = [
        f"model_identity_matches({run!r}, {ckpt!r}) = "
        f"{model_identity_matches(run, ckpt)!r}, expected {expected}"
        for expected, run, ckpt in cases
        if model_identity_matches(run, ckpt) is not expected
    ]
    if bad:
        problems.append(
            "the model identity guard is not fail-closed: " + "; ".join(bad)
        )
    if problems:
        status = (
            STATUS_CAPABILITY_MISSING
            if len(problems) == 1
            and problems[0].startswith(
                ("mlx_lm is not importable", "mlx_lm.models.cache")
            )
            else STATUS_FAIL
        )
        prefix = (
            "backend gap: " if status == STATUS_CAPABILITY_MISSING else "restore bug: "
        )
        _add(results, "prompt_cache_format", status, prefix + "; ".join(problems))
    else:
        _add(
            results,
            "prompt_cache_format",
            STATUS_PASS,
            "save/load round-trip preserves class identity and values; "
            "the class name is stored on disk; an unknown class name "
            "fails closed; the model identity guard rejects on any "
            "doubt",
        )


def _check_stream_ownership(ctx: _ProbeContext, results: list[CheckResult]) -> None:
    mx = ctx.mx
    assert mx is not None  # run_conformance only calls probes with mlx present
    missing = [
        n
        for n in (
            "new_thread_local_stream",
            "default_stream",
            "set_default_stream",
        )
        if not hasattr(mx, n)
    ]
    if missing:
        _add(
            results,
            "stream_ownership",
            STATUS_CAPABILITY_MISSING,
            "backend gap: the stream API is absent ("
            + ", ".join(missing)
            + "); the owner-thread stream semantics cannot be probed",
        )
        return
    dev = mx.default_device()
    observed: str | None = None
    try:
        s = mx.new_thread_local_stream(dev)
        with mx.stream(s):
            mx.eval(mx.array([0.0]) + mx.array([1.0]))
        observed = "multi"
    except RuntimeError as e:
        msg = str(e)
        if "Stream" in msg and (
            "no Stream" in msg or "not exist" in msg or "does not exist" in msg
        ):
            observed = "single"
        else:
            _add(
                results,
                "stream_ownership",
                STATUS_FAIL,
                f"unexpected stream failure mode: RuntimeError: {msg}",
            )
            return
    except Exception as e:
        _add(
            results,
            "stream_ownership",
            STATUS_FAIL,
            f"unexpected stream failure mode: {type(e).__name__}: {e}",
        )
        return
    # The default stream must be reachable from a non-owner thread
    # (read-only; the owner-thread invariant is about work submission,
    # not about reading the default stream).
    probe: dict[str, Any] = {}

    def _work() -> None:
        try:
            probe["ok"] = mx.default_stream(dev) is not None
        except Exception as e:  # noqa: BLE001
            probe["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_work)
    t.start()
    t.join()
    if "err" in probe:
        _add(
            results,
            "stream_ownership",
            STATUS_FAIL,
            f"the default stream is not reachable from a non-owner "
            f"thread ({probe['err']}); every thread must be able to "
            "read the default stream",
        )
        return
    # The owner thread must be able to set the default stream.
    try:
        mx.set_default_stream(mx.default_stream(dev))
    except Exception as e:
        _add(
            results,
            "stream_ownership",
            STATUS_FAIL,
            f"set_default_stream raised in the owner thread: {type(e).__name__}: {e}",
        )
        return
    detail = (
        f"observed stream semantics: {observed} "
        f"(new_thread_local_stream "
        f"{'works' if observed == 'multi' else 'raises the single-stream error'}; "
        "the default stream is reachable from a second thread; "
        "set_default_stream works in the owner thread)"
    )
    if ctx.backend is not None:
        try:
            declared = bool(ctx.backend.supports_secondary_streams())
        except Exception as e:
            _add(
                results,
                "stream_ownership",
                STATUS_FAIL,
                f"supports_secondary_streams() raised {type(e).__name__}: {e}",
            )
            return
        if (observed == "multi") != declared:
            _add(
                results,
                "stream_ownership",
                STATUS_FAIL,
                f"the backend declares supports_secondary_streams="
                f"{declared} but the live probe observed {observed}; the "
                "seam report is wrong",
            )
            return
        caps = ctx.backend.capabilities()
        expected_model = (
            StreamModel.MULTI_STREAM if declared else StreamModel.SINGLE_STREAM
        )
        if caps.stream_model != expected_model:
            _add(
                results,
                "stream_ownership",
                STATUS_FAIL,
                f"capabilities().stream_model={caps.stream_model} "
                f"contradicts supports_secondary_streams()={declared} "
                f"(expected {expected_model})",
            )
            return
        detail += f"; declared {declared} (consistent)"
    else:
        detail += (
            "; no backend resolved on this device; the declared "
            "semantics are not cross-checked"
        )
    _add(results, "stream_ownership", STATUS_PASS, detail)


def _check_compile_through(ctx: _ProbeContext, results: list[CheckResult]) -> None:
    mx = ctx.mx
    assert mx is not None  # run_conformance only calls probes with mlx present
    if not hasattr(mx, "compile"):
        _add(
            results,
            "compile_through",
            STATUS_CAPABILITY_MISSING,
            "backend gap: mx.compile is absent; the seeded-sampler fast "
            "path must degrade cleanly (the scheduler must not assume "
            "compile-through)",
        )
        return
    # The MLX compile cache lives under $TMPDIR/mlx; isolate the probe
    # so it never depends on host /tmp permissions or a stale cache.
    old_tmpdir = os.environ.get("TMPDIR")
    tmp = tempfile.mkdtemp(prefix="qmlx_conformance_compile_")
    os.environ["TMPDIR"] = tmp
    try:
        try:
            f = mx.compile(lambda a: a * 2)
            a = mx.array([1.0, 2.0])
            r = f(a)
            mx.eval(r)
            eager = a * 2
            mx.eval(eager)
            if list(r.tolist()) != list(eager.tolist()):
                _add(
                    results,
                    "compile_through",
                    STATUS_FAIL,
                    f"compile-through: the compiled result {r.tolist()} "
                    f"differs from eager {eager.tolist()}; the "
                    "seeded-sampler reproducibility boundary is at risk",
                )
                return
            r2 = f(a)
            mx.eval(r2)
            if list(r2.tolist()) != list(r.tolist()):
                _add(
                    results,
                    "compile_through",
                    STATUS_FAIL,
                    "compile-through: not deterministic across calls",
                )
                return
        except Exception as e:
            _add(
                results,
                "compile_through",
                STATUS_FAIL,
                f"mx.compile is present but the probe raised {type(e).__name__}: {e}",
            )
            return
    finally:
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir
        shutil.rmtree(tmp, ignore_errors=True)
    if ctx.backend is not None:
        caps = ctx.backend.capabilities()
        if not caps.compile_through:
            _add(
                results,
                "compile_through",
                STATUS_FAIL,
                "capabilities().compile_through=False but mx.compile "
                "works; the seam report is wrong",
            )
            return
    _add(
        results,
        "compile_through",
        STATUS_PASS,
        "compile-through works (compiled == eager, deterministic across "
        "calls); the probe runs in a TMPDIR-isolated cache directory",
    )


def _check_async_eval(ctx: _ProbeContext, results: list[CheckResult]) -> None:
    mx = ctx.mx
    assert mx is not None  # run_conformance only calls probes with mlx present
    if not hasattr(mx, "async_eval"):
        _add(
            results,
            "async_eval",
            STATUS_CAPABILITY_MISSING,
            "backend gap: mx.async_eval is absent; the scheduler's eval "
            "pump must degrade (evaluating in place)",
        )
        return
    try:
        x = mx.array([1.0])
        y = x + 1
        z = y + 1
        mx.async_eval()
        if float(z.item()) != 3.0:
            _add(
                results,
                "async_eval",
                STATUS_FAIL,
                f"async_eval: the pending chain was not evaluated in "
                f"order (z={float(z.item())}, expected 3.0)",
            )
            return
    except Exception as e:
        _add(
            results,
            "async_eval",
            STATUS_FAIL,
            f"mx.async_eval is present but the probe raised {type(e).__name__}: {e}",
        )
        return
    if ctx.backend is not None:
        caps = ctx.backend.capabilities()
        if not caps.async_eval:
            _add(
                results,
                "async_eval",
                STATUS_FAIL,
                "capabilities().async_eval=False but mx.async_eval "
                "works; the seam report is wrong",
            )
            return
    _add(
        results,
        "async_eval",
        STATUS_PASS,
        "async_eval evaluates the pending chain in order",
    )


_DEVICE_INFO_KEYS: tuple[tuple[str, type], ...] = (
    ("device_name", str),
    ("memory_size", int),
    ("max_recommended_working_set_size", int),
)


def _check_device_info_keys(ctx: _ProbeContext, results: list[CheckResult]) -> None:
    mx = ctx.mx
    assert mx is not None  # run_conformance only calls probes with mlx present
    try:
        info = mx.device_info()
    except Exception as e:
        for key, _ in _DEVICE_INFO_KEYS:
            _add(
                results,
                f"device_info.{key}",
                STATUS_FAIL,
                f"mx.device_info() raised {type(e).__name__}: {e}",
            )
        return
    if not isinstance(info, dict):
        for key, _ in _DEVICE_INFO_KEYS:
            _add(
                results,
                f"device_info.{key}",
                STATUS_FAIL,
                f"mx.device_info() returned {type(info).__name__}, expected a dict",
            )
        return
    for key, expected_type in _DEVICE_INFO_KEYS:
        name = f"device_info.{key}"
        if key not in info:
            _add(
                results,
                name,
                STATUS_CAPABILITY_MISSING,
                f"backend gap: mx.device_info() has no {key!r} on this device",
            )
            continue
        v = info[key]
        if isinstance(v, bool) or not isinstance(v, expected_type):
            _add(
                results,
                name,
                STATUS_FAIL,
                f"device_info[{key!r}] is {type(v).__name__} ({v!r}); "
                f"expected {expected_type.__name__}",
            )
            continue
        if ctx.backend is not None:
            if key == "device_name":
                seam = ctx.backend.device_name()
                if seam != v:
                    _add(
                        results,
                        name,
                        STATUS_FAIL,
                        f"seam report is wrong: backend.device_name() "
                        f"={seam!r} but device_info has {v!r}",
                    )
                    continue
            elif key == "memory_size":
                seam = ctx.backend.total_memory_bytes()
                if seam != v:
                    _add(
                        results,
                        name,
                        STATUS_FAIL,
                        f"seam report is wrong: "
                        f"backend.total_memory_bytes()={seam} but "
                        f"device_info has {v}",
                    )
                    continue
            else:  # max_recommended_working_set_size
                seam = ctx.backend.recommended_working_set_bytes()
                if seam != v:
                    _add(
                        results,
                        name,
                        STATUS_FAIL,
                        f"seam report is wrong: "
                        f"backend.recommended_working_set_bytes()={seam} "
                        f"but device_info has {v}",
                    )
                    continue
        _add(
            results,
            name,
            STATUS_PASS,
            f"present and {expected_type.__name__}-typed"
            + (
                f"; seam parity holds (backend accessor returns {v!r})"
                if ctx.backend is not None
                else "; no backend resolved; seam parity not checked"
            ),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_conformance(
    mx: Any = _UNSET,
    backend: Any = _UNSET,
    gap_register: list[str] | None = None,
) -> ConformanceReport:
    """Run the conformance suite.

    Args:
        mx: the ``mlx.core`` module to probe. Default: import
            ``mlx.core``. Pass ``None`` for metadata-only mode (mlx not
            importable — the device probes are skipped with a report
            line).
        backend: the ``Backend`` to test. Default: ``get_backend()``
            (``None`` when no GPU backend is present).
        gap_register: extra check names to treat as documented gaps
            (the port's gap register).

    Returns:
        The ``ConformanceReport``.
    """
    if mx is _UNSET:
        try:
            import mlx.core as mx  # noqa: F811
        except Exception:
            mx = None
    if backend is _UNSET:
        backend = get_backend()

    device_class: str | None = None
    if mx is not None:
        device_class = _device_class(mx)

    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "qmlx": _package_version("qmlx"),
        "mlx": getattr(mx, "__version__", None) if mx is not None else None,
        "mlx_lm": _package_version("mlx-lm") if mx is not None else None,
        "device": (repr(mx.default_device()) if mx is not None else None),
        "device_class": device_class,
        "backend": (
            getattr(backend.kind, "value", backend.kind)
            if backend is not None
            else None
        ),
        "probes_ran": mx is not None,
    }

    mlx_lm_present = False
    if mx is not None:
        try:
            import mlx_lm  # noqa: F401

            mlx_lm_present = True
        except ImportError:
            pass

    ctx = _ProbeContext(
        mx=mx,
        mlx_lm=mlx_lm_present,
        backend=backend,
        device_class=device_class,
        environment=environment,
    )

    results: list[CheckResult] = []
    _check_environment(ctx, results)
    if mx is not None:
        _check_seam_contract(ctx, results)
        _check_memory_cap_enforcement(ctx, results)
        _check_quantized_kv_roundtrip(ctx, results)
        _check_prompt_cache_format(ctx, results)
        _check_stream_ownership(ctx, results)
        _check_compile_through(ctx, results)
        _check_async_eval(ctx, results)
        _check_device_info_keys(ctx, results)

    return ConformanceReport(
        checks=results,
        environment=environment,
        documented_gaps=frozenset(documented_gap_names(device_class, gap_register)),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``python -m vllm_mlx.device.conformance``)."""
    parser = argparse.ArgumentParser(
        prog="python -m vllm_mlx.device.conformance",
        description=(
            "qMLX backend conformance suite — the mlx-rocm / mlx-cuda "
            "acceptance gate (repositioning item 7 of 9)"
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="write the machine-readable report to PATH",
    )
    parser.add_argument(
        "--gaps",
        metavar="FILE",
        help=(
            "JSON file: a list of check names to treat as documented "
            "gaps (the port's gap register)"
        ),
    )
    args = parser.parse_args(argv)
    gap_register: list[str] | None = None
    if args.gaps:
        try:
            with open(args.gaps, encoding="utf-8") as fh:
                gap_register = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"error: cannot read gap register {args.gaps!r}: {e}", file=sys.stderr
            )
            return 2
        if not isinstance(gap_register, list) or not all(
            isinstance(n, str) for n in gap_register
        ):
            print(
                f"error: {args.gaps} must contain a JSON list of check names",
                file=sys.stderr,
            )
            return 2
    report = run_conformance(gap_register=gap_register)
    print(report.format_table())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
            fh.write("\n")
    return 0 if report.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
