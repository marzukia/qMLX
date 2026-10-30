# SPDX-License-Identifier: Apache-2.0
"""D23.1: ``--wired-limit-gb`` (per-process GPU wired / resident cap).

Carbon parity journal (``/home/monky/reports/carbon-qmlx-perf/journal.md``,
entries E10-E11 + D23.1): ``--metal-residency-gb`` (PR #115) sets an
ADVISORY allocator hint (``set_memory_limit``,
``enforcement_mode == HINT``). The 10.4GB TWIN oQ2 27B weight set
stays FULLY WIRED at hint 9.5G AND 8.0G (no behavioral change), and
the first generation-step eval OOMs
(``kIOGPUCommandBufferCallbackErrorOutOfMemory``) — 4/4 rows, 0 tokens.

MTPLX runs the same model on the same box and survives (9.99GB active
under a 12.17GB peak at a 12GB budget) via a per-process WIRED limit
(``mx.set_wired_limit``): above the wired limit the OS pages weight
memory out to RAM/swap and it re-establishes lazily on the next
submit — NOT an allocator hint. This flag wires that mechanism into
the qMLX serve path behind an optional ``--wired-limit-gb``
(default ``None`` = no behavior change).

These tests pin:
- CLI flag wiring (parse + default, ``load_model`` plumbing, the
  DFlash/DDTree ignored-list entries);
- startup validation (non-positive rejected with a clear error before
  any device probe — the same posture as the residency flag);
- the default path never touches a wired setter (no behavior change);
- apply-time clamp to the system ``max_recommended_working_set_size``
  (MLX refuses a wired limit above it — a refused limit wires
  nothing, MTPLX parity);
- graceful no-op (warning, no crash) when the backend has no
  ``set_wired_limit`` (ported backends / older mlx);
- the real-Metal gate (Apple-Silicon CI only, like the ``test-macos``
  job) actually sets a wired limit on the live device.
"""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx import cli
from vllm_mlx.device.metal import MetalBackend
from vllm_mlx.engine.batched import (
    BatchedEngine,
    _compute_wired_limit_bytes,
)

# Carbon M1 Pro 16GB recommended working set (journal E1/E5: 11.45GB).
CARBON_WORKING_SET = 11_453_251_584


def _capture_serve_args(argv: list[str]):
    """Parse through the REAL ``cli.main()`` and return the Namespace
    ``serve_command`` would have received (same pattern as
    ``test_serve_listen_fd.py``). Uses a real alias so the model-name
    validation before dispatch passes."""
    captured: list = []
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    return captured[0]


class TestCliFlagWiring:
    """``--wired-limit-gb`` parses and reaches ``load_model``."""

    def test_flag_parses(self):
        args = _capture_serve_args(
            [
                "qmlx",
                "serve",
                "qwen3.5-4b-4bit",
                "--wired-limit-gb",
                "8.0",
            ]
        )
        assert args.wired_limit_gb == 8.0

    def test_flag_default_is_none(self):
        """Default off: no wired limit, no behavior change."""
        args = _capture_serve_args(["qmlx", "serve", "qwen3.5-4b-4bit"])
        assert args.wired_limit_gb is None

    def test_flag_plumbed_into_load_model(self):
        """Structural pin: the ``load_model`` call in
        ``serve_command`` must forward the parsed value (the #400
        silent-flag-drop failure mode)."""
        source = inspect.getsource(cli.serve_command)
        assert "wired_limit_gb=args.wired_limit_gb" in source

    def test_flag_in_dflash_and_ddtree_ignored_lists(self):
        """DFlash/DDTree bypass BatchedEngine — the flag must be
        surfaced in both ignored-flag warnings (same as
        --metal-residency-gb)."""
        source = inspect.getsource(cli.serve_command)
        assert (
            '_dflash_ignored.append("--wired-limit-gb")' in source
            and '_ddtree_ignored.append("--wired-limit-gb")' in source
        )


class TestStartupValidation:
    """``BatchedEngine.__init__`` validates before any model load."""

    def test_nonpositive_wired_rejected(self):
        """``--wired-limit-gb 0`` is nonsense — clear error before any
        device probe (works even without a backend)."""
        with pytest.raises(ValueError, match="wired_limit_gb must be > 0"):
            BatchedEngine("test-model", wired_limit_gb=0.0)

    def test_negative_wired_rejected(self):
        with pytest.raises(ValueError, match="wired_limit_gb must be > 0"):
            BatchedEngine("test-model", wired_limit_gb=-1.0)

    def test_positive_wired_accepted_no_backend_probe(self):
        """The wired axis is independent of the residency/cap axes —
        no cap-relationship check at startup (a wired cap below the
        weight size is exactly the point). Accepted even with no GPU
        backend (the clamp happens at apply time)."""
        with patch("vllm_mlx.device.get_backend", return_value=None):
            engine = BatchedEngine("test-model", wired_limit_gb=8.0)
        assert engine._wired_limit_gb == 8.0  # noqa: SLF001

    def test_flag_is_not_part_of_scheduler_config(self):
        """Structural pin: the wired cap must never leak into the
        D-METAL-CAP admission inputs."""
        from vllm_mlx.scheduler import SchedulerConfig

        fields = getattr(SchedulerConfig, "__dataclass_fields__", {})
        assert "wired_limit_gb" not in fields


class TestDefaultPathNoBehaviorChange:
    """``wired_limit_gb=None`` (default) never touches a wired
    setter — the serve path is byte-identical to pre-change."""

    def test_default_flag_is_none(self):
        engine = BatchedEngine("test-model")
        assert engine._wired_limit_gb is None  # noqa: SLF001

    def test_default_apply_is_noop(self):
        engine = BatchedEngine("test-model")
        calls: list = []
        backend = SimpleNamespace(set_wired_limit=calls.append)
        fragment = engine._apply_wired_limits(backend, CARBON_WORKING_SET)
        assert calls == []
        assert fragment == ""

    def test_compute_no_clamp_below_max(self):
        wired, clamped = _compute_wired_limit_bytes(8.0, CARBON_WORKING_SET)
        assert wired == int(8.0 * 1e9)
        assert clamped is False

    def test_compute_clamps_above_max(self):
        """MLX refuses a wired limit above the system maximum (a
        refused limit wires nothing) — clamp, MTPLX parity."""
        wired, clamped = _compute_wired_limit_bytes(12.0, CARBON_WORKING_SET)
        assert wired == CARBON_WORKING_SET
        assert clamped is True

    def test_compute_zero_max_never_clamps(self):
        """No working-set info (``max_recommended == 0``): no clamp
        possible, pass the requested value through."""
        wired, clamped = _compute_wired_limit_bytes(12.0, 0)
        assert wired == int(12.0 * 1e9)
        assert clamped is False


class TestApplyWiredLimits:
    """The apply path (``_set_metal_limits`` calls this per boot)."""

    def _engine(self, wired_gb: float) -> BatchedEngine:
        return BatchedEngine("test-model", wired_limit_gb=wired_gb)

    def test_applies_with_setter(self):
        engine = self._engine(8.0)
        calls: list = []
        backend = SimpleNamespace(set_wired_limit=calls.append)
        fragment = engine._apply_wired_limits(backend, CARBON_WORKING_SET)
        assert calls == [int(8.0 * 1e9)]
        assert "wired_limit=8.0GB" in fragment
        assert "enforcement=wired" in fragment

    def test_clamps_above_max_with_note(self):
        engine = self._engine(12.0)
        calls: list = []
        backend = SimpleNamespace(set_wired_limit=calls.append)
        fragment = engine._apply_wired_limits(backend, CARBON_WORKING_SET)
        assert calls == [CARBON_WORKING_SET]
        assert "clamped to max_recommended" in fragment

    def test_no_setter_is_warned_noop(self, caplog):
        """Ported backends / older mlx have no wired-limit API —
        warn, never crash (the flag is Metal-first by design)."""
        engine = self._engine(8.0)
        backend = SimpleNamespace(kind="rocm", enforcement_mode="hard")
        import logging

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.engine.batched"):
            fragment = engine._apply_wired_limits(backend, CARBON_WORKING_SET)
        assert fragment == ""
        assert any("wired-limit API" in m for m in caplog.messages)

    def test_nonpositive_never_reaches_apply(self):
        """Startup validation owns the ``<= 0`` rejection — by the
        time ``_apply_wired_limits`` runs the value is positive (or
        the engine was not constructed at all)."""
        with pytest.raises(ValueError, match="wired_limit_gb must be > 0"):
            BatchedEngine("test-model", wired_limit_gb=0.0)


class TestMetalBackendWiredLimit:
    """``MetalBackend.set_wired_limit`` parity (fake ``mlx.core``)."""

    def _fake_mx(self, *, top_level: bool = True, metal_level: bool = True):
        mx = SimpleNamespace()

        def _top(n):
            mx._wired = n

        metal = SimpleNamespace()
        if metal_level:

            def _metal_set(n):
                mx._wired = f"metal:{n}"

            metal.set_wired_limit = _metal_set
        mx.metal = metal
        if top_level:
            mx.set_wired_limit = _top
        return mx

    def test_top_level_api_preferred(self):
        mx = self._fake_mx()
        MetalBackend(mx=mx).set_wired_limit(1234)
        assert mx._wired == 1234  # not "metal:1234"

    def test_falls_back_to_metal_level(self):
        mx = self._fake_mx(top_level=False)
        MetalBackend(mx=mx).set_wired_limit(1234)
        assert mx._wired == "metal:1234"

    def test_neither_api_raises(self):
        mx = self._fake_mx(top_level=False, metal_level=False)
        with pytest.raises(AttributeError, match="wired-limit API unavailable"):
            MetalBackend(mx=mx).set_wired_limit(1234)


# ---------------------------------------------------------------------------
# Real-Metal gate (Apple-Silicon CI only — the macos-14 job; skipped
# everywhere else, same posture as
# ``test_device_backend.test_parity_against_real_mlx``)
# ---------------------------------------------------------------------------


def _real_mx_or_none():
    try:
        import mlx.core as mx

        return mx
    except Exception:
        return None


def test_real_metal_wired_limit_applies():
    """The whole point of the flag: on a live Apple-Silicon device the
    wired limit actually sets (and a limit above the system maximum is
    the clamp case, not a crash)."""
    mx = _real_mx_or_none()
    if mx is None:
        pytest.skip("mlx not installed (Linux CI)")
    if not getattr(mx.metal, "is_available", lambda: False)():
        pytest.skip("no Metal device on this machine")
    if not hasattr(mx, "set_wired_limit"):
        pytest.skip("this mlx build has no wired-limit API")
    max_recommended = int(mx.device_info().get("max_recommended_working_set_size", 0))
    if max_recommended <= 0:
        pytest.skip("no recommended working set reported")
    # 1 GiB is safely under any real machine's maximum — no clamp
    # expected; the call must succeed on the live device.
    mx.set_wired_limit(1 * 2**30)
