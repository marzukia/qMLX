# SPDX-License-Identifier: Apache-2.0
"""D11 decouple: ``--metal-residency-gb`` (residency hint vs admission cap).

Carbon parity journal (``/home/monky/reports/carbon-qmlx-perf/journal.md``,
entries E5-E9 + D10-D11): on a 16GB M1 Pro serving a ~10.4GB TWIN oQ2
27B model, 8K prefill always failed. ``--gpu-memory-utilization`` drove
BOTH the Metal weight-residency hint (``set_memory_limit``,
``enforcement_mode == HINT``) and the D-METAL-CAP admission cap
(``Scheduler._resolve_metal_cap_bytes``):

- util 0.90 → cap 10.3GB < 10.4GB weights → every request rejected at
  admission (E5);
- util 1.0 → hint 11.5GB >= weights → weights FULLY resident → 8K
  prefill peak (~11.7-12GB) exceeds the 11.5GB working set →
  ``kIOGPUCommandBufferCallbackErrorOutOfMemory`` (E6-E9).

No single value admitted AND forced partial weight residency. MTPLX
survives via ``--allow-swap`` (9.99GB active, weights page RAM/swap).
The fix decouples the two: ``--metal-residency-gb`` sets the residency
hint only; the admission cap stays at ``util x recommended working
set``; the pair is validated at startup (residency must be <= cap).

These tests pin:
- startup rejection when the residency hint exceeds the cap;
- the default path is unchanged (``None`` → hint == cap, exactly the
  pre-change math);
- the admission cap math is untouched by the flag (both in the
  ``_compute_metal_limit_bytes`` helper and in the real
  ``Scheduler._resolve_metal_cap_bytes``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.engine.batched import BatchedEngine, _compute_metal_limit_bytes
from vllm_mlx.scheduler import Scheduler, SchedulerConfig

# Carbon M1 Pro 16GB recommended working set (journal E1/E5).
CARBON_WORKING_SET = 11_500_000_000


def _fake_backend(recommended: int) -> SimpleNamespace:
    """Backend stub carrying only the attributes the limit path reads."""
    return SimpleNamespace(
        recommended_working_set_bytes=lambda: recommended,
        set_memory_limit=lambda limit_bytes: None,
        set_cache_limit=lambda limit_bytes: None,
    )


class TestStartupResidencyValidation:
    """``BatchedEngine.__init__`` rejects residency > cap at startup."""

    def test_residency_above_cap_rejected_with_clear_error(self):
        """E5 mode: 11.0GB hint > 10.35GB cap (util 0.90 x 11.5GB)
        must fail BEFORE any model load with both numbers in the
        message."""
        with (
            patch(
                "vllm_mlx.device.get_backend",
                return_value=_fake_backend(CARBON_WORKING_SET),
            ),
            pytest.raises(
                ValueError,
                match=r"exceeds the D-METAL-CAP admission cap 10\.35GB",
            ),
        ):
            BatchedEngine(
                "test-model",
                gpu_memory_utilization=0.90,
                metal_residency_gb=11.0,
            )

    def test_residency_equal_to_cap_accepted(self):
        """Boundary: residency == cap is legal (the validation is
        ``>``, matching the admission gate's ``>=`` active-side)."""
        with patch(
            "vllm_mlx.device.get_backend",
            return_value=_fake_backend(CARBON_WORKING_SET),
        ):
            engine = BatchedEngine(
                "test-model",
                gpu_memory_utilization=0.90,
                metal_residency_gb=10.35,
            )
        assert engine._metal_residency_gb == 10.35  # noqa: SLF001

    def test_residency_below_cap_accepted(self):
        """The 16GB fix: 9.5GB hint < 10.35GB cap admits (cap 1.0 →
        11.5GB) while keeping weights partially resident."""
        with patch(
            "vllm_mlx.device.get_backend",
            return_value=_fake_backend(CARBON_WORKING_SET),
        ):
            engine = BatchedEngine(
                "test-model",
                gpu_memory_utilization=1.0,
                metal_residency_gb=9.5,
            )
        assert engine._metal_residency_gb == 9.5  # noqa: SLF001

    def test_nonpositive_residency_rejected(self):
        """``--metal-residency-gb 0`` is nonsense — clear error before
        any device probe (works even without a backend)."""
        with pytest.raises(ValueError, match="must be > 0"):
            BatchedEngine(
                "test-model",
                gpu_memory_utilization=0.90,
                metal_residency_gb=0.0,
            )

    def test_cap_disabled_util_zero_skips_validation(self):
        """util <= 0 disables the admission cap (``_resolve_metal_cap_
        bytes`` returns 0 = "do not check") — there is nothing for
        the hint to stay under, so no rejection and no device probe."""
        engine = BatchedEngine(
            "test-model",
            gpu_memory_utilization=0.0,
            metal_residency_gb=1000.0,
        )
        assert engine._metal_residency_gb == 1000.0  # noqa: SLF001

    def test_no_gpu_backend_skips_validation(self):
        """Non-Apple hosts / minimal installs: no backend to probe,
        same "cap unknown" semantics as the scheduler — the flag
        stays set, validation is a no-op."""
        with patch("vllm_mlx.device.get_backend", return_value=None):
            engine = BatchedEngine(
                "test-model",
                gpu_memory_utilization=0.90,
                metal_residency_gb=11.0,
            )
        assert engine._metal_residency_gb == 11.0  # noqa: SLF001


class TestDefaultPathUnchanged:
    """``metal_residency_gb=None`` (default) → identical behavior to
    pre-change: residency hint equals the admission cap."""

    def test_default_flag_is_none(self):
        engine = BatchedEngine("test-model")
        assert engine._metal_residency_gb is None  # noqa: SLF001

    def test_default_limit_math_unchanged(self):
        """hint == cap == util x recommended — the exact expression
        the pre-change ``_set_metal_limits`` inlined."""
        hint, cap = _compute_metal_limit_bytes(None, 0.90, CARBON_WORKING_SET)
        assert cap == int(CARBON_WORKING_SET * 0.90)
        assert hint == cap

    def test_default_full_util(self):
        hint, cap = _compute_metal_limit_bytes(None, 1.0, CARBON_WORKING_SET)
        assert cap == CARBON_WORKING_SET
        assert hint == cap


class TestResidencyDecoupling:
    """The flag overrides ONLY the residency hint; the cap stays at
    ``util x recommended``."""

    def test_residency_overrides_hint_cap_untouched(self):
        hint, cap = _compute_metal_limit_bytes(9.5, 0.90, CARBON_WORKING_SET)
        assert hint == int(9.5 * 1e9)
        # Cap is the SAME number as the default path — decoupled.
        assert cap == int(CARBON_WORKING_SET * 0.90)

    def test_residency_at_full_util(self):
        """The carbon 16GB serving line: util 1.0 (cap 11.5GB admits
        the 10.4GB weights), hint 9.5GB pages weights for partial
        residency."""
        hint, cap = _compute_metal_limit_bytes(9.5, 1.0, CARBON_WORKING_SET)
        assert hint == int(9.5 * 1e9)
        assert cap == CARBON_WORKING_SET


class TestSchedulerCapMathUnaffected:
    """The real ``Scheduler._resolve_metal_cap_bytes`` never sees the
    flag (it is not threaded into ``SchedulerConfig``) — the cap math
    is structurally untouched."""

    def _make_scheduler(self, *, util: float) -> Scheduler:
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            gpu_memory_utilization=util,
        )
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))
        return Scheduler(model=MagicMock(), tokenizer=tokenizer, config=config)

    def test_flag_is_not_part_of_scheduler_config(self):
        """Structural pin: the decouple flag must never leak into the
        admission cap inputs."""
        fields = getattr(SchedulerConfig, "__dataclass_fields__", {})
        assert "metal_residency_gb" not in fields

    def test_cap_math_unchanged(self):
        """cap == util x recommended working set, exactly as pre-
        change — for both the 16GB failure configs."""
        sched = self._make_scheduler(util=0.90)
        with patch(
            "vllm_mlx.device.get_backend",
            return_value=_fake_backend(CARBON_WORKING_SET),
        ):
            assert sched._resolve_metal_cap_bytes() == int(CARBON_WORKING_SET * 0.90)  # noqa: SLF001
        sched_full = self._make_scheduler(util=1.0)
        with patch(
            "vllm_mlx.device.get_backend",
            return_value=_fake_backend(CARBON_WORKING_SET),
        ):
            assert sched_full._resolve_metal_cap_bytes() == (CARBON_WORKING_SET)  # noqa: SLF001
