# SPDX-License-Identifier: Apache-2.0
"""Cold-prefill frequency dashboard (Phase 0b).

Tracks how often cold prefill fires vs cache hits, exposing:

* ``qmlx_prefill_total{kind="cold|extend|exact"}`` — monotonic counters
  for each prefill kind (already tracked in HonestMetrics.prefill_kind).
* ``qmlx_kv_restore_total{result="hit|miss"}`` — monotonic counters for
  disk KV restore attempts (already tracked in HonestMetrics.kv_restore_result).
* ``qmlx_cold_prefill_ratio`` — rolling-window gauge: cold_prefill_count /
  total_prefill_count over the last N requests.

Also adds a one-shot log at startup and every 100 requests summarizing:
"PREFILL STATS: N cold, M extend, P exact (X% cold)".

Thread-safe: uses the same locking pattern as HonestMetrics.
Process-lifetime monotonic: counters never reset.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .honest_metrics import HonestMetrics

logger = logging.getLogger(__name__)


class PrefillFrequencyDashboard:
    """Rolling-window dashboard for cold-prefill frequency tracking.

    Maintains:
    - Process-lifetime monotonic counters for prefill kinds and KV restore results
    - A rolling window of the last N requests for computing the cold ratio gauge

    Args:
        window_size: Number of recent requests to include in the cold ratio calculation.
            Defaults to 1000 (roughly 10 seconds at 100 req/s).
    """

    __slots__ = (
        "_lock",
        "_window_size",
        "_recent_prefill_kinds",
        "total_cold",
        "total_extend",
        "total_exact",
        "total_kv_restore_hit",
        "total_kv_restore_miss",
        "_request_count",
    )

    def __init__(self, window_size: int = 1000) -> None:
        self._lock = threading.Lock()
        self._window_size = max(1, window_size)
        # Rolling window: stores "cold", "extend", or "exact" for recent requests
        self._recent_prefill_kinds: deque[str] = deque(maxlen=self._window_size)

        # Process-lifetime monotonic counters
        self.total_cold: int = 0
        self.total_extend: int = 0
        self.total_exact: int = 0
        self.total_kv_restore_hit: int = 0
        self.total_kv_restore_miss: int = 0

        # Total requests processed (for periodic logging)
        self._request_count: int = 0

    def record_prefill_kind(self, kind: str) -> None:
        """Record one prefill event by kind.

        Args:
            kind: One of "cold", "extend", or "exact".
        """
        with self._lock:
            self._request_count += 1
            self._recent_prefill_kinds.append(kind)

            if kind == "cold":
                self.total_cold += 1
            elif kind == "extend":
                self.total_extend += 1
            elif kind == "exact":
                self.total_exact += 1
            else:
                # Unknown kind: still count it but don't attribute to any series
                pass

            # Periodic log: startup (first request) + every 100 requests
            if self._request_count == 1 or self._request_count % 100 == 0:
                self._log_summary()

    def record_kv_restore_result(self, hit: bool) -> None:
        """Record one disk KV restore attempt.

        Args:
            hit: True if the restore hit, False if it missed.
        """
        with self._lock:
            if hit:
                self.total_kv_restore_hit += 1
            else:
                self.total_kv_restore_miss += 1

    def snapshot(self) -> dict[str, object]:
        """Copy out current state for Prometheus exposition.

        Returns:
            Dict with:
            - total_cold, total_extend, total_exact: monotonic counters
            - total_kv_restore_hit, total_kv_restore_miss: monotonic counters
            - cold_prefill_ratio: rolling-window gauge (0.0 if no requests yet)
            - request_count: total requests processed (for debugging)
        """
        with self._lock:
            total_prefill = len(self._recent_prefill_kinds)
            if total_prefill == 0:
                cold_ratio = 0.0
            else:
                cold_in_window = sum(
                    1 for k in self._recent_prefill_kinds if k == "cold"
                )
                cold_ratio = round(cold_in_window / total_prefill, 4)

            return {
                "total_cold": self.total_cold,
                "total_extend": self.total_extend,
                "total_exact": self.total_exact,
                "total_kv_restore_hit": self.total_kv_restore_hit,
                "total_kv_restore_miss": self.total_kv_restore_miss,
                "cold_prefill_ratio": cold_ratio,
                "request_count": self._request_count,
            }

    def _log_summary(self) -> None:
        """Log a summary of prefill statistics."""
        total = self.total_cold + self.total_extend + self.total_exact
        if total == 0:
            cold_pct = 0.0
        else:
            cold_pct = round((self.total_cold / total) * 100, 1)

        msg = (
            f"PREFILL STATS: {self.total_cold} cold, "
            f"{self.total_extend} extend, {self.total_exact} exact "
            f"({cold_pct}% cold)"
        )

        if self._request_count == 1:
            logger.info(msg)
        else:
            logger.info(msg)


# Global singleton instance (one per process)
_dashboard: PrefillFrequencyDashboard | None = None
_dashboard_lock = threading.Lock()


def get_dashboard(window_size: int = 1000) -> PrefillFrequencyDashboard:
    """Get or create the global dashboard singleton.

    Args:
        window_size: Rolling window size (only used on first call).

    Returns:
        The global PrefillFrequencyDashboard instance.
    """
    global _dashboard
    with _dashboard_lock:
        if _dashboard is None:
            _dashboard = PrefillFrequencyDashboard(window_size=window_size)
        return _dashboard


def reset_dashboard_for_tests() -> None:
    """Test-only hook: reset the global dashboard singleton."""
    global _dashboard
    with _dashboard_lock:
        _dashboard = None


def snapshot() -> dict[str, object]:
    """Convenience: snapshot the global dashboard.

    Returns:
        Same as PrefillFrequencyDashboard.snapshot().
    """
    return get_dashboard().snapshot()