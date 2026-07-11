# SPDX-License-Identifier: Apache-2.0
"""Honest reuse / latency metrics (issues #10, #2).

Two layers:

1. Pure accumulator unit tests against ``vllm_mlx.honest_metrics`` — no
   scheduler, no engine, no model. Assert each counter / histogram gets
   the exact value for a request with known (prompt, cached, source,
   output, timestamps).
2. Wire-level rendering tests against the Prometheus ``/metrics`` route,
   injecting a fake engine whose ``get_stats()`` carries a
   ``honest_metrics`` snapshot — verifying the exposition, the sticky
   monotonicity across a simulated reset, and the guardrail that no series
   equals the amortized ``(prompt + generated) / wall`` number.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.honest_metrics import (
    TTFT_BUCKET_BOUNDS,
    FixedBucketHistogram,
    HonestMetrics,
)

# ---------------------------------------------------------------------------
# Accumulator unit tests
# ---------------------------------------------------------------------------


def test_cold_request_computes_all_prompt_tokens():
    """A cold miss forwards every prompt token — computed == offered, reused 0."""
    hm = HonestMetrics()
    # cold: cached_tokens == 0, remaining == whole prompt (non-empty)
    hm.record_prefill(
        num_prompt_tokens=200,
        cached_tokens=0,
        cache_hit_type="miss",
        remaining_tokens=list(range(200)),
    )
    snap = hm.snapshot()
    assert snap["prompt_tokens_offered"] == 200
    assert snap["prompt_tokens_computed"] == 200
    assert snap["prompt_tokens_reused"] == {"memory": 0, "disk": 0}
    assert snap["prefill_kind"] == {"cold": 1, "extend": 0, "exact": 0}


def test_full_cache_hit_computes_zero():
    """An exact (full-prompt) hit forwards nothing — computed == 0."""
    hm = HonestMetrics()
    # exact: whole prompt reused, remaining == [] (not None)
    hm.record_prefill(
        num_prompt_tokens=128,
        cached_tokens=128,
        cache_hit_type="exact",
        remaining_tokens=[],
    )
    snap = hm.snapshot()
    assert snap["prompt_tokens_offered"] == 128
    assert snap["prompt_tokens_computed"] == 0
    assert snap["prompt_tokens_reused"] == {"memory": 128, "disk": 0}
    assert snap["prefill_kind"] == {"cold": 0, "extend": 0, "exact": 1}


def test_extend_request_partial_reuse():
    """Partial prefix reuse: computed = prompt - cached, kind == extend."""
    hm = HonestMetrics()
    hm.record_prefill(
        num_prompt_tokens=300,
        cached_tokens=120,
        cache_hit_type="prefix",
        remaining_tokens=list(range(180)),
    )
    snap = hm.snapshot()
    assert snap["prompt_tokens_offered"] == 300
    assert snap["prompt_tokens_computed"] == 180
    assert snap["prompt_tokens_reused"] == {"memory": 120, "disk": 0}
    assert snap["prefill_kind"]["extend"] == 1


def test_reuse_split_memory_vs_disk():
    """cache_hit_type routes reused tokens to the right source bucket."""
    hm = HonestMetrics()
    hm.record_prefill(64, 40, "prefix", list(range(24)))  # memory
    hm.record_prefill(90, 50, "disk", list(range(40)))  # disk restore
    snap = hm.snapshot()
    assert snap["prompt_tokens_reused"] == {"memory": 40, "disk": 50}
    # offered/computed accumulate across both regardless of source
    assert snap["prompt_tokens_offered"] == 64 + 90
    assert snap["prompt_tokens_computed"] == (64 - 40) + (90 - 50)


def test_reuse_not_counted_when_install_scrubbed():
    """cached_tokens==0 (failed install scrubbed it) contributes no reuse.

    Even if the hit_type still reads 'prefix' (the scheduler does not
    always reset it on an insert failure), a scrubbed cached_tokens of 0
    means nothing was actually installed, so reused stays 0 and the
    request is classified cold.
    """
    hm = HonestMetrics()
    hm.record_prefill(
        num_prompt_tokens=100,
        cached_tokens=0,
        cache_hit_type="prefix",
        remaining_tokens=list(range(100)),
    )
    snap = hm.snapshot()
    assert snap["prompt_tokens_reused"] == {"memory": 0, "disk": 0}
    assert snap["prompt_tokens_computed"] == 100
    assert snap["prefill_kind"] == {"cold": 1, "extend": 0, "exact": 0}


def test_prefix_match_type_distribution():
    """Only the five canonical memory-cache match types are tracked."""
    hm = HonestMetrics()
    for mt in ["exact", "exact", "prefix", "lcp", "miss", "supersequence"]:
        hm.record_prefix_match(mt)
    # unknowns (paged 'hit', disk-restore 'disk') are not match types
    hm.record_prefix_match("hit")
    hm.record_prefix_match("disk")
    hm.record_prefix_match(None)
    snap = hm.snapshot()
    assert snap["prefix_cache_match"] == {
        "exact": 2,
        "prefix": 1,
        "supersequence": 1,
        "lcp": 1,
        "miss": 1,
    }


def test_record_disk_restore_hit_and_miss():
    """The accumulator routes hit/miss into the right bucket, neither by default."""
    hm = HonestMetrics()
    assert hm.snapshot()["kv_restore_result"] == {"hit": 0, "miss": 0}
    hm.record_disk_restore(hit=True)
    hm.record_disk_restore(hit=True)
    hm.record_disk_restore(hit=False)
    assert hm.snapshot()["kv_restore_result"] == {"hit": 2, "miss": 1}


# --- scheduler-level wiring of the disk-restore hit/miss counter -----------
#
# These drive the real ``Scheduler._maybe_disk_restore`` control flow with a
# fake ``self`` / ``request`` and a monkeypatched disk-checkpoint module, so
# they assert the ACTUAL attempt boundary (which requests count) rather than
# just the accumulator contract. hit + miss must equal disk-restore attempts.


def _restore_self(honest: HonestMetrics, *, enabled: bool = True):
    return SimpleNamespace(
        config=SimpleNamespace(
            kv_disk_restore_enabled=enabled,
            kv_cache_dtype="bf16",
            metal_pressure_evict_fraction=0.9,
        ),
        _disk_restore_index_built=True,
        _model_name="test-model",
        _resolve_metal_cap_bytes=lambda: 0,  # skip the memory-headroom guard
        _current_metal_active_bytes=lambda: 0,
        honest_metrics=honest,
    )


def _restore_request():
    return SimpleNamespace(
        cache_hit_type="miss",
        prompt_cache=None,
        prompt_token_ids=[1, 2, 3, 4, 5],
        request_id="req-abcdef123456",
        cached_tokens=0,
        remaining_tokens=None,
    )


def _patch_dkc(monkeypatch, *, lookup_result, requires_full=False):
    import vllm_mlx.runtime.disk_kv_checkpoint as dkc

    monkeypatch.setattr(
        dkc,
        "get_content_index",
        lambda: SimpleNamespace(lookup=lambda ids: lookup_result),
    )
    monkeypatch.setattr(dkc, "record_restore_reject", lambda reason: None)
    monkeypatch.setattr(dkc, "record_hook_error", lambda: None)
    monkeypatch.setattr(
        dkc, "model_requires_full_checkpoint", lambda name: requires_full
    )
    return dkc


def test_disk_restore_hit_counts_hit(monkeypatch):
    from vllm_mlx.scheduler import Scheduler

    loaded = SimpleNamespace(
        token_offset=3,
        metadata={"model_name": "test-model"},
        kv_dtype="bf16",
        requires_full_checkpoint=False,
        cache=["fake-kv"],
        path=None,
    )
    _patch_dkc(monkeypatch, lookup_result=loaded, requires_full=False)
    hm = HonestMetrics()
    req = _restore_request()
    Scheduler._maybe_disk_restore(_restore_self(hm), req, pflash_compressed=False)
    # Verified + installed → the only HIT path.
    assert req.cache_hit_type == "disk"
    assert req.cached_tokens == 3
    assert hm.snapshot()["kv_restore_result"] == {"hit": 1, "miss": 0}


def test_disk_restore_verify_fail_counts_miss(monkeypatch):
    """A candidate found but rejected (kv_dtype drift) is a miss, not a hit."""
    from vllm_mlx.scheduler import Scheduler

    loaded = SimpleNamespace(
        token_offset=3,
        metadata={"model_name": "test-model"},
        kv_dtype="int4",  # mismatches the run's bf16 → verify-fail reject
        requires_full_checkpoint=False,
        cache=["fake-kv"],
        path=None,
    )
    _patch_dkc(monkeypatch, lookup_result=loaded, requires_full=False)
    hm = HonestMetrics()
    req = _restore_request()
    Scheduler._maybe_disk_restore(_restore_self(hm), req, pflash_compressed=False)
    assert req.cache_hit_type == "miss"  # untouched by the rejected restore
    assert hm.snapshot()["kv_restore_result"] == {"hit": 0, "miss": 1}


def test_disk_restore_lookup_miss_counts_miss(monkeypatch):
    """A lookup that finds no checkpoint is a miss (attempt still happened)."""
    from vllm_mlx.scheduler import Scheduler

    _patch_dkc(monkeypatch, lookup_result=None)
    hm = HonestMetrics()
    req = _restore_request()
    Scheduler._maybe_disk_restore(_restore_self(hm), req, pflash_compressed=False)
    assert hm.snapshot()["kv_restore_result"] == {"hit": 0, "miss": 1}


def test_disk_restore_no_attempt_counts_neither(monkeypatch):
    """A request that never engaged the disk-restore path counts as neither."""
    from vllm_mlx.scheduler import Scheduler

    called = {"lookup": False}

    import vllm_mlx.runtime.disk_kv_checkpoint as dkc

    def _boom():
        called["lookup"] = True
        return SimpleNamespace(lookup=lambda ids: None)

    monkeypatch.setattr(dkc, "get_content_index", _boom)
    hm = HonestMetrics()
    req = _restore_request()
    # Feature disabled → gate returns before the lookup ever runs.
    Scheduler._maybe_disk_restore(
        _restore_self(hm, enabled=False), req, pflash_compressed=False
    )
    assert called["lookup"] is False  # no lookup == no attempt
    assert hm.snapshot()["kv_restore_result"] == {"hit": 0, "miss": 0}


def test_ttft_is_first_minus_arrival():
    """TTFT observation is first_token_time - arrival_time."""
    hm = HonestMetrics()
    hm.record_finish(
        arrival_time=1000.0,
        first_token_time=1000.30,
        t_last_token=1002.0,
        num_output_tokens=10,
    )
    snap = hm.snapshot()
    ttft = snap["ttft_seconds"]
    assert ttft["count"] == 1
    assert ttft["sum"] == pytest.approx(0.30)
    # 0.30 falls in the le="0.5" bucket and every bucket above it.
    buckets = dict(ttft["buckets"])
    assert buckets["0.25"] == 0
    assert buckets["0.5"] == 1
    assert buckets["+Inf"] == 1


def test_decode_uses_n_minus_1_gaps():
    """Decode rate is (n-1) inter-token gaps / decode window, not n / t."""
    hm = HonestMetrics()
    # 11 output tokens, first at t=1.0, last at t=3.0 → 10 gaps / 2.0s = 5 tok/s
    hm.record_finish(
        arrival_time=0.0,
        first_token_time=1.0,
        t_last_token=3.0,
        num_output_tokens=11,
    )
    snap = hm.snapshot()
    dec = snap["decode_tokens_per_second"]
    assert dec["count"] == 1
    assert dec["sum"] == pytest.approx(5.0)  # (11-1)/2.0, NOT 11/2.0 == 5.5


def test_full_cache_hit_single_output_token_no_decode_obs():
    """A finished request with <2 output tokens spans no decode gap."""
    hm = HonestMetrics()
    hm.record_finish(
        arrival_time=0.0,
        first_token_time=0.1,
        t_last_token=0.1,
        num_output_tokens=1,
    )
    snap = hm.snapshot()
    assert snap["decode_tokens_per_second"]["count"] == 0
    # TTFT still recorded (a first token WAS produced).
    assert snap["ttft_seconds"]["count"] == 1


def test_full_cache_hit_two_output_tokens_records_decode():
    """A full-cache-hit request DOES contribute to decode with >=2 tokens."""
    hm = HonestMetrics()
    hm.record_prefill(50, 50, "exact", [])
    hm.record_finish(
        arrival_time=0.0,
        first_token_time=0.05,
        t_last_token=0.15,
        num_output_tokens=2,
    )
    snap = hm.snapshot()
    dec = snap["decode_tokens_per_second"]
    assert dec["count"] == 1
    assert dec["sum"] == pytest.approx(1 / 0.10)  # (2-1)/0.10 == 10 tok/s


def test_decode_skipped_on_nonpositive_window():
    """A non-positive decode window is dropped, not divided by zero."""
    hm = HonestMetrics()
    hm.record_finish(0.0, 1.0, 1.0, num_output_tokens=5)  # window == 0
    assert hm.snapshot()["decode_tokens_per_second"]["count"] == 0


def test_histogram_buckets_are_cumulative_and_monotonic():
    """``_bucket`` cumulative counts never decrease with le."""
    h = FixedBucketHistogram(TTFT_BUCKET_BOUNDS)
    for v in (0.02, 0.2, 0.2, 3.0, 100.0):
        h.observe(v)
    snap = h.snapshot()
    cums = [c for _, c in snap["buckets"]]
    assert cums == sorted(cums)  # monotonic non-decreasing
    assert snap["count"] == 5
    assert snap["buckets"][-1] == ("+Inf", 5)


# ---------------------------------------------------------------------------
# Wire-level rendering tests
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics_client():
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    cfg.api_key = "test-secret"
    _reset_accumulator_for_tests()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()
    _reset_accumulator_for_tests()


def _fake_engine(stats: dict[str, Any]):
    return SimpleNamespace(get_stats=lambda: stats)


def _base_stats(**overrides: Any) -> dict[str, Any]:
    stats = {
        "num_waiting": 0,
        "num_running": 0,
        "num_requests_processed": 3,
        "total_prompt_tokens": 500,
        "total_completion_tokens": 90,
        "steps_executed": 10,
        "uptime_seconds": 1.0,
    }
    stats.update(overrides)
    return stats


def _honest_block() -> dict[str, Any]:
    hm = HonestMetrics()
    hm.record_prefill(200, 0, "miss", list(range(200)))  # cold
    hm.record_prefill(300, 120, "prefix", list(range(180)))  # extend, memory
    hm.record_prefill(128, 128, "disk", [])  # exact, disk
    hm.record_prefix_match("miss")
    hm.record_prefix_match("prefix")
    hm.record_prefix_match("exact")
    hm.record_disk_restore(hit=True)
    hm.record_disk_restore(hit=True)
    hm.record_disk_restore(hit=False)
    hm.record_finish(0.0, 0.30, 2.30, 11)  # ttft .3, decode (11-1)/2 = 5
    return hm.snapshot()


def _sample_value(body: str, name_with_labels: str) -> float:
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(name_with_labels + " "):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"sample not found: {name_with_labels}\n{body}")


def test_route_renders_offered_computed_reused(metrics_client):
    metrics_client.cfg.engine = _fake_engine(
        _base_stats(honest_metrics=_honest_block())
    )
    body = metrics_client.client.get("/metrics").text

    # offered = 200 + 300 + 128 = 628
    assert _sample_value(body, "qmlx_prompt_tokens_offered_total") == 628
    # computed = 200 + 180 + 0 = 380
    assert _sample_value(body, "qmlx_prompt_tokens_computed_total") == 380
    # reused: memory = 120, disk = 128
    assert (
        _sample_value(body, 'qmlx_prompt_tokens_reused_total{source="memory"}')
        == 120
    )
    assert (
        _sample_value(body, 'qmlx_prompt_tokens_reused_total{source="disk"}')
        == 128
    )
    # offered - computed == total reused
    assert 628 - 380 == 120 + 128


def test_route_renders_prefill_kind_and_match(metrics_client):
    metrics_client.cfg.engine = _fake_engine(
        _base_stats(honest_metrics=_honest_block())
    )
    body = metrics_client.client.get("/metrics").text
    assert _sample_value(body, 'qmlx_prefill_kind_total{kind="cold"}') == 1
    assert _sample_value(body, 'qmlx_prefill_kind_total{kind="extend"}') == 1
    assert _sample_value(body, 'qmlx_prefill_kind_total{kind="exact"}') == 1
    assert _sample_value(body, 'qmlx_prefix_cache_match_total{type="miss"}') == 1
    assert _sample_value(body, 'qmlx_prefix_cache_match_total{type="prefix"}') == 1
    assert _sample_value(body, 'qmlx_prefix_cache_match_total{type="exact"}') == 1
    # unused canonical types still present at 0 (flat-line, not "no data")
    assert _sample_value(body, 'qmlx_prefix_cache_match_total{type="lcp"}') == 0


def test_route_renders_kv_restore_hit_miss(metrics_client):
    metrics_client.cfg.engine = _fake_engine(
        _base_stats(honest_metrics=_honest_block())
    )
    body = metrics_client.client.get("/metrics").text
    assert (
        "# HELP qmlx_kv_restore_total Disk KV restore attempts by result" in body
    )
    assert _sample_value(body, 'qmlx_kv_restore_total{result="hit"}') == 2
    assert _sample_value(body, 'qmlx_kv_restore_total{result="miss"}') == 1
    # Disk hit rate = 2 / (2 + 1) computable in PromQL from these two series.


def test_kv_restore_counter_sticky(metrics_client):
    """qmlx_kv_restore_total never decreases across a source reset."""
    metrics_client.cfg.engine = _fake_engine(
        _base_stats(honest_metrics=_honest_block())
    )
    body1 = metrics_client.client.get("/metrics").text
    assert _sample_value(body1, 'qmlx_kv_restore_total{result="hit"}') == 2

    hm_lo = HonestMetrics()
    hm_lo.record_disk_restore(hit=True)  # raw hit drops to 1
    metrics_client.cfg.engine = _fake_engine(
        _base_stats(honest_metrics=hm_lo.snapshot())
    )
    body2 = metrics_client.client.get("/metrics").text
    hit2 = _sample_value(body2, 'qmlx_kv_restore_total{result="hit"}')
    assert hit2 >= 2  # folded baseline(2) + raw(1) == 3, never below 2
    assert hit2 == 3


def test_route_renders_ttft_and_decode_histograms(metrics_client):
    metrics_client.cfg.engine = _fake_engine(
        _base_stats(honest_metrics=_honest_block())
    )
    body = metrics_client.client.get("/metrics").text
    assert "# TYPE qmlx_ttft_seconds histogram" in body
    assert "# TYPE qmlx_decode_tokens_per_second histogram" in body
    assert _sample_value(body, "qmlx_ttft_seconds_count") == 1
    assert _sample_value(body, "qmlx_ttft_seconds_sum") == pytest.approx(0.30)
    assert _sample_value(body, "qmlx_decode_tokens_per_second_count") == 1
    assert _sample_value(
        body, "qmlx_decode_tokens_per_second_sum"
    ) == pytest.approx(5.0)
    # +Inf bucket equals count.
    assert (
        _sample_value(body, 'qmlx_decode_tokens_per_second_bucket{le="+Inf"}') == 1
    )


def test_route_renders_restore_reject_reasons(metrics_client):
    stats = _base_stats(
        honest_metrics=_honest_block(),
        kv_checkpoint={
            "writes": 5,
            "loads": 2,
            "bytes": 100,
            "evictions": 0,
            "hook_errors": 0,
            "restore_rejects": {
                "memory_headroom": 3,
                "kv_dtype_mismatch": 1,
                "exception": 0,
            },
        },
    )
    metrics_client.cfg.engine = _fake_engine(stats)
    body = metrics_client.client.get("/metrics").text
    assert "# TYPE qmlx_kv_restore_reject_total counter" in body
    assert (
        _sample_value(
            body, 'qmlx_kv_restore_reject_total{reason="memory_headroom"}'
        )
        == 3
    )
    assert (
        _sample_value(
            body, 'qmlx_kv_restore_reject_total{reason="kv_dtype_mismatch"}'
        )
        == 1
    )


def test_reuse_counters_survive_reset_sticky(metrics_client):
    """Sticky accumulator: a snapshot that goes backwards never decrements.

    The token-reuse counters route through the same
    ``_StickyCounterAccumulator`` as the cache series, so a reset of the
    underlying source (values dropping) folds into a baseline instead of
    the Prometheus counter going backwards.
    """
    hm_hi = _honest_block()
    metrics_client.cfg.engine = _fake_engine(_base_stats(honest_metrics=hm_hi))
    body1 = metrics_client.client.get("/metrics").text
    assert _sample_value(body1, "qmlx_prompt_tokens_offered_total") == 628

    # Simulate a process-internal reset: the raw source now reports LOWER
    # values. The exposed counter must not decrease.
    hm_lo = HonestMetrics()
    hm_lo.record_prefill(10, 0, "miss", list(range(10)))  # offered now only 10
    metrics_client.cfg.engine = _fake_engine(
        _base_stats(honest_metrics=hm_lo.snapshot())
    )
    body2 = metrics_client.client.get("/metrics").text
    offered2 = _sample_value(body2, "qmlx_prompt_tokens_offered_total")
    # baseline(628) + raw(10) == 638, and crucially >= 628 (never went down)
    assert offered2 >= 628
    assert offered2 == 638


def test_no_series_equals_prompt_plus_gen_over_wall(metrics_client):
    """Guardrail: no emitted sample equals (prompt + generated) / wall.

    That amortized number is the lie the honest-metrics pass exists to
    kill. Construct a request whose amortized rate is a distinctive value
    and assert it appears in NO sample line.
    """
    # prompt=1000, gen=100, wall (arrival->last) = 5.0s.
    # Amortized lie = (1000 + 100) / 5.0 = 220.0.
    # Honest decode = (100-1)/(decode window). Keep the decode window such
    # that honest decode != 220 and != the amortized value.
    hm = HonestMetrics()
    hm.record_prefill(1000, 0, "miss", list(range(1000)))
    hm.record_finish(
        arrival_time=0.0,
        first_token_time=1.0,  # 1s prefill+ttft
        t_last_token=5.0,  # last token at 5s → decode window 4s
        num_output_tokens=100,
    )
    forbidden = (1000 + 100) / 5.0  # 220.0
    metrics_client.cfg.engine = _fake_engine(_base_stats(honest_metrics=hm.snapshot()))
    body = metrics_client.client.get("/metrics").text

    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^\S+ ([-\d.eE+]+)$", line)
        if not m:
            continue
        val = float(m.group(1))
        assert val != pytest.approx(forbidden), f"amortized lie leaked: {line}"

    # Sanity: the honest decode sum IS present and equals (100-1)/4.0.
    assert _sample_value(
        body, "qmlx_decode_tokens_per_second_sum"
    ) == pytest.approx(99 / 4.0)
