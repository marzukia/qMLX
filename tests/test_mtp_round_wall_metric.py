# SPDX-License-Identifier: Apache-2.0
"""Per-K round wall-time metric — ``qmlx_spec_decode_round_wall_ms{k=N}``
(#109).

The auto-K controller's ``CostModel`` already tracks a per-depth EWMA
of verify-round wall time (measured with it: k=0 53.2 ms, k=1 55.8,
k=2 62.8, k=3 71.3 over ~130 mixed rounds on 27B oQ4e — the round
machinery is near-bandwidth-ideal; the gap to target is temp-1
acceptance, not round cost). Surface that learning as a live gauge
without any new per-round timing instrumentation.

Pinned here:

1. ``sum_cost_model()`` is registry-lock-safe: two controllers with
   interleaved ``CostModel.observe`` updates from two threads, plus a
   concurrent renderer loop — the merge matches a hand-computed
   replay (per-depth MAX: round wall times are per-model wall
   clocks, not additive, so the slowest model dominates).
2. The metric is absent when MTP is off (no rounds observed -> empty
   merge, no zero-emission rows) and k=0 is present as soon as any
   rounds exist.
3. No new per-round allocation on the hot path: the merge happens at
   render time only (``sum_cost_model`` is called from the metrics
   renderer, never from ``observe`` / ``record``).

No-MLX: the accept/registry modules are pure Python; the metrics tests
install the same minimal ``mlx`` / ``mlx_lm`` stubs as
``test_mtp_prefix_length_metric.py`` (Linux CI has no MLX; the
renderer module transitively imports it through ``vllm_mlx.config``).
``sys.modules`` is save/restored so a real MLX install is untouched.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

# ----------------------------------------------------------------------
# mlx / mlx_lm stubs — same pattern as test_mtp_prefix_length_metric.py
# ----------------------------------------------------------------------


class _AnyBase:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _AnyBase()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _AnyBase()


@pytest.fixture()
def _mlx_stubs(monkeypatch):
    saved = {k: v for k, v in sys.modules.items() if k == "mlx" or k.startswith("mlx.")}

    def stub(name):
        m = _StubModule(name)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, name, m)
        return m

    mx = stub("mlx")
    mx.core = stub("mlx.core")
    mx.nn = stub("mlx.nn")
    mlm = stub("mlx_lm")
    for sub in (
        "generate",
        "sample_utils",
        "tokenizer_utils",
        "utils",
        "models",
        "quantize",
        "convert",
        "models.base",
        "models.cache",
        "models.gated_delta",
        "models.switch_layers",
    ):
        m = stub(f"mlx_lm.{sub}")
        setattr(mlm, sub.split(".")[-1], m)
    yield
    for k in [k for k in sys.modules if k == "mlx" or k.startswith("mlx.")]:
        del sys.modules[k]
    sys.modules.update(saved)


@pytest.fixture()
def _clean_registry():
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import reset_controllers

    reset_controllers()
    yield
    reset_controllers()


# ----------------------------------------------------------------------
# 1. sum_cost_model — registry-lock-safe merge, hand-computed
# ---------------------------------------------------------------------------


def _ewma_replay(samples: list[float], alpha: float, clamp: float) -> float | None:
    """Independent re-derivation of the clamped-EWMA update (same
    operation order as ``CostModel.observe`` -> bit-identical floats)."""
    prev: float | None = None
    for new in samples:
        if prev is None:
            prev = new
        else:
            limit = clamp * prev
            innovation = new - prev
            if innovation > limit:
                innovation = limit
            elif innovation < -limit:
                innovation = -limit
            prev = prev + alpha * innovation
    return prev


def test_sum_cost_model_empty_when_no_controllers(_clean_registry):
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import sum_cost_model

    assert sum_cost_model() == {}


def test_sum_cost_model_merges_max_per_depth_and_is_lock_safe(
    _clean_registry,
):
    """Two controllers, interleaved observes from two threads, plus a
    third thread hammering sum_cost_model() concurrently — the final
    merge must equal the hand-computed per-depth MAX of each
    controller's replayed EWMA, and the concurrent reads must never
    raise (lock integrity)."""
    from vllm_mlx.spec_decode.mtp import draft_k_controller_v2 as dk
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        get_or_create_controller,
        sum_cost_model,
    )

    ctrl_a = get_or_create_controller("model-a")
    ctrl_b = get_or_create_controller("model-b")
    assert ctrl_a is not ctrl_b

    alpha, clamp = dk.COST_EWMA_ALPHA, dk.COST_CLAMP_FRACTION
    samples_a = [10.0 + (i % 7) * 0.5 for i in range(400)]
    samples_b = [20.0 + (i % 5) * 1.1 for i in range(400)]

    errors: list[BaseException] = []

    def worker(ctrl, samples):
        try:
            for i, wall in enumerate(samples):
                ctrl.cost.observe(1 + (i % 2) * 2, wall)
        except BaseException as exc:  # noqa: B036
            errors.append(exc)

    def renderer():
        try:
            for _ in range(200):
                merged = sum_cost_model()
                assert isinstance(merged, dict)
                for v in merged.values():
                    assert v > 0.0
        except BaseException as exc:  # noqa: B036
            errors.append(exc)

    t_a = threading.Thread(target=worker, args=(ctrl_a, samples_a))
    t_b = threading.Thread(target=worker, args=(ctrl_b, samples_b))
    t_r = threading.Thread(target=renderer)
    t_a.start()
    t_b.start()
    t_r.start()
    t_a.join()
    t_b.join()
    t_r.join()
    assert not errors, f"concurrent worker/renderer raised: {errors}"

    # Hand-computed merge: replay each controller's exact per-depth
    # update sequence (its own thread is the only writer ->
    # deterministic), then MAX per depth across controllers.
    expected: dict[int, float] = {}
    for depth in (1, 3):
        a_replay = _ewma_replay(
            [w for i, w in enumerate(samples_a) if 1 + (i % 2) * 2 == depth],
            alpha,
            clamp,
        )
        b_replay = _ewma_replay(
            [w for i, w in enumerate(samples_b) if 1 + (i % 2) * 2 == depth],
            alpha,
            clamp,
        )
        expected[depth] = max(a_replay, b_replay)

    got = sum_cost_model()
    assert got == expected, f"merge mismatch:\n got={got}\n exp={expected}"


def test_sum_cost_model_max_policy_with_disjoint_depths(_clean_registry):
    """Overlap -> max per depth; disjoint depths -> union."""
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        get_or_create_controller,
        sum_cost_model,
    )

    ctrl_a = get_or_create_controller("model-a")
    ctrl_b = get_or_create_controller("model-b")
    ctrl_a.cost.observe(0, 50.0)
    ctrl_a.cost.observe(1, 60.0)
    ctrl_b.cost.observe(0, 55.0)  # overlap at k=0 -> max 55.0
    ctrl_b.cost.observe(2, 70.0)  # disjoint k=2 -> union

    got = sum_cost_model()
    assert got == {0: 55.0, 1: 60.0, 2: 70.0}


# ----------------------------------------------------------------------
# 2. Renderer — absent when MTP off, k=0 as soon as rounds exist
# ---------------------------------------------------------------------------


def _render() -> str:
    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters

    class _Cfg:
        model_alias = "qwen3.5-9b-4bit"

    return "\n".join(_render_spec_decode_mtp_counters(_Cfg()))


def test_metric_absent_when_mtp_off(_mlx_stubs, _clean_registry):
    """No controller has sampled any depth (MTP off / cold start):
    the round_wall series is absent — no zero-emission rows (the
    #302 counters still render their zero rows; only #109 is absent)."""
    body = _render()
    assert "qmlx_spec_decode_round_wall_ms" not in body
    # The existing series are still rendered (no regression).
    assert "qmlx_spec_decode_attempts_total" in body
    assert "qmlx_spec_decode_park_total" in body


def test_k0_present_as_soon_as_any_rounds_exist(_mlx_stubs, _clean_registry):
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        get_or_create_controller,
    )

    # Any round (k=0 plain-decode reference) lands a gauge.
    get_or_create_controller("model-a").cost.observe(0, 53.2)
    body = _render()
    needle = (
        'qmlx_spec_decode_round_wall_ms{family="qwen3.5-9b-4bit",'
        'method="mtp",k="0"} 53.2'
    )
    assert needle in body
    # No other depth sampled -> no k=1/k=2 rows.
    assert 'k="1"}' not in body
    assert 'k="2"}' not in body

    # A deeper round appears as its own labelled row.
    get_or_create_controller("model-a").cost.observe(2, 62.8)
    body = _render()
    assert (
        'qmlx_spec_decode_round_wall_ms{family="qwen3.5-9b-4bit",'
        'method="mtp",k="2"} 62.8'
    ) in body
    assert "# TYPE qmlx_spec_decode_round_wall_ms gauge" in body


def test_multi_model_merge_renders_max_per_depth(_mlx_stubs, _clean_registry):
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        get_or_create_controller,
    )

    get_or_create_controller("model-a").cost.observe(1, 55.0)
    get_or_create_controller("model-b").cost.observe(1, 58.0)
    body = _render()
    needle = (
        'qmlx_spec_decode_round_wall_ms{family="qwen3.5-9b-4bit",'
        'method="mtp",k="1"} 58.0'
    )
    assert needle in body  # max(55, 58) — slowest model dominates


# ----------------------------------------------------------------------
# 3. No hot-path merge — structural check
# ---------------------------------------------------------------------------
def test_merge_happens_at_render_time_only():
    from pathlib import Path

    ctrl_src = (
        Path("vllm_mlx/spec_decode/mtp/draft_k_controller_v2.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    # The controller module references sum_cost_model() ONLY in its
    # own def line (the merge takes the registry lock — calling it
    # from observe()/record() would allocate a fresh dict per round
    # and nest the lock). No call site inside the module.
    call_lines = [
        line
        for line in ctrl_src
        if "sum_cost_model()" in line and "def sum_cost_model" not in line
    ]
    assert not call_lines, f"hot-path call site in controller: {call_lines}"

    metrics_src = (
        Path("vllm_mlx/routes/metrics.py").read_text(encoding="utf-8").splitlines()
    )
    assert any("sum_cost_model" in line for line in metrics_src), (
        "metrics renderer lost its sum_cost_model import/call"
    )
    assert any("sum_cost_model()" in line for line in metrics_src), (
        "metrics renderer never calls sum_cost_model()"
    )
