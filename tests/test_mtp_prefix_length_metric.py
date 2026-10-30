# SPDX-License-Identifier: Apache-2.0
"""MTP prefix-length metric — per-round accepted-count distribution (#107).

``qmlx_spec_decode_accept_ratio`` averages acceptance across rounds,
so the operator cannot tell whether K=5 was *right*: mass at 0-1
(drafts rejected up front) says the EV controller should lower K;
mass at the top bucket says there is headroom to raise it. These
tests pin the metric's contract:

* ``record_prefix`` clamps the accepted count to ``[0, 5]``.
* The snapshot is a 6-bucket distribution (buckets 0..5; the K=5 chain
  records exactly one bucket per round) taken under the counter's
  existing lock — concurrent ``record_prefix`` + ``snapshot`` is safe.
* ``mtp_generate_step`` records the accepted count at the same point
  it records the round wall time (source-level wiring check).
* The Prometheus renderer emits all 5+1 buckets with the
  ``prefix_len`` label and zeroes on cold start.

No-MLX: the accept-counter tests run without MLX (the module is pure
Python); the metrics tests install the minimal ``mlx`` / ``mlx_lm``
stubs shown at the top of this file (Linux CI has no MLX; the
renderer module transitively imports it through ``vllm_mlx.config``).
``sys.modules`` is save/restored so a real MLX install is untouched.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

# ----------------------------------------------------------------------
# Minimal mlx / mlx_lm stubs (no-mlx host — Linux CI has no MLX; the
# metrics renderer module transitively imports ``mlx.core`` / ``mlx_lm``
# through ``vllm_mlx.config``). The accept-counter tests need none of
# this; only the metrics tests install the stubs, and they save / restore
# ``sys.modules`` so a real MLX install (Apple-Silicon CI) is untouched.
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


# ---------------------------------------------------------------------------
# 1. Accept-counter prefix distribution
# ---------------------------------------------------------------------------


def test_record_prefix_clamps_to_0_to_5():
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    counter.record_prefix(0)
    counter.record_prefix(3)
    counter.record_prefix(5)
    # Clamped: >5 folds into the top bucket, <0 into bucket 0.
    counter.record_prefix(7)
    counter.record_prefix(-2)
    dist = counter.prefix_lengths
    assert len(dist) == 6
    assert sum(dist) == 5
    assert dist[0] == 2  # 0 and -2
    assert dist[3] == 1
    assert dist[5] == 2  # 5 and 7
    assert dist[1] == 0
    assert dist[4] == 0


def test_snapshot_is_six_bucket_tuple_and_reset_clears():
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    counter.record_prefix(4)
    snap = counter.snapshot()
    assert isinstance(snap.prefix_lengths, tuple)
    assert len(snap.prefix_lengths) == 6
    assert snap.prefix_lengths[4] == 1
    assert sum(snap.prefix_lengths) == 1
    counter.reset()
    assert sum(counter.snapshot().prefix_lengths) == 0


def test_snapshot_is_safe_under_concurrent_writes():
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    n_writers = 8
    n_per_writer = 250

    def writer(offset: int) -> None:
        for i in range(n_per_writer):
            counter.record_prefix((offset + i) % 6)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_writers)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        # Interleave reads; the distribution must stay causally
        # consistent (6 buckets, no negative or missing entries).
        snap = counter.snapshot()
        assert len(snap.prefix_lengths) == 6
        assert all(v >= 0 for v in snap.prefix_lengths)
    for t in threads:
        t.join()
    total = sum(counter.snapshot().prefix_lengths)
    assert total == n_writers * n_per_writer


def test_reset_global_counter_for_tests_clears_prefix():
    from vllm_mlx.spec_decode.mtp.accept_counter import (
        get_global_counter,
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()
    counter = get_global_counter()
    counter.record_prefix(2)
    assert sum(counter.prefix_lengths) == 1
    reset_global_counter_for_tests()
    assert sum(get_global_counter().prefix_lengths) == 0


# ---------------------------------------------------------------------------
# 2. Generator wiring (source-level — a full mtp_generate_step run
#    needs real MTP weights; the wiring is a one-line call at the
#    existing record point)
# ---------------------------------------------------------------------------


def test_generator_records_prefix_at_the_round_record_point():
    from pathlib import Path

    src = (
        Path("vllm_mlx/spec_decode/mtp/generator.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    rec_round = [
        i for i, line in enumerate(src) if "_record_round(k_len, round_wall_ms" in line
    ]
    assert rec_round, "mtp_generate_step lost its _record_round call"
    rec_prefix = [
        i
        for i, line in enumerate(src)
        if "accept_counter.record_prefix(accepted_count)" in line
    ]
    assert rec_prefix, "mtp_generate_step never records the prefix length"
    # Recorded at the same point as the round record (within a few
    # lines after it), not somewhere detached.
    assert min(abs(pr - rr) for pr in rec_prefix for rr in rec_round) <= 5


# ---------------------------------------------------------------------------
# 3. Prometheus rendering
# ---------------------------------------------------------------------------


def test_metrics_renders_all_six_prefix_buckets_and_cold_start_zeroes(
    _mlx_stubs,
):
    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from vllm_mlx.spec_decode.mtp.accept_counter import (
        get_global_counter,
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()
    counter = get_global_counter()
    # 2 rounds with 0 accepted, 1 with 3, 1 with 5, 1 with 1.
    for accepted in (0, 0, 3, 5, 1):
        counter.record_prefix(accepted)

    class _Cfg:
        model_alias = "qwen3.5-9b-4bit"

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    # All buckets present, labelled, values matching the distribution.
    expected = {0: 2, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}
    for bucket, count in expected.items():
        needle = (
            f'qmlx_spec_decode_prefix_len{{family="qwen3.5-9b-4bit",'
            f'method="mtp",prefix_len="{bucket}"}} {count}'
        )
        assert needle in body, f"missing/incorrect bucket {bucket}: {needle}"
    # The existing series are still rendered (no regression).
    assert "qmlx_spec_decode_attempts_total" in body
    assert "qmlx_spec_decode_accept_ratio" in body


def test_metrics_prefix_buckets_zero_at_cold_start(_mlx_stubs):
    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from vllm_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()

    class _Cfg:
        model_alias = "qwen3.5-9b-4bit"

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    for bucket in range(6):
        needle = (
            f'qmlx_spec_decode_prefix_len{{family="qwen3.5-9b-4bit",'
            f'method="mtp",prefix_len="{bucket}"}} 0'
        )
        assert needle in body, f"cold-start bucket {bucket} missing or non-zero"
