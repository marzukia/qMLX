"""Unit tests for the off-main checkpoint persist worker (q9b fix).

Stdlib-only: the module under test has no mlx dependency, so these run
on the CPU-only CI venv.
"""

import time

from vllm_mlx.persist_worker import PersistWorker


def test_submit_runs_work_in_order() -> None:
    w = PersistWorker(max_queue=4)
    seen = []
    for i in range(4):
        assert w.submit(lambda i=i: seen.append(i))
    w.shutdown(timeout=2.0)
    assert seen == [0, 1, 2, 3]


def test_overflow_returns_false_and_does_not_block() -> None:
    w = PersistWorker(max_queue=1)
    w.submit(lambda: time.sleep(0.5))  # occupy the single slot
    t0 = time.monotonic()
    assert not w.submit(lambda: None)
    assert time.monotonic() - t0 < 0.2  # no blocking
    w.shutdown(timeout=3.0)


def test_failing_task_does_not_kill_worker() -> None:
    w = PersistWorker(max_queue=2)
    w.submit(lambda: 1 / 0)
    time.sleep(0.3)  # let the failing task drain first
    assert w.submit(lambda: None)
    w.shutdown(timeout=2.0)  # worker thread must still be joinable


def test_shutdown_marks_pending_lost() -> None:
    w = PersistWorker(max_queue=4)
    for _ in range(4):
        w.submit(lambda: None)
    w.shutdown(timeout=0.0)  # no drain: whatever is left is lost
    assert w.shutdown(timeout=0.0) == 0  # idempotent


def test_submit_after_shutdown_returns_false() -> None:
    w = PersistWorker(max_queue=1)
    w.shutdown(timeout=2.0)
    assert not w.submit(lambda: None)
