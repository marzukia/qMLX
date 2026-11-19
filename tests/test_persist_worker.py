"""Unit tests for the off-main checkpoint persist worker (q9b fix).

Stdlib-only: the module under test has no mlx dependency, so these run
on the CPU-only CI venv.
"""

import threading
import time

from vllm_mlx.persist_worker import PersistWorker, gated_work


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


def test_gated_work_waits_for_gate_then_runs() -> None:
    # The q9b completion-write deferral: the task must NOT run before
    # the gate is set (teardown done), and must run after.
    w = PersistWorker(max_queue=4)
    gate = threading.Event()
    ran = []
    assert w.submit(gated_work(lambda: ran.append(1), gate, timeout=5.0))
    time.sleep(0.4)  # give the worker ample time if the gate were bypassed
    assert ran == []
    gate.set()
    w.shutdown(timeout=3.0)
    assert ran == [1]


def test_gated_work_none_gate_runs_immediately() -> None:
    w = PersistWorker(max_queue=4)
    ran = []
    assert w.submit(gated_work(lambda: ran.append(1), None, timeout=5.0))
    w.shutdown(timeout=3.0)
    assert ran == [1]


def test_gated_work_proceeds_after_timeout_without_deadlock() -> None:
    # A gate that is never set must not hang the worker: after the
    # bounded timeout the task proceeds anyway (best-effort write).
    w = PersistWorker(max_queue=4)
    never = threading.Event()
    ran = []
    assert w.submit(gated_work(lambda: ran.append(1), never, timeout=0.4))
    time.sleep(0.2)  # still inside the wait window
    assert ran == []
    w.shutdown(timeout=4.0)  # must return (no deadlock)
    assert ran == [1]


def test_gated_work_preserves_queue_order() -> None:
    # The completion task sits behind the prompt task; deferring the
    # latter must not reorder the two.
    w = PersistWorker(max_queue=4)
    gate = threading.Event()
    order = []
    assert w.submit(lambda: order.append("prompt"))
    assert w.submit(gated_work(lambda: order.append("completion"), gate, timeout=5.0))
    time.sleep(0.3)
    gate.set()
    w.shutdown(timeout=3.0)
    assert order == ["prompt", "completion"]
