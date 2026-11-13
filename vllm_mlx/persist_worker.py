"""Dedicated worker thread for disk checkpoint persistence (q9b fix).

Disk checkpoint saves used to run on the main thread between a request's
prefill async-eval and its first decode async-eval. The save's
force-eval of the extracted live-state views plus the disk write freed
an MLX event object inside that window, whose backing memory was then
reused by the persist allocations, corrupting the first decode eval
(q9b UAF). This module moves the heavy half of the save (force-eval,
numpy/quant helpers, raw disk write) off the main thread: the store
sites only extract the cache views (cheap, proven safe) and hand them
to this worker.

Stdlib only. The worker holds strong refs to the handed-over views
(until the write completes, then releases them). Semantics:

- ``submit(work)``: enqueue a zero-arg callable; returns ``False`` when
  the bounded queue is full. The caller then SKIPS that checkpoint
  (checkpointing is resilience, not correctness) and must not block.
- Tasks run in order; a failing task is logged and swallowed (never
  kills the worker).
- ``shutdown(timeout)``: bounded drain; work still pending on timeout is
  logged as lost. The thread is a daemon, so process exit can never
  hang on this worker.
"""

import logging
import queue
import threading

logger = logging.getLogger(__name__)


class PersistWorker:
    """Single daemon thread performing checkpoint work off the main thread."""

    def __init__(self, max_queue: int = 8, name: str = "qmlx-persist"):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, work) -> bool:
        """Enqueue a task; returns False (skip) when the queue is full."""
        if self._closed:
            return False
        try:
            self._queue.put_nowait(work)
            return True
        except queue.Full:
            return False

    def pending(self) -> int:
        """Number of tasks waiting to run."""
        return self._queue.qsize()

    def shutdown(self, timeout: float = 2.0) -> int:
        """Signal stop and drain for at most ``timeout`` seconds.

        Returns the task count still pending on timeout (logged lost).
        """
        if self._closed:
            return 0
        self._closed = True
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            # Queue still full of real work; the join below times out
            # and the remainder is counted as lost.
            pass
        self._thread.join(timeout=max(timeout, 0.0))
        lost = self._queue.qsize()
        if lost:
            logger.warning(
                "[persist_worker] shutdown: %d pending checkpoint write(s) lost",
                lost,
            )
        return lost

    def _run(self) -> None:
        while True:
            try:
                work = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._closed:
                    return
                continue
            if work is None:
                return
            try:
                work()
            except Exception as exc:  # best-effort work: never kill the worker
                logger.debug("[persist_worker] task failed: %s", exc)
            finally:
                self._queue.task_done()
