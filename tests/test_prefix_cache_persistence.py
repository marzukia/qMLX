# SPDX-License-Identifier: Apache-2.0
"""
Reproduction tests for prefix-cache disk-persistence corruption bugs.

Failure modes documented here (see analysis 2026-05-03):

A. Stale ``index.json`` + freshly-overwritten ``entry_i.*`` files —
   if a server is killed mid-shutdown after rewriting some entry files
   but before ``index.json`` is rewritten, the next start loads using
   the stale ``num_tokens`` field. ``arr.fromfile(f, num_tokens_old)``
   silently truncates the new tokens.bin, producing an entry whose
   ``tokens_key`` length disagrees with ``cache.offset``. Subsequent
   fetches return that mismatched cache to the scheduler, which
   appends new tokens at the wrong position → garbage attention →
   token-id-0 collapse (``!!!!!`` in user output).

B. Orphan files from a previous save are not removed when the next
   save writes fewer entries. They sit on disk indefinitely; the next
   crash that interrupts ``save_to_disk`` mid-rewrite turns them into
   the inconsistency described in (A).

C. ``mx.save_safetensors`` is called directly on the target path
   (no ``.tmp`` + rename), so a SIGKILL during a single-entry write
   leaves a half-written safetensors. ``mx.load`` will usually raise
   on it (caught and dropped silently), but combined with (A) it can
   amplify the inconsistency.

D. ``mx.load`` is lazy — it parses the header and returns array
   handles without materializing data. A safetensors with a valid
   header but truncated body passes ``load_from_disk`` silently and
   is registered as a usable cache entry. The corruption only
   surfaces at the first attention call, often inside a worker thread
   where the RuntimeError can be swallowed.

These tests use real ``mlx_lm`` ``KVCache`` objects with very small
tensors (1×4×N×8 fp16) so they run fast (<1s each).
"""

from __future__ import annotations

import pytest

# These tests only exercise the runtime/cache.py shutdown-save plumbing and the
# fsync / should-abort helpers in memory_cache. The in-memory prefix cache and
# its disk round-trip tests were removed with the class in the SSD-first
# refactor (issue #16); the disk checkpoint tier is covered by
# tests/test_disk_kv_checkpoint.py.
pytest.importorskip("mlx.core")


def test_shutdown_save_prefix_cache_runs_off_event_loop(tmp_path, monkeypatch):
    """Regression for the lifespan shutdown bug, pinned at the production
    callsite.

    Drives ``server._shutdown_save_prefix_cache`` directly — NOT a
    test-local ``asyncio.to_thread`` wrapper around the save function.
    Codex flagged PR #667 round 1 because the prior shape wrapped
    ``to_thread`` test-side, so a regression that dropped the wrapper
    from the lifespan helper would still pass. This version exercises
    the helper that's literally what ``lifespan`` awaits — if anyone
    replaces the ``await asyncio.to_thread(...)`` line in
    ``_shutdown_save_prefix_cache`` with a direct call, the slow fake
    engine below blocks the loop and the ticker count drops to 1.

    Shape: pretend the engine takes ~600ms to flush. Drive the helper
    AND a 50ms ticker concurrently. Assert the ticker advances
    multiple times — i.e. the loop wasn't blocked.
    """
    import asyncio
    import time as _time

    from vllm_mlx import config as _config_mod
    from vllm_mlx import server as _server_mod
    from vllm_mlx.runtime import cache as _cache_mod

    class _SlowEngine:
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            # Block for ~600ms on the worker thread — production wrap
            # is asyncio.to_thread so the loop stays responsive.
            _time.sleep(0.6)
            return True

    class _FakeCfg:
        engine = _SlowEngine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())
    # ``_shutdown_save_prefix_cache`` checks ``_engine is not None``
    # AND ``hasattr(_engine, "save_cache_to_disk")`` before delegating
    # to the runtime save. Substitute a stub that satisfies both so
    # the helper actually runs.
    monkeypatch.setattr(_server_mod, "_engine", _SlowEngine())

    async def _drive():
        ticks: list[float] = []
        t0 = _time.monotonic()

        async def _ticker():
            while _time.monotonic() - t0 < 0.6:
                ticks.append(_time.monotonic() - t0)
                await asyncio.sleep(0.05)

        # Drive the production lifespan helper as-is. Don't wrap it
        # in asyncio.to_thread out here — that would re-introduce the
        # original bug the test exists to catch.
        await asyncio.gather(
            _server_mod._shutdown_save_prefix_cache(),
            _ticker(),
        )
        return ticks

    ticks = asyncio.run(_drive())
    assert len(ticks) >= 5, (
        f"event loop was blocked during cache flush — only saw {len(ticks)} "
        f"ticks in 600ms (expected ≥5). Did _shutdown_save_prefix_cache "
        f"lose its asyncio.to_thread wrap?"
    )


def test_shutdown_save_prefix_cache_no_op_when_engine_missing(monkeypatch):
    """Companion guard: when ``server._engine`` is None (no model loaded
    yet at shutdown, or already torn down), the helper returns silently
    instead of blowing up with AttributeError. This is the production
    failure mode for ``qmlx serve --help`` lifecycle interruption
    where shutdown lands before startup finished.
    """
    import asyncio

    from vllm_mlx import server as _server_mod

    monkeypatch.setattr(_server_mod, "_engine", None)
    asyncio.run(_server_mod._shutdown_save_prefix_cache())  # must not raise


def test_save_prefix_cache_to_disk_respects_budget(tmp_path, monkeypatch):
    """``save_prefix_cache_to_disk`` must build a deadline predicate from
    the budget arg and forward it as ``should_abort`` to the engine. With
    a 0.1s budget and a save that takes ≥0.2s to even get its first
    callback, the predicate is True on first call.
    """
    import time as _time

    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    captured = {"pred": None}

    class _Engine:
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            captured["pred"] = should_abort
            # Sleep past the deadline so the predicate is guaranteed True.
            _time.sleep(0.25)
            return False

    class _FakeCfg:
        engine = _Engine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    _cache_mod.save_prefix_cache_to_disk(budget_sec=0.1)
    pred = captured["pred"]
    assert pred is not None, (
        "save_prefix_cache_to_disk must pass a should_abort predicate"
    )
    assert pred() is True, "deadline-backed predicate should be tripped after sleep"


def test_save_prefix_cache_to_disk_predicate_is_forward_looking(tmp_path, monkeypatch):
    """Codex PR #667 round 1 BLOCKING-2 regression.

    The predicate must accept a ``predicted_sec`` argument and return
    True when starting that operation would push past the budget — not
    just when wall-clock has already crossed the deadline. Without
    this shape, a single 300 MB ``save_prompt_cache`` call lasting 2 s
    can straddle a 3.5 s budget and get SIGKILL'd mid-write, leaving
    ``cache_dir.new/`` orphaned. We exercise the contract directly so
    a regression that drops the kwarg fails locally instead of
    presenting as "rare orphan dir on shutdown" in production.
    """
    import time as _time

    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    captured = {"pred": None}

    class _Engine:
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            captured["pred"] = should_abort
            return False

    class _FakeCfg:
        engine = _Engine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    # 2 s budget — comfortably under the headroom-padded deadline (so
    # the at-deadline check is False) but a 5 s predicted operation
    # blows past it.
    _cache_mod.save_prefix_cache_to_disk(budget_sec=2.0)
    pred = captured["pred"]
    assert pred is not None
    assert pred(predicted_sec=0.0) is False, (
        "predicate fires too eagerly — a 2s budget shouldn't trip at t=0 "
        "with no predicted duration"
    )
    assert pred(predicted_sec=5.0) is True, (
        "predicate must look forward: starting a 5s op under a 2s budget "
        "should abort BEFORE the op begins, not let it straddle the deadline"
    )
    # Sanity: the no-arg call shape (legacy `should_abort()`) also still
    # works thanks to the default arg, but should NOT trip at t=0.
    # This keeps tests / callers that don't yet pass predicted_sec
    # compatible during the transition.
    _ = pred()
    # Wait past the deadline (minus headroom) — at-now check should now trip.
    _time.sleep(2.0)
    assert pred() is True, "at-now check should trip after wall-clock past deadline"


def test_save_prefix_cache_to_disk_fallback_for_legacy_engine_signature(monkeypatch):
    """Codex PR #667 round 1 BLOCKING-1 regression.

    External / third-party engine implementations may still expose the
    legacy one-argument ``save_cache_to_disk(cache_dir)`` signature.
    The runtime helper unconditionally passes ``should_abort=`` which
    would raise ``TypeError`` against those engines, losing the entire
    save. The fallback retries the call with the legacy positional
    shape — losing deadline awareness, but preserving the save.
    """
    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    calls = []

    class _LegacyEngine:
        # Note: no ``should_abort`` kwarg — emulates a pre-#667 plugin.
        def save_cache_to_disk(self, cache_dir):
            calls.append(cache_dir)
            return True

    class _FakeCfg:
        engine = _LegacyEngine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    # Must not raise. Must reach the engine. Must not crash the
    # lifespan shutdown.
    _cache_mod.save_prefix_cache_to_disk(budget_sec=1.0)
    assert len(calls) == 1, (
        "legacy engine should be invoked once after the deadline-aware "
        "call's TypeError is caught + fallback retried"
    )


def test_save_prefix_cache_to_disk_no_retry_on_internal_typeerror(monkeypatch):
    """Codex PR #667 round 2 BLOCKING-2 regression.

    The round-1 fallback caught any ``TypeError`` whose message
    mentioned ``should_abort`` and retried via the legacy signature —
    a compatible engine raising that error AFTER partial side effects
    (write to disk, metric increment) would be invoked TWICE, doubling
    the side effects.

    Round-2 detection is signature-based (``inspect.signature``) up
    front, so an internal TypeError raised by the engine method body
    propagates without retry. We assert exactly one call.
    """
    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    calls = {"n": 0}

    class _BuggyEngine:
        # Signature DOES accept should_abort — so detection passes,
        # call proceeds, error raised internally must NOT trigger a
        # legacy-signature retry.
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            calls["n"] += 1
            raise TypeError("internal failure mentioning should_abort somewhere")

    class _FakeCfg:
        engine = _BuggyEngine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    _cache_mod.save_prefix_cache_to_disk(budget_sec=1.0)
    assert calls["n"] == 1, (
        f"engine method must be invoked exactly once even on internal "
        f"TypeError — round-1 fallback double-called via legacy path, "
        f"got {calls['n']} calls"
    )


def test_adapt_should_abort_handles_keyword_only_and_args_kwargs():
    """``_adapt_should_abort`` must classify a few non-obvious callable
    shapes correctly. Regressions here would silently call the wrong
    shape and either raise TypeError mid-save or drop the
    ``predicted_sec`` arg (re-introducing the round-1 BLOCKING-2 bug).
    """
    from vllm_mlx.memory_cache import _adapt_should_abort

    # None passes through.
    assert _adapt_should_abort(None) is None

    # Zero-arg: must be called with no args.
    captured = []
    adapted = _adapt_should_abort(lambda: captured.append("z") or True)
    assert adapted(0.5) is True
    assert captured == ["z"]

    # One-arg positional: must receive predicted_sec.
    captured = []
    adapted = _adapt_should_abort(lambda p: captured.append(p) or False)
    assert adapted(1.5) is False
    assert captured == [1.5]

    # **kwargs only: must be called BY KEYWORD as predicted_sec=...
    # (codex PR #667 round 4 BLOCKING-1: round 3 classified **kwargs
    # as positional-capable and raised TypeError on call). The
    # predicate sees the value via ``kw["predicted_sec"]``.
    captured = []

    def kw_only_pred(**kw):
        captured.append(kw.get("predicted_sec", "no-arg"))
        return False

    adapted = _adapt_should_abort(kw_only_pred)
    assert adapted(2.5) is False
    assert captured == [2.5], (
        f"**kwargs predicate must receive predicted_sec by keyword, got {captured}"
    )

    # Keyword-only ``predicted_sec=...`` — same routing as **kwargs.
    captured = []

    def kw_only_named(*, predicted_sec=0.0):
        captured.append(predicted_sec)
        return False

    adapted = _adapt_should_abort(kw_only_named)
    assert adapted(3.5) is False
    assert captured == [3.5]

    # ``*args, **kwargs`` — positional path wins (it accepts positional
    # via the ``*args`` part).
    captured = []

    def star_args(*a, **kw):
        captured.append(("args", a, "kw", kw))
        return False

    adapted = _adapt_should_abort(star_args)
    assert adapted(2.5) is False
    assert captured == [("args", (2.5,), "kw", {})]


def test_save_prefix_cache_to_disk_zero_budget_disables_deadline(tmp_path, monkeypatch):
    """A budget of 0 (or negative) means "full flush, no deadline" — the
    engine should receive ``should_abort=None`` so the offline CLI path
    is unaffected.
    """
    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    captured = {"pred": "unset"}

    class _Engine:
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            captured["pred"] = should_abort
            return True

    class _FakeCfg:
        engine = _Engine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    _cache_mod.save_prefix_cache_to_disk(budget_sec=0.0)
    assert captured["pred"] is None


# --------------------------------------------------------------------------
# R8-M7 — commit-phase hardening (dogfood-089 Talia r1/r2)
# --------------------------------------------------------------------------
#
# The commit phase of save_to_disk does two non-atomic renames
# (``cache_dir → .old`` then ``.new → cache_dir``). Pre-R8-M7 there was
# no exception handler around either: a transient OSError (PermissionError
# from a fs-event-driven antivirus touching cache_dir mid-rename, observed
# on macOS Spotlight rebuilds; ENOSPC mid-shutdown; EBUSY on Windows)
# raised up to the caller and left ``cache_dir`` absent + ``.new`` orphan.
# load_from_disk's recovery path handled THAT shape — but the next save's
# pre-clean unconditionally rmtree'd ``.new``, silently discarding the
# just-saved snapshot if reboot didn't happen between the failed save
# and the next save attempt (e.g. an embedded harness that does multiple
# in-process save cycles).
#
# The R8-M7 fix wraps the rename phase in a try/except that attempts
# in-process recovery before returning, plus an ``fsync`` on the staging
# dir before the rename so the rename actually commits the right
# contents on a kernel-level crash.


def test_fsync_dir_works_on_real_directory(tmp_path):
    """The fsync helper itself works on a vanilla tmp directory.

    Regression guard: a future refactor that swaps ``os.O_RDONLY`` for
    ``O_DIRECTORY`` (not available on every platform) or changes the
    file-descriptor lifecycle would silently break this helper, and
    save_to_disk would still succeed (the except clause is non-fatal)
    but lose the rename-atomicity guarantee. The test below pins the
    happy path so a regression in helper semantics is observable.
    """
    from vllm_mlx.memory_cache import _fsync_dir

    # No exception on a real, readable directory.
    _fsync_dir(str(tmp_path))


def test_fsync_file_works_on_real_file(tmp_path):
    """The per-file fsync helper works on a vanilla tmp file.

    R8-M7 codex r1 BLOCKING #3: ``_fsync_dir`` only covers directory
    metadata; ``_fsync_file`` covers the file body. Pin both so a
    future refactor that drops one or the other (or breaks the
    fd-lifecycle in either) surfaces here.
    """
    from vllm_mlx.memory_cache import _fsync_file

    p = tmp_path / "blob.bin"
    p.write_bytes(b"x" * 4096)
    _fsync_file(str(p))
