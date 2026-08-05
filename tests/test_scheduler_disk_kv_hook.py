# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the scheduler's disk-KV hook wiring.

The existing :mod:`tests.test_disk_kv_checkpoint` battery pins the
`vllm_mlx.runtime.disk_kv_checkpoint` module API in isolation — but PR
#919 shipped wrong-attribute typos (``self.scheduler_config`` and
``self.batch_gen``) **inside the scheduler hook** that called that
module, and the silent-swallow wrapper at ``Scheduler._process_batch_``
``responses`` swallowed the ``AttributeError`` with ``logger.debug`` for
two releases. ``qmlx_kv_checkpoint_writes_total`` sat at 0 in
production while every unit test below still passed.

These tests exercise the scheduler hook end-to-end (with a stubbed
``BatchGenerator``) so the same class of bug cannot ship silently
again:

1. ``test_scheduler_hook_increments_writes_at_256_tok_boundary`` —
   drives ``_maybe_disk_checkpoint`` across 0/255/256/512 token
   counts and asserts the writes counter ticks. Catches the
   ``self.scheduler_config`` / ``self.batch_gen`` typo class.

2. ``test_scheduler_hook_no_op_when_interval_disabled`` —
   ``kv_disk_checkpoint_interval == 0`` must short-circuit before
   any disk IO. Pins the hot-path-cost contract.

3. ``test_scheduler_hook_no_op_when_batch_generator_absent`` —
   pre-prefill state (no ``batch_generator``) must early-return
   without raising. Pins the canonical ``getattr`` default.

4. ``test_safe_disk_checkpoint_records_silent_failure`` — when
   ``_maybe_disk_checkpoint`` raises (we patch in an injected
   ``AttributeError`` standing in for the wrong-attribute typo class
   of bug), the wrapper must (a) bump
   ``hook_errors`` so the failure is visible in ``/metrics``,
   (b) emit a ``warning`` log so operators tailing the server log
   notice, and (c) never re-raise. This is the explicit regression
   guard for the silent-swallow pattern.

The tests instantiate ``Scheduler`` directly with a no-op model and a
stub tokenizer — booting a real model would be a 5-second-per-test
overhead; the hook's interaction surface is small enough that a stub
is sufficient and faster.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.runtime import disk_kv_checkpoint as _dkc  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``get_default_root`` at a per-test directory + zero counters."""
    root = tmp_path / "kv-checkpoints"
    root.mkdir()
    monkeypatch.setattr(_dkc, "get_default_root", lambda: str(root))
    _dkc.reset_stats_for_tests()
    return root


def _make_scheduler(interval: int = 256) -> Scheduler:
    """Build a minimal ``Scheduler`` for hook-only tests.

    The hook reads only ``self.config``, ``self.batch_generator`` and
    ``self._model_name``; the rest of the scheduler is irrelevant to
    the code under test. We pass a stub model + tokenizer so
    ``Scheduler.__init__`` succeeds without booting MLX.
    """
    cfg = SchedulerConfig()
    cfg.kv_disk_checkpoint_interval = interval
    cfg.kv_cache_dtype = "bf16"
    # SSD-first PR3 (#16) flipped kv_disk_restore_enabled to True by default,
    # which gates OFF the interval write hook (its tokens-less bodies pollute
    # restore). These fixtures exercise the write hook, so default it off;
    # the restore-gating tests set it True explicitly.
    cfg.kv_disk_restore_enabled = False
    # Disable prefix-cache machinery so __init__ stays cheap and we
    # don't need a real model.
    cfg.enable_prefix_cache = False

    class _StubTok:
        eos_token_id = 0
        pad_token_id = 0
        special_tokens_map: dict[str, Any] = {}

        def decode(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
            return ""

    sched = Scheduler(model=object(), tokenizer=_StubTok(), config=cfg)
    sched._model_name = "test-model"
    return sched


def _seed_kv_cache(num_tokens: int = 16) -> list[Any]:
    """One-layer prompt cache seeded with a small KV pair.

    Mirrors :func:`tests.test_disk_kv_checkpoint._seed_kv_cache`. Small
    shapes keep the safetensors write under 100 KB per checkpoint so
    the suite is still CPU/disk-cheap.
    """
    from mlx_lm.models.cache import KVCache

    cache = KVCache()
    k = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(0))
    v = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(1))
    cache.update_and_fetch(k, v)
    return [cache]


def _make_request(num_tokens: int, batch_uid: int = 7) -> Request:
    """Synthesize a Request that reports ``num_tokens`` via the property.

    The hook reads ``request.num_tokens`` (prompt + output) and uses
    ``request.batch_uid`` to index into the BatchGenerator. We set
    ``num_prompt_tokens`` to the target so the property returns it
    without needing to roll the output_token_ids list.
    """
    req = Request(
        request_id=f"req-{batch_uid}",
        prompt="ignored",
        sampling_params=SamplingParams(max_tokens=2048),
    )
    req.num_prompt_tokens = num_tokens
    req.batch_uid = batch_uid
    return req


def _attach_stub_batch_generator(sched: Scheduler, request: Request) -> None:
    """Give the scheduler a stub ``batch_generator`` exposing the hook surface.

    The hook walks ``batch._generation_batch`` first, then
    ``batch.active_batch`` — we expose ``_generation_batch`` with
    ``uids`` + ``extract_cache`` matching the mlx-lm 0.31+ shape.
    """
    cache = _seed_kv_cache(num_tokens=16)
    gen_batch = SimpleNamespace(
        uids=[request.batch_uid],
        extract_cache=lambda e: cache if e == 0 else None,
    )
    batch = SimpleNamespace(
        _generation_batch=gen_batch,
        active_batch=None,
    )
    sched.batch_generator = batch


# ---------------------------------------------------------------------------
# 1) Hook reaches maybe_write_checkpoint and ticks writes_total
# ---------------------------------------------------------------------------


def test_scheduler_hook_increments_writes_at_256_tok_boundary(
    isolated_root: Path,
) -> None:
    """Below 256: no write. At 256 and 512: one write each.

    Catches the wrong-attribute typo class of bug introduced in PR
    #919 — both ``self.scheduler_config`` (config) and ``self.batch_gen``
    (BatchGenerator) raised ``AttributeError`` here, the wrapper
    swallowed the exception at ``logger.debug``, and ``writes_total``
    sat at 0. This test exercises the exact ``getattr`` reads on the
    real ``Scheduler`` instance — no stub of the scheduler itself —
    so an attribute-name regression cannot pass this test.
    """
    sched = _make_scheduler(interval=256)
    req = _make_request(num_tokens=200)
    _attach_stub_batch_generator(sched, req)

    # Below first boundary — no write.
    sched._maybe_disk_checkpoint(req, response=SimpleNamespace())
    assert _dkc.get_stats()["writes"] == 0

    # Cross the 256 boundary — writes ticks to 1.
    req.num_prompt_tokens = 260
    sched._maybe_disk_checkpoint(req, response=SimpleNamespace())
    assert _dkc.get_stats()["writes"] == 1

    # Cross the 512 boundary — writes ticks to 2.
    req.num_prompt_tokens = 520
    sched._maybe_disk_checkpoint(req, response=SimpleNamespace())
    assert _dkc.get_stats()["writes"] == 2

    # And the checkpoint dir was created under the isolated root.
    files = list(isolated_root.rglob("checkpoint-*.safetensors"))
    assert len(files) >= 2, f"expected at least 2 checkpoint files, got {files}"


# ---------------------------------------------------------------------------
# 2) Disabled-interval contract: no disk IO
# ---------------------------------------------------------------------------


def test_scheduler_hook_no_op_when_interval_disabled(isolated_root: Path) -> None:
    """``kv_disk_checkpoint_interval == 0`` must short-circuit.

    Pins the hot-path-cost contract — operators who haven't opted in
    pay one int comparison, not a disk-cache scan.
    """
    sched = _make_scheduler(interval=0)
    req = _make_request(num_tokens=1024)
    _attach_stub_batch_generator(sched, req)

    sched._maybe_disk_checkpoint(req, response=SimpleNamespace())

    assert _dkc.get_stats()["writes"] == 0
    assert not list(isolated_root.rglob("*.safetensors"))


# ---------------------------------------------------------------------------
# 3) No-batch-generator early-return (expected skip path)
# ---------------------------------------------------------------------------


def test_scheduler_hook_no_op_when_batch_generator_absent(
    isolated_root: Path,
) -> None:
    """Pre-prefill state (no ``batch_generator``) must early-return.

    Pins the canonical ``getattr(self, "batch_generator", None)`` default
    — and protects against a regression where the hook would raise on
    None, which the wrapper would then record as a ``hook_errors`` tick.
    """
    sched = _make_scheduler(interval=256)
    req = _make_request(num_tokens=512)
    # Intentionally do not attach batch_generator.

    sched._maybe_disk_checkpoint(req, response=SimpleNamespace())

    stats = _dkc.get_stats()
    assert stats["writes"] == 0
    assert stats["hook_errors"] == 0  # Expected skip, not an error.


# ---------------------------------------------------------------------------
# 4) Silent-swallow regression guard
# ---------------------------------------------------------------------------


def test_safe_disk_checkpoint_records_silent_failure(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``_safe_disk_checkpoint`` must bump ``hook_errors`` + warn on raise.

    This is the explicit regression guard for the bug class PR #919
    shipped: a wrong-attribute typo inside ``_maybe_disk_checkpoint``
    raised ``AttributeError`` every step, the wrapper at
    ``Scheduler._process_batch_responses`` swallowed it at
    ``logger.debug``, and the failure was invisible until an operator
    happened to scrape ``/metrics`` and notice
    ``writes_total == 0`` after a 4k-token completion.

    With the new contract:
    * ``hook_errors`` must increment (visible in ``/metrics``).
    * A ``warning``-level log must fire (visible to operators tailing
      the server log).
    * The wrapper must NOT re-raise (the live decode path must keep
      streaming tokens even if the hook is broken).
    """
    sched = _make_scheduler(interval=256)
    req = _make_request(num_tokens=512)

    # Patch _maybe_disk_checkpoint to raise an AttributeError — exactly
    # what the #919 typos did. Direct attribute write because Scheduler
    # is a regular class, not a dataclass.
    def _raises(self: Scheduler, request: Request, response: Any) -> None:
        raise AttributeError("Scheduler object has no attribute 'scheduler_config'")

    monkeypatch.setattr(Scheduler, "_maybe_disk_checkpoint", _raises)

    before = _dkc.get_stats()["hook_errors"]

    with caplog.at_level(logging.WARNING, logger="qmlx.scheduler"):
        # Wrapper must not raise.
        sched._safe_disk_checkpoint(req, response=SimpleNamespace())

    after = _dkc.get_stats()["hook_errors"]

    # 1. Prometheus counter visible signal.
    assert after == before + 1, (
        f"hook_errors must tick on silent failure (before={before}, after={after})"
    )

    # 2. Warning log visible signal.
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "kv_checkpoint" in r.getMessage()
    ]
    assert warnings, (
        "wrapper must emit warning on hook failure — silence is the bug we are "
        f"guarding against. caplog records: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )

    # 3. Never re-raise — the wrapper's contract is that it MUST NOT
    # propagate exceptions, because a disk-IO failure must not crash a
    # live decode. Re-invoke the wrapper inside ``pytest.raises`` with a
    # ``no exception`` clause (negative-control idiom) so a future
    # refactor that drops the broad ``except`` is caught here, not by
    # a request timing out in production.
    sched._safe_disk_checkpoint(req, response=SimpleNamespace())  # must not raise

    # And the regression test would have also caught the *original* bug
    # had it been in place at #919's review — bump the assertion to
    # double-check the second call ticked again, so a "swallows but
    # forgets to record" regression also fails here.
    assert _dkc.get_stats()["hook_errors"] == after + 1


# ---------------------------------------------------------------------------
# 5) Multi-step boundary checkpoint survives the interval-write flood (#9)
# ---------------------------------------------------------------------------


def test_multistep_boundary_survives_interval_flood(isolated_root: Path) -> None:
    """The received-prompt boundary checkpoint (written by the store mirror,
    keyed on prompt_token_ids) must survive a flood of tokens-less interval
    writes and still be a true-prefix restore hit for the NEXT turn's prompt.

    Reproduces issue #9: a multi-step turn generates long output, firing many
    interval writes; before the matchable-aware eviction those tokens-less
    bodies evicted the boundary checkpoint the next turn needed, cold-filling
    the cache.
    """
    _dkc.reset_content_index_for_tests()
    sched = _make_scheduler(interval=256)

    # Deposit the boundary checkpoint via the real store mirror (matchable:
    # carries tokens_key == prompt, so it is indexed + restorable).
    prompt = list(range(1000, 1000 + 256))
    sched._disk_persist_boundary(prompt, _seed_kv_cache(num_tokens=256))

    tok_blobs = list(isolated_root.rglob("*.tokens.bin"))
    assert len(tok_blobs) == 1, f"mirror did not deposit boundary blob: {tok_blobs}"
    boundary_body = Path(str(tok_blobs[0]).replace(".tokens.bin", ".safetensors"))
    matchable_size = boundary_body.stat().st_size + tok_blobs[0].stat().st_size

    # Flood with tokens-less interval-style checkpoints (no tokens_key).
    root = str(isolated_root)
    for i in range(8):
        p = _dkc.write_checkpoint(
            _seed_kv_cache(num_tokens=8),
            root=root,
            req_hash=_dkc.request_hash(f"interval-{i}", model_name="m"),
            token_offset=256,
            kv_dtype="bf16",
            model_name="m",
        )
        assert p is not None

    # Enforce a cap that would evict under a naive oldest-first policy. With
    # low_water=1.0 the loop drains to exactly the cap, and the matchable
    # boundary (evicted last, by class) is left untouched.
    _dkc.enforce_disk_cap(root, max_bytes=matchable_size, low_water_fraction=1.0)

    # Boundary checkpoint (body + tokens blob) survived.
    assert boundary_body.exists()
    assert tok_blobs[0].exists()

    # And the NEXT turn's prompt (prompt + tool_call + tool_result tokens)
    # still resolves to it as a true-prefix restore hit.
    query = prompt + [50, 51, 77, 78]
    loaded = _dkc.get_content_index().lookup(query)
    assert loaded is not None
    assert loaded.token_offset == len(prompt)


# ---------------------------------------------------------------------------
# 6) Interval hook is gated off when disk-KV restore is enabled (Change 2a)
# ---------------------------------------------------------------------------


def test_interval_hook_skipped_when_restore_enabled(isolated_root: Path) -> None:
    """With ``kv_disk_restore_enabled`` true, the interval hook must not
    write. Its tokens-less bodies are never matchable / never consumed by
    restore, and pre-#9 they evicted the boundary checkpoints restore needs.

    The skip is an EXPECTED early-return, so it must not bump ``hook_errors``.
    """
    sched = _make_scheduler(interval=256)
    sched.config.kv_disk_restore_enabled = True
    req = _make_request(num_tokens=512)
    _attach_stub_batch_generator(sched, req)

    sched._maybe_disk_checkpoint(req, response=SimpleNamespace())

    stats = _dkc.get_stats()
    assert stats["writes"] == 0
    assert stats["hook_errors"] == 0  # expected skip, not an error
    assert not list(isolated_root.rglob("*.safetensors"))


def test_interval_hook_still_writes_when_restore_disabled(isolated_root: Path) -> None:
    """Control for the gate: with restore explicitly DISABLED, the interval
    hook still writes at the boundary — the gate must key off the restore
    flag only, not disable the hook wholesale.

    Note: SSD-first PR3 (#16) flipped ``kv_disk_restore_enabled`` to True by
    default, so this control sets it False explicitly.
    """
    sched = _make_scheduler(interval=256)
    sched.config.kv_disk_restore_enabled = False
    assert sched.config.kv_disk_restore_enabled is False
    req = _make_request(num_tokens=512)
    _attach_stub_batch_generator(sched, req)

    sched._maybe_disk_checkpoint(req, response=SimpleNamespace())

    assert _dkc.get_stats()["writes"] == 1


# ---------------------------------------------------------------------------
# 7) Disk write path survives the in-memory prefix-cache deletion (PR3, #16)
# ---------------------------------------------------------------------------


def test_disk_persist_survives_ram_cache_deletion(isolated_root: Path) -> None:
    """Guard for SSD-first PR3 (#16): the four disk-persist store sites were
    previously gated on ``self.memory_aware_cache is not None``. Deleting the
    RAM prefix cache without re-gating on the disk predicate would silently
    stop ALL disk writes.

    Drive ``_disk_persist_boundary`` directly (the store-site body kept after
    the RAM ``.store()`` calls were dropped) with a >=256-token cache and
    assert (a) a checkpoint safetensors lands under ``get_default_root()`` and
    (b) a second identical lookup resolves to it as a true-prefix restore hit.
    """
    _dkc.reset_content_index_for_tests()
    sched = _make_scheduler(interval=256)
    sched.config.kv_disk_restore_enabled = True
    # Disk-only tier is default now; there is no memory_aware_cache attribute.
    assert not hasattr(sched, "memory_aware_cache")

    prompt = list(range(2000, 2000 + 256))
    sched._disk_persist_boundary(prompt, _seed_kv_cache(num_tokens=256))

    bodies = list(isolated_root.rglob("*.safetensors"))
    assert bodies, f"disk write path produced no checkpoint under {isolated_root}"

    loaded = _dkc.get_content_index().lookup(prompt)
    assert loaded is not None
    assert loaded.token_offset == len(prompt)


def test_cleanup_finished_store_site_writes_checkpoint(isolated_root: Path) -> None:
    """End-to-end guard for the SSD-first PR3 (#16) store-site re-gate.

    The other guard drives ``_disk_persist_boundary`` directly, so a future
    regression that re-wraps the ``_cleanup_finished`` store block in a dead
    condition would slip through. Drive the ACTUAL completion store site: a
    finished request carrying ``_extracted_cache`` through
    ``_cleanup_finished``, and assert a checkpoint lands and restores.
    """
    _dkc.reset_content_index_for_tests()
    sched = _make_scheduler(interval=256)
    sched.config.kv_disk_restore_enabled = True
    assert not hasattr(sched, "memory_aware_cache")

    req = Request(
        request_id="fin-1",
        prompt="ignored",
        sampling_params=SamplingParams(max_tokens=2048),
    )
    req.prompt_token_ids = list(range(4000, 4000 + 250))
    req.output_token_ids = list(range(9000, 9000 + 6))  # full seq = 256
    req._extracted_cache = _seed_kv_cache(num_tokens=256)
    sched.running = {req.request_id: req}

    sched._cleanup_finished({req.request_id})

    bodies = list(isolated_root.rglob("*.safetensors"))
    assert bodies, (
        f"_cleanup_finished store site wrote no checkpoint under {isolated_root}"
    )

    full = list(range(4000, 4000 + 250)) + list(range(9000, 9000 + 6))
    loaded = _dkc.get_content_index().lookup(full)
    assert loaded is not None
    assert loaded.token_offset == 256
    assert req.request_id not in sched.running


def test_completion_checkpoint_keyed_on_stable_prompt_prefix(
    isolated_root: Path,
) -> None:
    """Regression for the multi-turn KV-restore prefix break (#9).

    The completion store site sees prompt + generated-output tokens. The output
    tokens (empty ``<think></think>`` scaffold, tool-call formatting) do NOT
    round-trip when the client re-tokenizes the conversation next turn, so a
    checkpoint keyed on the prompt+output tail is never a prefix of the next
    prompt: every turn falls back to the system-prompt checkpoint and cold-
    prefills the whole conversation. The store site now marks the prompt length
    as the round-trip-stable prefix, so the checkpoint is keyed (and its KV
    sliced) to the prompt, and the NEXT turn's prompt matches it.
    """
    _dkc.reset_content_index_for_tests()
    sched = _make_scheduler(interval=256)
    sched.config.kv_disk_restore_enabled = True

    prompt = list(range(1000, 1000 + 300))  # 300 round-trip-stable tokens
    output = [40001, 40002, 40003]  # volatile generated-output tokens
    req = Request(
        request_id="turn-n",
        prompt="ignored",
        sampling_params=SamplingParams(max_tokens=2048),
    )
    req.prompt_token_ids = list(prompt)
    req.output_token_ids = list(output)
    req._extracted_cache = _seed_kv_cache(num_tokens=len(prompt) + len(output))
    sched.running = {req.request_id: req}

    sched._cleanup_finished({req.request_id})

    idx = _dkc.get_content_index()
    # Keyed on the PROMPT prefix, not the volatile prompt+output tail.
    assert tuple(prompt) in idx._by_key
    assert tuple(prompt + output) not in idx._by_key

    # Next turn: the client re-tokenized the assistant turn to different IDs, so
    # the stored output tokens are NOT a prefix, but the prompt is. The restore
    # must land the growing prompt checkpoint, not fall back to nothing.
    next_prompt = prompt + [50001, 50002] + [60001, 60002]
    loaded = idx.lookup(next_prompt)
    assert loaded is not None
    assert loaded.token_offset == len(prompt)
