# SPDX-License-Identifier: Apache-2.0
"""Disk-backed KV-cache checkpointing at 256-token boundaries (R15-P1 task #296).

This module is the long-context partner of the in-process radix prefix cache
(task #303): instead of holding the full KV tail in RAM for the lifetime of a
session, the scheduler snapshots the cache to disk at fixed token boundaries
(default 256, matching upstream MLX-LM's ``step=256`` allocator and LMCache's
external-chunk size). When the same prefix shows up later — same session
resumed, same shared system prompt, or a long-running agent walking up to its
context cap — the on-disk snapshot is reloaded instead of re-prefilled, which
is the headline 体感 row that unlocks "Mac users can run all day" (82% peak
RAM reduction at long context + 2.2× parallel-chat throughput).

The design is a deliberate port of LM Studio MLX-engine PR #326's prompt-
cache layer (specifically the ``prompt_cache/`` package at commit
``ea1a6bb16``), narrowed to the rapid-mlx use case:

- **Boundary granularity**: 256 tokens (``DEFAULT_CHECKPOINT_INTERVAL``),
  same as ``mlx_engine/.../types.py::DEFAULT_PREFIX_CHUNK_SIZE``. The MLX-LM
  KV cache allocates in 256-token steps, so writing at a multiple of 256 keeps
  the on-disk shape aligned with the in-memory shape — important because the
  loader uses ``mlx_lm.load_prompt_cache`` which reads the cache class name
  out of the safetensors metadata, then constructs a fresh
  ``KVCache``/``QuantizedKVCache`` whose step rounding has to match.
- **On-disk format**: ``mlx_lm.models.cache.save_prompt_cache`` /
  ``load_prompt_cache`` directly. Same path the in-process radix already uses
  for its store/fetch round-trip (memory_cache.py:2017). Pre-existing
  round-trip / corruption / dedup guards (R10-D, R12-T1) apply here too with
  zero extra code.
- **Atomic writes**: write to ``<token_offset>.safetensors.tmp`` + fsync +
  rename to ``<token_offset>.safetensors``. Mirrors the prefix-cache
  ``cache_dir.new/`` → ``cache_dir`` rename in ``MemoryAwarePrefixCache``.
- **Disk-budget eviction**: oldest-first across all checkpoints in
  ``~/.cache/rapid-mlx/kv_checkpoints/``, capped at a configurable byte cap
  (default 20 GiB, env override ``RAPID_MLX_KV_CHECKPOINT_MAX_BYTES``).
  ``mtime``-ordered LRU rather than the size-aware policy in PR #326 because
  the rapid-mlx scheduler is single-tenant per process and the cap exists
  primarily to keep a runaway agent from filling the disk, not to optimize
  hit rate across a vision-mixed workload.
- **Special-model handling**: a small registry
  (:data:`MODELS_REQUIRING_FULL_CHECKPOINT`) marks families whose attention
  cache cannot be sliced — Gemma 4 sliding-window (the cache holds the live
  window state and the offset alone can't reconstruct it) and Qwen3.5 hybrid
  attention (full + sliding layers alternate). For these we write the WHOLE
  ``prompt_cache`` list at the boundary; for everything else we write the
  whole list too (we don't slice — rapid-mlx loads checkpoints "as a
  resumable suspension point" and the writer doesn't have to know which
  layers are sliceable). The registry is exposed for the loader because a
  partial restore policy could be added later: today both paths converge.

Deviations from LM Studio PR #326 (documented for the PR body):

- **No record-kind slicing**. The upstream code separates ``kv_delta`` /
  ``rotating_delta`` / ``state_checkpoint`` and writes per-layer per-chunk.
  We write the whole cache list at one boundary because (a) rapid-mlx's
  in-process radix already handles cross-tenant prefix dedup, so disk
  checkpoints don't need to dedup against each other, and (b) the upstream
  delta path requires fine-grained slicing that doesn't compose with the
  ``QuantizedKVCache`` (whose triple of (packed, scales, biases) can't be
  sliced cheaply on the seq axis without dequantizing first — the prefix
  cache already learnt this the hard way at memory_cache.py:2014).
- **No image-span hashing**. We index by request hash, not by chunk hash.
  Image / vision is handled by the rapid-mlx ``mllm_*`` lane on a separate
  cache.
- **No blob-store coalescing**. Each checkpoint is its own safetensors file
  under a per-request directory; the disk-cap eviction policy is
  mtime-ordered across all files. The upstream
  ``TemporarySafetensorBlobStore`` uses a packed temp-file with extent
  coalescing because the upstream coordinator may write hundreds of small
  delta records per request; we write one per 256-token boundary per
  request, where the disk pressure simply doesn't justify a packed store.

Integration touchpoints:

- The scheduler calls :func:`maybe_write_checkpoint` after every step that
  pushes ``request.num_computed_tokens`` past the next 256-token boundary.
  Cheap when disabled (``interval=0`` short-circuits with no I/O).
- ``vllm_mlx.runtime.cache`` loads checkpoints during startup via
  :func:`scan_checkpoints` — the radix index gets a metadata flag so the
  next lookup knows the entry's source was disk, not RAM. Hand-off to the
  radix is best-effort: a missing index entry just means the next request
  will re-prefill, not crash.
- ``vllm_mlx.runtime.disk_kv_checkpoint.get_stats`` returns a stats dict
  the scheduler folds into ``get_stats()`` so ``/metrics`` can render the
  four ``rapid_mlx_kv_checkpoint_*`` series (writes, loads, bytes,
  evictions).

Concurrency: every public function takes a per-checkpoint-root ``RLock``
(module-level). Writers serialise on the lock; readers ``scan_checkpoints``
also takes the lock to avoid racing the disk-cap eviction. The lock is
cheap because writes happen at most once per 256 generated tokens — way
below the per-step scheduler cadence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Default checkpoint interval — matches MLX-LM's KVCache.step (256) and
# LMCache's external chunk size. Picked because round-tripping a checkpoint
# whose token count isn't a multiple of the underlying cache step would
# force the loader to allocate a non-step-aligned buffer on first reuse,
# which then trips the same allocation-noise path the in-process radix is
# already careful to avoid (see _grow_kv_cache step rounding in
# vllm_mlx/positioned_kv_cache.py).
DEFAULT_CHECKPOINT_INTERVAL = 256

# Disk cap default: 20 GiB. The env override is honoured at scan-time so an
# operator can shrink/grow without restarting the server. Picked to match the
# headroom rapid-desktop reserves under ~/.cache/rapid-mlx (#194).
DEFAULT_MAX_DISK_BYTES = 20 * 1024 * 1024 * 1024
_DISK_CAP_ENV = "RAPID_MLX_KV_CHECKPOINT_MAX_BYTES"

# Models that require a FULL cache-state snapshot at each boundary — i.e. the
# attention cache cannot be reconstructed from a position offset alone:
#
# - **Gemma 4 sliding-window**: every layer holds a fixed-size window that
#   rolls forward; rewinding by N tokens requires the actual window contents
#   at that position, not just the offset. Our writer captures the entire
#   ``prompt_cache`` list, which includes the live window state, so the
#   loader gets a faithful resume point.
# - **Qwen3.5 hybrid attention**: full-attention and sliding layers
#   alternate. The sliding layers have the same constraint as Gemma 4; we
#   therefore checkpoint both layer types together — there's no benefit to
#   per-layer slicing because the cache state has to round-trip in
#   lock-step.
#
# Pattern match is case-insensitive substring over BOTH the alias key and
# the resolved HF path (mirrors ``kv_cache_dtype._is_sliding_window``).
# Entries are family globs ("gemma-4-*") rather than exact aliases so newly-
# uploaded quants (e.g. ``mlx-community/gemma-4-12b-int4``) auto-pick the
# right policy without an aliases.json edit.
MODELS_REQUIRING_FULL_CHECKPOINT: frozenset[str] = frozenset(
    {
        "gemma-4",
        "gemma_4",
        "gemma4",
        "qwen3.5",
        "qwen3_5",
        "qwen35",
    }
)

# Filename suffix on the persisted safetensors blob. Kept short because a
# busy disk root accumulates one file per boundary per request and Linux
# ``readdir`` cost scales with path length.
_CHECKPOINT_EXT = ".safetensors"
_METADATA_EXT = ".json"
# Sidecar carrying the EXACT prompt token ids the checkpoint represents,
# in the same magic/length/save_uuid/int32-LE format the in-process radix
# uses on disk (``memory_cache._write_tokens_bin_v3``). Written next to the
# safetensors body so a later restore can byte-verify the prefix before it
# trusts the cache — a token blob that doesn't match the incoming prompt is
# the difference between a hit and a silent-corruption reload.
_TOKENS_EXT = ".tokens.bin"

# Highest sidecar ``schema_version`` this build knows how to load. The
# writer stamps ``schema_version=1``; :func:`load_checkpoint` refuses any
# sidecar whose version is absent or greater than this so a checkpoint from
# a newer, format-shifted build can never be mis-read as a current one
# (reject-and-reprefill on any doubt — a wrong restore corrupts silently).
_KNOWN_SCHEMA_VERSION = 1
# Tmp file name shape: ``<basename>.tmp.safetensors``. The trailing
# ``.safetensors`` is REQUIRED because ``mlx.core.save_safetensors``
# silently auto-appends ``.safetensors`` when the path does not already
# end in it (mlx.core 0.31.3 behaviour, verified empirically). Without
# this shape, calling ``save_safetensors('foo.safetensors.tmp', ...)``
# actually writes ``foo.safetensors.tmp.safetensors`` and the subsequent
# rename fails. The ``.tmp.`` infix is what ``scan_checkpoints`` strips
# from on rescan.
_TMP_INFIX = ".tmp"


# ---------------------------------------------------------------------------
# Stats dataclass — folded into Scheduler.get_stats() for /metrics
# ---------------------------------------------------------------------------


@dataclass
class CheckpointStats:
    """Process-monotonic counters surfaced via ``/metrics``.

    Attributes:
        writes: cumulative ``write_checkpoint`` calls that committed (renamed
            the .tmp into place). Failed writes do NOT increment.
        loads: cumulative ``load_checkpoint`` calls that returned a non-None
            cache list.
        bytes: live total byte count across every committed checkpoint under
            the root, refreshed on every ``write_checkpoint`` / scan / evict.
            Gauge (not counter) because the value goes down on eviction.
        evictions: cumulative oldest-first evictions performed because the
            byte total crossed the cap. One per evicted file (so a single
            scan that releases 5 files bumps the counter by 5).
        hook_errors: cumulative unexpected exceptions caught by the
            scheduler's disk-KV hook wrapper (``Scheduler.``
            ``_process_batch_responses`` ``try/except`` around
            ``_maybe_disk_checkpoint``, plus the ``enforce_disk_cap``
            catch inside the hook itself). Counts wrong-attribute typos
            and similarly silent-shipped regressions. **Operators expect
            this to stay 0.** Added 2026-06-29 after PR #919's
            ``self.scheduler_config`` / ``self.batch_gen`` typos shipped
            for two releases without any signal — see the parent commit
            of this PR for the root-cause writeup.
    """

    writes: int = 0
    loads: int = 0
    bytes: int = 0
    evictions: int = 0
    hook_errors: int = 0
    # R15-P4 (task #303): per-reason restore-reject tally. Keyed by the
    # reason strings in :data:`RESTORE_REJECT_REASONS`. A restore that fails
    # ANY validation guard bumps exactly one reason here so operators can see
    # WHY disk restore is falling back to prefill (dtype drift vs a full/
    # partial mismatch vs a memory-headroom skip look identical in the loads
    # counter, which never moved because the load was refused).
    restore_rejects: dict[str, int] = field(default_factory=dict)


# Known restore-reject reasons. Emitted at 0 by /metrics even before the first
# rejection so a dashboard panel stays flat-line rather than "no data". Any
# reason passed to :func:`record_restore_reject` that is not in this set is
# still counted (under its own label) — the set only seeds the always-present
# series.
RESTORE_REJECT_REASONS: tuple[str, ...] = (
    "offset_out_of_range",
    "kv_dtype_mismatch",
    "full_checkpoint_mismatch",
    "memory_headroom",
    "exception",
)


# Module-level stats (process-monotonic). Mutated under the lock below.
_STATS = CheckpointStats()
_STATS_LOCK = threading.Lock()
_DISK_LOCK = threading.RLock()

# ---------------------------------------------------------------------------
# LOCK ORDERING (R15-P2). Three locks can now be live at once — the
# in-process prefix cache's ``MemoryAwarePrefixCache._lock``
# (memory_cache.py), this module's ``_DISK_LOCK`` (guards all checkpoint
# filesystem I/O), and a :class:`DiskCheckpointIndex`'s own lock (guards the
# in-memory prompt→checkpoint map). To stay deadlock-free every acquisition
# path uses the SAME outer→inner order:
#
#     MemoryAwarePrefixCache._lock  >  _DISK_LOCK  >  DiskCheckpointIndex._lock
#
# Concretely:
#   1. A thread NEVER takes ``MemoryAwarePrefixCache._lock`` while holding
#      either lock below it. The disk layer never calls back into the
#      in-RAM cache, so this is satisfied structurally — the scheduler's
#      add_request resolves the in-RAM prefix cache first, releases that
#      lock, and only then queries the disk index.
#   2. A thread NEVER takes ``_DISK_LOCK`` while holding a
#      ``DiskCheckpointIndex`` lock. Both :meth:`DiskCheckpointIndex.lookup`
#      and :meth:`DiskCheckpointIndex.build_from_root` do their disk I/O
#      (which grabs ``_DISK_LOCK``) in a phase where the index lock is NOT
#      held, then take the index lock separately to read/populate the
#      in-memory map. The one legal nesting is the reverse:
#      :func:`write_checkpoint` already holds ``_DISK_LOCK`` and calls
#      :meth:`DiskCheckpointIndex.index_checkpoint`, which takes only the
#      index lock — matching the outer→inner order above.
#   3. None of these locks is held across a model forward. ``lookup``
#      returns the materialised cache list and releases every lock before
#      the scheduler feeds it to the generator.
_CONTENT_INDEX: DiskCheckpointIndex | None = None
_CONTENT_INDEX_LOCK = threading.Lock()


def get_stats() -> dict[str, int]:
    """Snapshot the process-monotonic counters as a dict.

    Called by ``Scheduler.get_stats()`` so ``/metrics`` can fold the four
    ``rapid_mlx_kv_checkpoint_*`` series next to the existing prefix-cache
    series. Snapshotting under the lock keeps a concurrent
    ``write_checkpoint`` from publishing a torn (writes, bytes) pair.
    """
    with _STATS_LOCK:
        # Seed the known reasons at 0 so /metrics always emits every series,
        # then overlay the live tallies. Copy so the caller can't mutate the
        # module-level dict.
        rejects = {reason: 0 for reason in RESTORE_REJECT_REASONS}
        rejects.update(_STATS.restore_rejects)
        return {
            "writes": _STATS.writes,
            "loads": _STATS.loads,
            "bytes": _STATS.bytes,
            "evictions": _STATS.evictions,
            "hook_errors": _STATS.hook_errors,
            "restore_rejects": rejects,
        }


def record_hook_error() -> None:
    """Bump the ``hook_errors`` counter under the stats lock.

    The scheduler's wrapper at ``Scheduler._process_batch_responses``
    calls this every time the disk-KV hook raises an unexpected
    exception (every *expected* skip path is an early-return inside
    ``_maybe_disk_checkpoint`` and never reaches the wrapper). The
    `enforce_disk_cap` catch inside the hook itself also calls this.
    Surfaces silent regressions like the wrong-attribute typos shipped
    in #919 — see the ``hook_errors`` field doc.
    """
    with _STATS_LOCK:
        _STATS.hook_errors += 1


def record_restore_reject(reason: str) -> None:
    """Bump the per-reason restore-reject tally under the stats lock.

    R15-P4 (task #303). Called from the scheduler's ``_maybe_disk_restore``
    every time a looked-up checkpoint fails a validation guard and the
    request falls back to prefill. ``reason`` should be one of
    :data:`RESTORE_REJECT_REASONS`, but an unknown reason is still counted
    under its own label rather than dropped — a mislabelled reject is better
    than a silent one.
    """
    key = str(reason) or "unknown"
    with _STATS_LOCK:
        _STATS.restore_rejects[key] = _STATS.restore_rejects.get(key, 0) + 1


def reset_stats_for_tests() -> None:
    """Test-only hook: zero the module-level counters.

    Prod code never calls this; the counters are process-monotonic by
    contract (matches every other Prometheus client library).
    """
    global _STATS
    with _STATS_LOCK:
        _STATS = CheckpointStats()


# ---------------------------------------------------------------------------
# Helpers — config resolution + path layout
# ---------------------------------------------------------------------------


def get_default_root() -> str:
    """Return the on-disk root for KV checkpoints.

    ``~/.cache/rapid-mlx/kv_checkpoints/`` — sibling of the existing
    ``prefix_cache/`` directory used by the in-process radix. The dir is
    created lazily by the first ``write_checkpoint`` so operators who never
    enable disk checkpointing don't see an empty directory show up.
    """
    return os.path.join(
        os.path.expanduser("~"), ".cache", "rapid-mlx", "kv_checkpoints"
    )


def resolve_max_disk_bytes(default: int = DEFAULT_MAX_DISK_BYTES) -> int:
    """Resolve the disk cap, honouring the env override.

    Returns 0 (cap disabled) when the env var is explicitly set to ``0``
    or a negative integer. Matches the convention the prefix cache uses
    for ``RAPID_MLX_PREFIX_CACHE_MAX_BYTES``: an explicit ``0`` is the
    escape hatch, not "use default".
    """
    raw = os.environ.get(_DISK_CAP_ENV)
    if raw is None:
        return max(0, int(default))
    try:
        n = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"[disk_kv_checkpoint] invalid {_DISK_CAP_ENV}={raw!r}; "
            f"falling back to default {default}"
        )
        return max(0, int(default))
    return max(0, n)


def request_hash(request_id: str, model_name: str | None = None) -> str:
    """Return a short stable hash that pins a request to its checkpoint dir.

    Includes the model name so the same ``request_id`` against two
    different models can't collide (the on-disk safetensors carries the
    cache class names and would error out at load time, but a hash
    collision in the directory layer is the cleaner failure mode).
    """
    raw = f"{model_name or ''}::{request_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def checkpoint_path(root: str, req_hash: str, token_offset: int) -> str:
    """Return the absolute safetensors path for one checkpoint."""
    return os.path.join(root, req_hash, f"checkpoint-{token_offset}{_CHECKPOINT_EXT}")


def metadata_path(root: str, req_hash: str, token_offset: int) -> str:
    """Return the absolute metadata JSON path for one checkpoint.

    The JSON sits next to the .safetensors and records the model name,
    KV dtype, sliding/hybrid flags, token offset, and write timestamp.
    Useful for the scan path and for the radix-index hand-off, which
    needs to know "where did this loaded entry come from".
    """
    return os.path.join(root, req_hash, f"checkpoint-{token_offset}{_METADATA_EXT}")


def tokens_path(root: str, req_hash: str, token_offset: int) -> str:
    """Return the absolute tokens-blob path for one checkpoint.

    Sits next to the ``.safetensors`` / ``.json`` pair and holds the exact
    prompt token ids the snapshot covers (v3 magic + length + save_uuid +
    int32-LE tokens). Present only when the writer was handed a
    ``tokens_key``; an older checkpoint without one simply has no blob and
    the loader treats it as "can't verify a prefix" (safe: re-prefill).
    """
    return os.path.join(root, req_hash, f"checkpoint-{token_offset}{_TOKENS_EXT}")


def model_requires_full_checkpoint(
    model_name: str | None,
    hf_path: str | None = None,
    alias_metadata: dict[str, Any] | None = None,
    hf_config: dict[str, Any] | None = None,
) -> bool:
    """Detect whether this model family must checkpoint the WHOLE cache.

    Detection order (cheapest first):
    1. ``alias_metadata['requires_full_checkpoint'] is True`` — explicit
       operator pin via aliases.json (works for verified-tier aliases
       whose family doesn't match a substring pattern). Does NOT touch
       the closed-key fields ``architecture`` / ``family`` /
       ``quantization`` / ``notes`` per the aliases.json schema rule —
       this is a new boolean key only.
    2. ``hf_config['sliding_window']`` populated — the canonical HF
       signal for sliding-window attention. Catches Gemma 4 + sliding
       Mistral variants without name matching.
    3. ``hf_config['hybrid_attention']`` populated truthy — Qwen3.5
       hybrid layer toggle.
    4. Substring match against :data:`MODELS_REQUIRING_FULL_CHECKPOINT`
       over both ``model_name`` and ``hf_path`` (case-insensitive).
       Picks up freshly-quantized community uploads that don't have
       an alias entry yet.

    Returns False on ``None``/empty inputs — disk checkpointing is
    best-effort and a "don't know, assume sliceable" answer just means
    we write the same full snapshot anyway (today both branches converge
    to a full write; the registry gates a future partial path).
    """
    if alias_metadata is not None:
        flag = alias_metadata.get("requires_full_checkpoint")
        if isinstance(flag, bool) and flag:
            return True

    if hf_config is not None:
        sw = hf_config.get("sliding_window")
        if isinstance(sw, int) and sw > 0:
            return True
        if hf_config.get("hybrid_attention"):
            return True

    needle = f"{model_name or ''} {hf_path or ''}".lower()
    return any(pat in needle for pat in MODELS_REQUIRING_FULL_CHECKPOINT)


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def should_checkpoint(
    num_tokens: int,
    last_checkpoint_at: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
) -> bool:
    """Return True when ``num_tokens`` has crossed the next boundary.

    Boundary semantics (locked by ``test_disk_kv_checkpoint.py``):

    - ``interval=0`` → never checkpoint. The CLI flag uses 0 as the
      disable sentinel; the helper honours it so callers don't have to
      add a separate gate at every call site.
    - ``num_tokens < interval`` → no checkpoint yet. The first boundary
      lands AT ``interval`` (so for the default 256: offsets 0..255 do
      nothing, 256 fires the first checkpoint, 257..511 stay quiet,
      512 fires the second, …).
    - ``num_tokens >= last_checkpoint_at + interval`` → fire. Using
      ``last_checkpoint_at`` rather than a strict ``% interval == 0``
      keeps the trigger correct even when the scheduler skips token
      counts (spec decode can advance by multiple tokens per step).
    - Negative / NaN tokens are floored to 0 (defensive — the scheduler
      caller already validates, but the unit test exercises this).
    """
    if interval <= 0:
        return False
    if not isinstance(num_tokens, int):
        # Be paranoid — a stray float from a user-supplied SamplingParams
        # field that survived Field validation could end up here.
        try:
            num_tokens = int(num_tokens)
        except (TypeError, ValueError):
            return False
    if num_tokens < 0:
        return False
    if num_tokens < interval:
        return False
    return num_tokens >= last_checkpoint_at + interval


def write_checkpoint(
    cache: list[Any],
    *,
    root: str,
    req_hash: str,
    token_offset: int,
    kv_dtype: str = "bf16",
    requires_full_checkpoint: bool = False,
    model_name: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    radix_index: Any | None = None,
) -> str | None:
    """Write one cache snapshot to disk at ``token_offset`` atomically.

    Path layout:
        <root>/<req_hash>/checkpoint-<token_offset>.safetensors
        <root>/<req_hash>/checkpoint-<token_offset>.json

    Atomicity contract:
    - The safetensors body is written to ``<...>.safetensors.tmp``,
      fsync'd, then atomically renamed into place. A SIGKILL between
      ``open`` and ``rename`` leaves only the .tmp file, which
      ``scan_checkpoints`` ignores AND clears on first visit.
    - The metadata JSON is written + fsync'd + renamed AFTER the
      safetensors rename so a partial commit can never expose a JSON
      that points at a missing body.

    Returns the safetensors path on success, or None when:
    - ``interval <= 0`` (caller already short-circuited via
      :func:`should_checkpoint`, but defensive)
    - the write failed before rename (logged, counters untouched —
      this is the "best-effort persistence" contract the in-process
      radix already uses)

    Args:
        cache: ``list`` of MLX-LM cache layers (KVCache /
            QuantizedKVCache / hybrid). Must round-trip through
            ``mlx_lm.save_prompt_cache``. The caller is responsible for
            using :func:`vllm_mlx.positioned_kv_cache.positioned_update_and_fetch`
            for any pre-checkpoint writes; passing a ``PositionedKVCache``
            subclass instance here would WORK at write time but FAIL at
            load time because ``mlx_lm.load_prompt_cache`` looks the
            class name up in the upstream module globals.
        root: directory containing per-request subdirs. Created on
            demand; survives across restarts.
        req_hash: short stable hash (see :func:`request_hash`).
        token_offset: number of tokens already in the cache. Used as
            both the filename suffix and the metadata field for the
            radix-index hand-off.
        kv_dtype: ``"bf16"``/``"int8"``/``"int4"`` — recorded in
            metadata for the loader's bookkeeping. Does NOT change the
            on-disk format; ``save_prompt_cache`` writes the cache
            class names regardless.
        requires_full_checkpoint: pre-resolved via
            :func:`model_requires_full_checkpoint`. Recorded in the
            metadata so the loader can refuse to restore a partial
            snapshot from a model family that needs full state.
        model_name: alias key or HF path. Recorded for observability.
        extra_metadata: free-form dict added to the JSON. Used by the
            radix hand-off to record the source token sequence hash.
        radix_index: optional radix-index handle; when provided AND
            the metadata carries a ``tokens_key`` list, the radix is
            notified via ``radix_index.insert(tokens_key)`` so the
            next prefix lookup can find the on-disk entry without a
            re-scan. Best-effort: any radix exception is logged and
            the write succeeds anyway.
    """
    # The CLI / scheduler caller already gates on
    # :func:`should_checkpoint`; the guard here is a belt for the
    # in-process radix path that pokes ``write_checkpoint`` directly.
    if not isinstance(token_offset, int) or token_offset < 0:
        return None
    if cache is None or not cache:
        return None

    with _DISK_LOCK:
        try:
            from mlx_lm.models.cache import save_prompt_cache
        except ImportError:  # pragma: no cover — every prod env has mlx_lm
            logger.warning("[disk_kv_checkpoint] mlx_lm not importable; skipping")
            return None

        dst_dir = os.path.join(root, req_hash)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = checkpoint_path(root, req_hash, token_offset)
        meta_path = metadata_path(root, req_hash, token_offset)
        # See ``_TMP_INFIX`` comment: the tmp path must end in
        # ``.safetensors`` or ``mx.save_safetensors`` will rewrite the
        # filename and the rename will fail.
        tmp_path = dst_path.replace(_CHECKPOINT_EXT, _TMP_INFIX + _CHECKPOINT_EXT)
        meta_tmp = meta_path + _TMP_INFIX

        # One ``save_uuid`` per write binds the three on-disk artifacts —
        # the safetensors body (embedded metadata below), the JSON sidecar,
        # and the tokens blob — into a single logical commit. A restore that
        # finds a body / sidecar / tokens triple whose uuids disagree knows
        # it stitched two different writes together and must re-prefill.
        # The scheduler may pre-mint one (so it can index the same uuid
        # elsewhere); otherwise we mint it here.
        save_uuid = None
        if extra_metadata:
            candidate = extra_metadata.get("save_uuid")
            if isinstance(candidate, str) and candidate:
                save_uuid = candidate
        if save_uuid is None:
            save_uuid = uuid.uuid4().hex

        # Exact prompt token ids this checkpoint covers, if the caller
        # handed them over. Persisted in the tokens blob (canonical, uuid-
        # bound) so a later restore can byte-verify the prefix. Kept
        # backward-safe: absent tokens just means no blob gets written.
        tokens_for_blob: list[int] | None = None
        if extra_metadata:
            raw_tokens = extra_metadata.get("tokens_key")
            if isinstance(raw_tokens, (list, tuple)) and raw_tokens:
                try:
                    tokens_for_blob = [int(t) for t in raw_tokens]
                except (TypeError, ValueError):
                    tokens_for_blob = None

        # Build the safetensors metadata that ships INSIDE the file.
        # ``save_prompt_cache`` requires str→str — JSON-encode the
        # boolean / int fields so the round-trip is faithful.
        st_meta = {
            "token_offset": str(token_offset),
            "kv_dtype": kv_dtype,
            "requires_full_checkpoint": "true" if requires_full_checkpoint else "false",
            "save_uuid": save_uuid,
        }
        if model_name:
            st_meta["model_name"] = str(model_name)

        try:
            save_prompt_cache(tmp_path, cache, metadata=st_meta)
            # Durably commit the body BEFORE the rename. Same rationale as
            # ``memory_cache.py`` R8-M7 codex r1 BLOCKING #3 — without
            # the fsync a SIGTERM-driven shutdown could leave a renamed
            # file with empty/partial contents on hard reset.
            _fsync_file(tmp_path)
            os.replace(tmp_path, dst_path)
            _fsync_dir(dst_dir)
        except Exception as e:
            logger.warning(
                f"[disk_kv_checkpoint] safetensors write failed at {dst_path!r}: {e}",
                exc_info=True,
            )
            # Best-effort cleanup so the next ``scan_checkpoints`` doesn't
            # see a stale .tmp.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return None

        # Tokens blob — the exact prompt token ids this checkpoint covers,
        # in the radix's v3 wire format (magic + count + save_uuid + int32
        # LE). Written AFTER the body but BEFORE the sidecar so the sidecar,
        # the loader's source of truth, is the last thing to land. A write
        # failure here is non-fatal: the body + sidecar stay valid, the blob
        # is simply absent, and a later restore falls back to re-prefill
        # rather than trusting an unverifiable cache.
        tokens_persisted = False
        tokens_count = 0
        if tokens_for_blob is not None:
            try:
                from vllm_mlx.memory_cache import _write_tokens_bin_v3
            except Exception as e:  # pragma: no cover — defensive
                logger.debug(
                    f"[disk_kv_checkpoint] tokens-blob writer unavailable: {e}"
                )
                _write_tokens_bin_v3 = None
            if _write_tokens_bin_v3 is not None:
                tok_path = tokens_path(root, req_hash, token_offset)
                tok_tmp = tok_path + _TMP_INFIX
                try:
                    _write_tokens_bin_v3(tok_tmp, tokens_for_blob, save_uuid)
                    _fsync_file(tok_tmp)
                    os.replace(tok_tmp, tok_path)
                    _fsync_dir(dst_dir)
                    tokens_persisted = True
                    tokens_count = len(tokens_for_blob)
                except Exception as e:
                    logger.warning(
                        f"[disk_kv_checkpoint] tokens-blob write failed at "
                        f"{tok_path!r}: {e}; checkpoint kept without token verify"
                    )
                    try:
                        os.unlink(tok_tmp)
                    except OSError:
                        pass

        # Sidecar JSON — written AFTER the safetensors so a torn shutdown
        # can never leave a JSON pointing at a missing body.
        meta_payload: dict[str, Any] = {
            "schema_version": 1,
            "token_offset": int(token_offset),
            "kv_dtype": str(kv_dtype),
            "requires_full_checkpoint": bool(requires_full_checkpoint),
            "model_name": model_name,
            "created_at": time.time(),
            "size_bytes": _safe_filesize(dst_path),
            "save_uuid": save_uuid,
            "has_tokens": tokens_persisted,
            "tokens_count": tokens_count,
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                if k in meta_payload:
                    # Don't let extra_metadata clobber the fields we own;
                    # silently skip rather than raise so a buggy caller
                    # can't tear the write down.
                    continue
                meta_payload[k] = v

        try:
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(meta_payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(meta_tmp, meta_path)
            _fsync_dir(dst_dir)
        except Exception as e:
            logger.warning(
                f"[disk_kv_checkpoint] metadata write failed at "
                f"{meta_path!r}: {e}; body left in place at {dst_path!r}"
            )
            try:
                os.unlink(meta_tmp)
            except OSError:
                pass
            # Body is still valid; the loader tolerates a missing
            # metadata sidecar (treats it as "unknown source").

        # Stats: writes++, bytes refreshed against the live filesystem so
        # we never report stale totals after an eviction.
        with _STATS_LOCK:
            _STATS.writes += 1
            _STATS.bytes = _measure_root_bytes(root)

        # Radix hand-off — best-effort, mirrors the in-process store path.
        if radix_index is not None and extra_metadata is not None:
            tokens_key = extra_metadata.get("tokens_key")
            if isinstance(tokens_key, (list, tuple)) and tokens_key:
                try:
                    radix_index.insert(list(tokens_key))
                except Exception as e:  # pragma: no cover — radix is optional
                    logger.debug(f"[disk_kv_checkpoint] radix.insert failed: {e}")

        # Content-index hand-off (R15-P2). When we persisted the exact prompt
        # tokens, register this checkpoint in the process-wide prompt→checkpoint
        # map so a LATER, differently-keyed request can find it by prefix
        # without a rescan. In-memory only; takes just the content-index lock.
        # We already hold ``_DISK_LOCK`` here, which is the legal outer→inner
        # nesting per the module LOCK ORDERING note (_DISK_LOCK > index lock).
        if tokens_persisted and tokens_for_blob is not None:
            try:
                get_content_index().index_checkpoint(
                    tokens_for_blob,
                    root=root,
                    req_hash=req_hash,
                    token_offset=token_offset,
                    save_uuid=save_uuid,
                )
            except Exception as e:  # pragma: no cover — index is best-effort
                logger.debug(
                    f"[disk_kv_checkpoint] content-index hand-off failed: {e}"
                )

        return dst_path


def maybe_write_checkpoint(
    cache: list[Any],
    *,
    root: str,
    req_hash: str,
    num_tokens: int,
    last_checkpoint_at: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    kv_dtype: str = "bf16",
    requires_full_checkpoint: bool = False,
    model_name: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    radix_index: Any | None = None,
) -> tuple[int, str | None]:
    """Convenience wrapper: gate via :func:`should_checkpoint`, then write.

    Returns ``(new_last_checkpoint_at, path_or_None)``:
    - ``new_last_checkpoint_at`` is the largest multiple of ``interval``
      that is ``<= num_tokens``. The scheduler stores this on the
      request so the next call doesn't re-fire.
    - ``path_or_None`` is the safetensors path on success, None when
      the gate was open but the write failed.

    Called once per scheduler step from the hook in
    :mod:`vllm_mlx.scheduler`.
    """
    if not should_checkpoint(num_tokens, last_checkpoint_at, interval):
        return last_checkpoint_at, None

    # Snap the new boundary to the largest multiple of ``interval`` that
    # is still ``<= num_tokens``. Without snapping, a step that advances
    # by N>interval (e.g. spec decode) would fire one checkpoint and
    # then re-fire on the next step because ``last_checkpoint_at`` only
    # bumped by interval, not by the actual gap.
    new_boundary = (num_tokens // interval) * interval

    path = write_checkpoint(
        cache,
        root=root,
        req_hash=req_hash,
        token_offset=new_boundary,
        kv_dtype=kv_dtype,
        requires_full_checkpoint=requires_full_checkpoint,
        model_name=model_name,
        extra_metadata=extra_metadata,
        radix_index=radix_index,
    )
    if path is None:
        # Don't advance the watermark when the write failed — the next
        # boundary still gets a try.
        return last_checkpoint_at, None
    return new_boundary, path


# ---------------------------------------------------------------------------
# Load + scan path
# ---------------------------------------------------------------------------


@dataclass
class LoadedCheckpoint:
    """Result of a successful :func:`load_checkpoint` call.

    Attributes:
        cache: ``list`` of MLX-LM cache layers ready to feed into the
            BatchGenerator's prompt cache slot.
        token_offset: number of tokens already in ``cache``.
        kv_dtype: ``"bf16"``/``"int8"``/``"int4"`` recorded at write
            time. Loader uses this to refuse a mismatched re-load if
            the operator switched ``--kv-cache-dtype`` between runs.
        requires_full_checkpoint: True when the source model is in
            :data:`MODELS_REQUIRING_FULL_CHECKPOINT`. The scheduler can
            use this to refuse a partial restore.
        metadata: sidecar JSON contents — free-form, useful for the
            radix-index hand-off.
        path: absolute safetensors path the cache came from.
    """

    cache: list[Any]
    token_offset: int
    kv_dtype: str
    requires_full_checkpoint: bool
    metadata: dict[str, Any]
    path: str


def load_checkpoint(path: str) -> LoadedCheckpoint | None:
    """Load one checkpoint by absolute safetensors path.

    Returns ``None`` and logs a warning when:
    - the file is missing / unreadable
    - ``mlx_lm.load_prompt_cache`` raises (corrupt body, class-name
      mismatch — the latter is the trap the disk format inherits from
      ``save_prompt_cache``)
    - the sidecar JSON is missing AND the safetensors metadata fails to
      decode

    Calls the ``loads`` counter on success.
    """
    with _DISK_LOCK:
        try:
            from mlx_lm.models.cache import load_prompt_cache
        except ImportError:  # pragma: no cover — every prod env has mlx_lm
            logger.warning("[disk_kv_checkpoint] mlx_lm not importable; skipping")
            return None

        if not os.path.isfile(path):
            return None

        try:
            cache, st_meta = load_prompt_cache(path, return_metadata=True)
        except Exception as e:
            logger.warning(
                f"[disk_kv_checkpoint] load_prompt_cache failed at {path!r}: {e}"
            )
            return None

        # Sidecar metadata is the source of truth; fall back to the
        # safetensors metadata if the sidecar went missing.
        meta_path_str = path.replace(_CHECKPOINT_EXT, _METADATA_EXT)
        sidecar: dict[str, Any] = {}
        if os.path.isfile(meta_path_str):
            try:
                with open(meta_path_str, encoding="utf-8") as fh:
                    sidecar = json.load(fh)
            except Exception as e:
                logger.warning(
                    f"[disk_kv_checkpoint] sidecar load failed at "
                    f"{meta_path_str!r}: {e}; falling back to embedded metadata"
                )

        # Schema guard — reject-and-reprefill on any doubt. A wrong restore
        # corrupts output silently (no exception), so we refuse a checkpoint
        # whose sidecar version we can't positively vouch for: a missing
        # sidecar / absent version (can't tell what wrote it) or a version
        # newer than this build knows (a later, format-shifted writer).
        version = sidecar.get("schema_version") if sidecar else None
        if not isinstance(version, int) or version > _KNOWN_SCHEMA_VERSION:
            logger.warning(
                f"[disk_kv_checkpoint] refusing checkpoint at {path!r}: "
                f"sidecar schema_version={version!r} is absent or newer than "
                f"supported ({_KNOWN_SCHEMA_VERSION}); will re-prefill"
            )
            return None

        # The embedded metadata is str→str; coerce safely.
        embedded = st_meta or {}
        token_offset = int(
            sidecar.get("token_offset")
            if sidecar.get("token_offset") is not None
            else embedded.get("token_offset", 0) or 0
        )
        kv_dtype = str(
            sidecar.get("kv_dtype") or embedded.get("kv_dtype", "bf16") or "bf16"
        )
        requires_full = bool(
            sidecar.get("requires_full_checkpoint")
            if "requires_full_checkpoint" in sidecar
            else (
                str(embedded.get("requires_full_checkpoint", "false")).lower() == "true"
            )
        )

        with _STATS_LOCK:
            _STATS.loads += 1

        return LoadedCheckpoint(
            cache=cache,
            token_offset=token_offset,
            kv_dtype=kv_dtype,
            requires_full_checkpoint=requires_full,
            metadata=sidecar,
            path=path,
        )


def scan_checkpoints(root: str) -> list[tuple[str, float, int]]:
    """Return ``[(path, mtime, size_bytes), …]`` for every committed checkpoint.

    Cleans up stale ``.tmp`` files as a side effect (a SIGKILL between
    the safetensors write and rename leaves them; they're never
    recoverable so erasing them is strictly safe).

    Used by:
    - The disk-cap eviction loop in :func:`enforce_disk_cap`.
    - The startup loader hand-off in
      :mod:`vllm_mlx.runtime.cache` (a future iteration; today the loader
      is gated on memory-aware cache presence and disk checkpoints aren't
      auto-loaded back into a fresh engine).
    """
    with _DISK_LOCK:
        if not os.path.isdir(root):
            return []

        out: list[tuple[str, float, int]] = []
        # Tmp suffix shapes:
        #   <name>.tmp.safetensors  — safetensors body tmp
        #   <name>.json.tmp         — sidecar JSON tmp
        # Both are stale on rescan and must be cleaned up.
        tmp_body_marker = _TMP_INFIX + _CHECKPOINT_EXT  # e.g. ".tmp.safetensors"
        tmp_json_marker = _METADATA_EXT + _TMP_INFIX  # e.g. ".json.tmp"
        tmp_tokens_marker = _TOKENS_EXT + _TMP_INFIX  # e.g. ".tokens.bin.tmp"
        for entry in os.scandir(root):
            if not entry.is_dir(follow_symlinks=False):
                continue
            try:
                for child in os.scandir(entry.path):
                    name = child.name
                    if (
                        name.endswith(tmp_body_marker)
                        or name.endswith(tmp_json_marker)
                        or name.endswith(tmp_tokens_marker)
                    ):
                        # Stale tmp from a torn write — best-effort cleanup.
                        try:
                            os.unlink(child.path)
                        except OSError:
                            pass
                        continue
                    if not name.endswith(_CHECKPOINT_EXT):
                        continue
                    try:
                        stat = child.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    out.append((child.path, stat.st_mtime, stat.st_size))
            except OSError:
                # Per-request dir vanished mid-scan — fine, move on.
                continue

        out.sort(key=lambda row: row[1])
        return out


def enforce_disk_cap(root: str, *, max_bytes: int | None = None) -> tuple[int, int]:
    """Evict oldest checkpoints until the on-disk total fits in ``max_bytes``.

    Returns ``(num_evicted, bytes_remaining)`` for the caller's log line.
    ``max_bytes`` defaults to :func:`resolve_max_disk_bytes`; pass ``0``
    to skip the cap (escape hatch — operators on a big disk who don't
    want eviction at all). NaN-safe: any non-finite float is clamped to
    the default.
    """
    if max_bytes is None:
        max_bytes = resolve_max_disk_bytes()
    elif isinstance(max_bytes, float) and not math.isfinite(max_bytes):
        # NaN/Inf coercion — Pydantic Field(ge=) does NOT reject these,
        # so the validation has to happen here for any user-input float
        # that survived the schema layer.
        max_bytes = resolve_max_disk_bytes()
    max_bytes = max(0, int(max_bytes))

    with _DISK_LOCK:
        entries = scan_checkpoints(root)
        total = sum(size for _, _, size in entries)
        if max_bytes == 0 or total <= max_bytes:
            with _STATS_LOCK:
                _STATS.bytes = total
            return 0, total

        evicted = 0
        for path, _mtime, size in entries:
            if total <= max_bytes:
                break
            try:
                os.unlink(path)
            except OSError as e:
                logger.warning(
                    f"[disk_kv_checkpoint] eviction unlink({path!r}) failed: {e}"
                )
                continue
            sidecar = path.replace(_CHECKPOINT_EXT, _METADATA_EXT)
            try:
                os.unlink(sidecar)
            except OSError:
                pass
            # Drop the paired tokens blob too so eviction doesn't strand it
            # (it's not counted in the byte total — scan only sums
            # safetensors — but leaving it would keep the parent dir from
            # being pruned and orphan a token list with no cache).
            tok_blob = path.replace(_CHECKPOINT_EXT, _TOKENS_EXT)
            try:
                os.unlink(tok_blob)
            except OSError:
                pass
            total -= size
            evicted += 1
            # Best-effort prune of the parent directory if the eviction
            # just emptied it. Keeps the scan loop cheap on long-running
            # servers.
            parent = os.path.dirname(path)
            try:
                if not os.listdir(parent):
                    os.rmdir(parent)
            except OSError:
                pass

        with _STATS_LOCK:
            _STATS.evictions += evicted
            _STATS.bytes = total

        return evicted, total


# ---------------------------------------------------------------------------
# Prompt → checkpoint content index (R15-P2, task #297)
# ---------------------------------------------------------------------------
#
# The write path keys every checkpoint dir by ``request_hash`` =
# sha256(model::request_id), which is fine for the SAME request resuming but
# useless for a brand-new request that happens to share a prefix (the Cursor
# / Claude-Code shared-system-prompt workload). This index closes that gap:
# it maps ``tuple(prompt_token_ids)`` → the checkpoint whose persisted tokens
# are that exact prefix, so ``add_request`` can find a restore candidate by
# CONTENT rather than by request id.
#
# It is populated two ways, both feeding the same map:
#   * write-time hand-off — :func:`write_checkpoint` calls
#     :meth:`DiskCheckpointIndex.index_checkpoint` right after it persists the
#     tokens blob (no rescan needed for freshly-written checkpoints);
#   * boot-time scan — :meth:`DiskCheckpointIndex.build_from_root` walks
#     :func:`scan_checkpoints`, reads each checkpoint's persisted tokens blob,
#     and indexes it (recovers checkpoints written by a previous process).
#
# Lookup returns the checkpoint with the LARGEST verified ``token_offset``
# that is a true prefix of the incoming request, after a byte-level re-verify
# of the on-disk tokens blob (reject-and-reprefill on ANY doubt — a wrong
# restore corrupts output silently).


@dataclass(frozen=True)
class _CheckpointRef:
    """Immutable pointer from a prompt prefix to its on-disk checkpoint.

    Stored as the value side of :class:`DiskCheckpointIndex._by_key`. Frozen
    so a concurrent reader can hold a reference without it changing under
    them. Carries everything :meth:`DiskCheckpointIndex.lookup` needs to
    reconstruct the paths (``root`` + ``req_hash`` + ``token_offset``) and to
    bind the triple of on-disk artifacts (``save_uuid``).
    """

    path: str
    root: str
    req_hash: str
    token_offset: int
    save_uuid: str | None


class DiskCheckpointIndex:
    """Process-wide prompt→checkpoint map for restore-on-miss lookup.

    Wraps a :class:`vllm_mlx.runtime.radix_index.RadixPrefixIndex` for the
    O(prefix_len) longest-prefix walk and a side dict that carries the
    checkpoint location the radix key alone can't (the radix only knows
    "this token tuple is stored", not where its safetensors lives).

    Concurrency: guarded by its own ``RLock``. Per the module-level LOCK
    ORDERING note, this lock is the INNERMOST of the three checkpoint locks —
    no method takes ``_DISK_LOCK`` (or the prefix-cache lock) while holding
    it. :meth:`lookup` and :meth:`build_from_root` therefore split into a
    disk phase (holds ``_DISK_LOCK``, not the index lock) and an index phase
    (holds the index lock, no disk I/O). The only legal nesting is the
    reverse — :func:`write_checkpoint` holds ``_DISK_LOCK`` and calls
    :meth:`index_checkpoint`, which takes only this lock.
    """

    def __init__(self) -> None:
        from .radix_index import RadixPrefixIndex

        self._lock = threading.RLock()
        self._radix = RadixPrefixIndex()
        self._by_key: dict[tuple[int, ...], _CheckpointRef] = {}

    # ------------------------------------------------------------------ #
    # Population                                                         #
    # ------------------------------------------------------------------ #

    def index_checkpoint(
        self,
        tokens: list[int] | tuple[int, ...],
        *,
        root: str,
        req_hash: str,
        token_offset: int,
        save_uuid: str | None = None,
    ) -> bool:
        """Register one freshly-written checkpoint. In-memory only.

        Safe to call while holding ``_DISK_LOCK`` (does no file I/O). The
        persisted-token length MUST equal ``token_offset`` — a checkpoint
        whose tokens blob doesn't cover exactly the offset it claims can't be
        trusted as a prefix, so we skip it rather than index an unverifiable
        entry. Newest write for a given key wins (a re-run of the same prefix
        points at the most recent, still-present file).

        Returns True when the entry was indexed, False when skipped.
        """
        if not tokens:
            return False
        try:
            key = tuple(int(t) for t in tokens)
        except (TypeError, ValueError):
            return False
        if not isinstance(token_offset, int) or token_offset <= 0:
            return False
        if len(key) != token_offset:
            return False
        ref = _CheckpointRef(
            path=checkpoint_path(root, req_hash, token_offset),
            root=root,
            req_hash=req_hash,
            token_offset=token_offset,
            save_uuid=save_uuid,
        )
        with self._lock:
            if key not in self._by_key:
                self._radix.insert(key)
            self._by_key[key] = ref
        return True

    def build_from_root(self, root: str) -> int:
        """Populate the index by scanning every checkpoint under ``root``.

        Two-phase to honour the lock order: the disk phase enumerates and
        reads token blobs (taking ``_DISK_LOCK`` inside the helpers, never
        the index lock); the index phase takes the index lock and bulk-loads
        the collected entries. Returns the number of checkpoints indexed.

        Best-effort: a checkpoint with no tokens blob, a failing schema
        guard, or a blob whose length disagrees with its filename offset is
        skipped (it simply won't be a restore candidate — the request
        re-prefills).
        """
        # --- disk phase (index lock NOT held) ---
        collected: list[tuple[tuple[int, ...], _CheckpointRef]] = []
        for path, _mtime, _size in scan_checkpoints(root):
            parsed = _parse_checkpoint_path(path)
            if parsed is None:
                continue
            req_hash, token_offset = parsed
            tokens, save_uuid = _read_checkpoint_tokens(path)
            if tokens is None:
                continue
            if len(tokens) != token_offset:
                continue
            key = tuple(tokens)
            collected.append(
                (
                    key,
                    _CheckpointRef(
                        path=path,
                        root=root,
                        req_hash=req_hash,
                        token_offset=token_offset,
                        save_uuid=save_uuid,
                    ),
                )
            )

        # --- index phase (index lock held, no disk I/O) ---
        indexed = 0
        with self._lock:
            for key, ref in collected:
                if key not in self._by_key:
                    self._radix.insert(key)
                self._by_key[key] = ref
                indexed += 1
        return indexed

    def forget_request(self, req_hash: str) -> int:
        """Drop every indexed entry belonging to one request hash.

        Called from :func:`cleanup_request` so a finished/evicted request
        doesn't leave a dangling map entry. Stale entries are already
        invalidation-safe at :meth:`lookup` (they fail the on-disk re-verify),
        so this is a footprint optimization, not a correctness requirement.
        In-memory only; takes just the index lock.
        """
        with self._lock:
            doomed = [k for k, ref in self._by_key.items() if ref.req_hash == req_hash]
            for key in doomed:
                del self._by_key[key]
                self._radix.remove(key)
            return len(doomed)

    def clear(self) -> None:
        """Reset the map (test hook / reindex)."""
        with self._lock:
            self._radix.clear()
            self._by_key.clear()

    # ------------------------------------------------------------------ #
    # Lookup                                                             #
    # ------------------------------------------------------------------ #

    def lookup(self, query_tokens: list[int] | tuple[int, ...]) -> LoadedCheckpoint | None:
        """Return the best restore candidate for ``query_tokens``, or None.

        "Best" = the checkpoint with the LARGEST ``token_offset`` whose
        persisted tokens are a true prefix of ``query_tokens``. Because every
        indexed key's length equals its ``token_offset``, the radix's
        longest-prefix walk yields exactly that checkpoint.

        Every uncertainty is a None (re-prefill) — a wrong restore corrupts
        output silently, so we only return a cache we could byte-verify:

        1. radix miss / no side-map entry → None;
        2. the matched key isn't a true prefix of the query, or its length
           disagrees with the claimed offset → None;
        3. the on-disk tokens blob doesn't re-read byte-identically to the
           matched key (with the uuid binding) → None;
        4. ``load_checkpoint`` fails its own schema guard, or the loaded
           cache's live offset disagrees with the claimed offset → None.

        Lock discipline: phase 1 resolves the in-memory ref under the index
        lock and releases it; phase 2 does all disk work with the index lock
        dropped (see the module LOCK ORDERING note). No lock is held on
        return, so the caller can forward the cache freely.
        """
        if not query_tokens:
            return None
        try:
            query = [int(t) for t in query_tokens]
        except (TypeError, ValueError):
            return None

        # --- phase 1: in-memory resolution (index lock only) ---
        with self._lock:
            _matched, key = self._radix.longest_prefix(query)
            if key is None:
                return None
            ref = self._by_key.get(key)
        if ref is None:
            return None

        offset = ref.token_offset
        if offset <= 0 or offset != len(key):
            return None
        if list(key) != query[:offset]:
            return None

        # --- phase 2: disk verify + materialise (index lock dropped) ---
        tok_path = tokens_path(ref.root, ref.req_hash, offset)
        if not _verify_tokens_blob(tok_path, key, ref.save_uuid):
            return None
        loaded = load_checkpoint(ref.path)
        if loaded is None:
            return None
        if loaded.token_offset != offset:
            return None
        if not _cache_offset_matches(loaded.cache, offset):
            return None
        return loaded

    def stats(self) -> dict[str, int]:
        """Snapshot of index size for /metrics folding."""
        with self._lock:
            return {
                "content_index_entries": len(self._by_key),
                "content_index_nodes": self._radix.stats().get("node_count", 0),
            }


def get_content_index() -> DiskCheckpointIndex:
    """Return the process-wide :class:`DiskCheckpointIndex` singleton.

    Constructed lazily on first use so a server that never enables disk
    checkpointing never builds it (the write hand-off and the scheduler
    lookup are the only callers, both gated on the off-by-default interval).
    """
    global _CONTENT_INDEX
    with _CONTENT_INDEX_LOCK:
        if _CONTENT_INDEX is None:
            _CONTENT_INDEX = DiskCheckpointIndex()
        return _CONTENT_INDEX


def reset_content_index_for_tests() -> None:
    """Test-only: drop the singleton so the next call rebuilds it fresh."""
    global _CONTENT_INDEX
    with _CONTENT_INDEX_LOCK:
        _CONTENT_INDEX = None


def build_content_index(root: str | None = None) -> int:
    """Populate the singleton content index by scanning ``root``.

    The one-liner a boot path (Phase 3 restore-on-miss) calls to recover the
    prompt→checkpoint map from checkpoints a previous process left on disk.
    Defaults to :func:`get_default_root`. Returns the number of checkpoints
    indexed. Cheap no-op when the root doesn't exist yet (an operator who
    never enabled disk checkpointing).
    """
    if root is None:
        root = get_default_root()
    return get_content_index().build_from_root(root)


def _parse_checkpoint_path(path: str) -> tuple[str, int] | None:
    """Recover ``(req_hash, token_offset)`` from a checkpoint safetensors path.

    Layout is ``<root>/<req_hash>/checkpoint-<offset>.safetensors`` — see
    :func:`checkpoint_path`. Returns None on any shape we don't recognise so
    a stray file in the root can't crash the index build.
    """
    if not path.endswith(_CHECKPOINT_EXT):
        return None
    base = os.path.basename(path)[: -len(_CHECKPOINT_EXT)]
    if not base.startswith("checkpoint-"):
        return None
    try:
        offset = int(base[len("checkpoint-") :])
    except ValueError:
        return None
    req_hash = os.path.basename(os.path.dirname(path))
    if not req_hash:
        return None
    return req_hash, offset


def _read_checkpoint_tokens(path: str) -> tuple[list[int] | None, str | None]:
    """Read a checkpoint's persisted prompt tokens for indexing.

    Returns ``(tokens, save_uuid)`` or ``(None, None)`` when the checkpoint
    has no tokens blob, fails the sidecar schema guard, or the blob doesn't
    read cleanly. Takes ``_DISK_LOCK`` for the file reads (never called with
    the index lock held — see LOCK ORDERING).
    """
    tok_path = path.replace(_CHECKPOINT_EXT, _TOKENS_EXT)
    meta_path_str = path.replace(_CHECKPOINT_EXT, _METADATA_EXT)
    with _DISK_LOCK:
        if not os.path.isfile(tok_path):
            return None, None
        sidecar: dict[str, Any] = {}
        if os.path.isfile(meta_path_str):
            try:
                with open(meta_path_str, encoding="utf-8") as fh:
                    sidecar = json.load(fh)
            except Exception:
                return None, None
        # Same schema guard as load_checkpoint: reject absent/newer versions.
        version = sidecar.get("schema_version") if sidecar else None
        if not isinstance(version, int) or version > _KNOWN_SCHEMA_VERSION:
            return None, None
        save_uuid = sidecar.get("save_uuid")
        save_uuid = save_uuid if isinstance(save_uuid, str) and save_uuid else None
        try:
            from vllm_mlx.memory_cache import (
                _peek_tokens_bin_header,
                _read_tokens_bin,
            )
        except Exception:  # pragma: no cover — defensive
            return None, None
        count, blob_uuid, reason = _peek_tokens_bin_header(tok_path)
        if reason or count is None:
            return None, None
        tokens, reason = _read_tokens_bin(tok_path, count, save_uuid)
        if reason or tokens is None:
            return None, None
    return tokens, save_uuid or blob_uuid


def _verify_tokens_blob(
    tok_path: str,
    expected: tuple[int, ...],
    expected_uuid: str | None,
) -> bool:
    """Byte-verify an on-disk tokens blob equals ``expected``.

    The last gate before a restore trusts a cache: re-reads the tokens blob
    (enforcing the uuid binding when present) and confirms it matches the
    matched key exactly. Any mismatch / read failure returns False so the
    caller re-prefills. Takes ``_DISK_LOCK`` (index lock not held).
    """
    with _DISK_LOCK:
        if not os.path.isfile(tok_path):
            return False
        try:
            from vllm_mlx.memory_cache import _read_tokens_bin
        except Exception:  # pragma: no cover — defensive
            return False
        try:
            tokens, reason = _read_tokens_bin(tok_path, len(expected), expected_uuid)
        except Exception:
            return False
    if reason or tokens is None:
        return False
    return tokens == list(expected)


def _cache_offset_matches(cache: list[Any], offset: int) -> bool:
    """Reject a materialised cache whose live length disagrees with ``offset``.

    A checkpoint written when a step overshot the boundary (e.g. spec decode
    advancing several tokens at once) can hold MORE KV than the ``offset`` its
    tokens blob covers. Restoring it as an ``offset``-length prefix would
    silently feed the model extra state, corrupting output. We cross-check
    every attention layer that exposes an integer ``offset`` attribute; if any
    disagrees, reject. Recurrent layers with no ``offset`` are skipped — the
    tokens-blob byte match already vouched for the prefix identity.
    """
    for layer in cache:
        off = getattr(layer, "offset", None)
        if isinstance(off, int) and off != offset:
            return False
    return True


# ---------------------------------------------------------------------------
# Per-request bookkeeping helpers (for the scheduler)
# ---------------------------------------------------------------------------


@dataclass
class RequestCheckpointState:
    """In-memory bookkeeping the scheduler stores per-request.

    Carries the last successfully-written boundary so
    :func:`should_checkpoint` can stay stateless. Optional fields are
    populated by the boot path:

    Attributes:
        req_hash: stable hash from :func:`request_hash`. Cached so the
            hot path doesn't re-hash on every step.
        interval: per-request override (defaults to the CLI flag value).
            ``0`` disables disk checkpointing for THIS request only.
        last_checkpoint_at: number of tokens already on disk for this
            request. Bumped by :func:`maybe_write_checkpoint`.
        requires_full_checkpoint: pre-resolved via
            :func:`model_requires_full_checkpoint`. Passed through to
            the writer at every boundary.
        kv_dtype: ``"bf16"`` / ``"int8"`` / ``"int4"`` — recorded in
            metadata so the loader can refuse a mismatched re-load.
    """

    req_hash: str
    interval: int = DEFAULT_CHECKPOINT_INTERVAL
    last_checkpoint_at: int = 0
    requires_full_checkpoint: bool = False
    kv_dtype: str = "bf16"
    model_name: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _fsync_file(path: str) -> None:
    """fsync a file by path, swallowing errors the caller will retry."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """fsync a directory so the rename is durable on hard reset.

    Linux requires an explicit dir-fsync after a rename for the new
    name to survive power loss; macOS (HFS+/APFS) handles this within
    the rename syscall but the extra call is cheap and matches the
    cross-platform contract the prefix cache already uses.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # macOS sometimes refuses fsync on a dir descriptor — non-fatal.
        pass
    finally:
        os.close(fd)


def _safe_filesize(path: str) -> int:
    """Return the byte size of ``path``, or 0 if the stat fails.

    Used only for the sidecar metadata, so a missed read just means
    the JSON records 0 — the actual disk-cap accounting uses
    ``scan_checkpoints`` and never trusts the sidecar value.
    """
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _measure_root_bytes(root: str) -> int:
    """Return the total live bytes under the checkpoint root.

    Cheap O(N) scan via ``scan_checkpoints``; called only after a
    successful write or eviction so it's amortized across the
    256-token boundary cadence, not the per-step hot path.
    """
    try:
        return sum(size for _, _, size in scan_checkpoints(root))
    except Exception:  # pragma: no cover — defensive
        return 0


def cleanup_request(root: str, req_hash: str) -> int:
    """Drop every checkpoint for one request (e.g. on completion).

    Returns the number of files removed. Best-effort — partial cleanup
    is fine, the next ``enforce_disk_cap`` pass will mop up.

    Called by the scheduler when a request finishes / errors out so the
    on-disk footprint matches the live request set.
    """
    with _DISK_LOCK:
        dir_path = os.path.join(root, req_hash)
        if not os.path.isdir(dir_path):
            return 0
        try:
            n = sum(1 for _ in os.scandir(dir_path))
        except OSError:
            n = 0
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
        except Exception:  # pragma: no cover — defensive
            return 0
        with _STATS_LOCK:
            _STATS.bytes = _measure_root_bytes(root)
    # Drop the request's content-index entries AFTER releasing _DISK_LOCK so
    # the index lock is never taken while a disk lock is held in the reverse
    # of the module LOCK ORDERING (index lock is the innermost; here it is
    # taken as a leaf with no other lock held). Stale entries would fail the
    # lookup re-verify anyway, so this is a footprint cleanup, not a
    # correctness gate.
    if _CONTENT_INDEX is not None:
        try:
            _CONTENT_INDEX.forget_request(req_hash)
        except Exception as e:  # pragma: no cover — best-effort
            logger.debug(f"[disk_kv_checkpoint] content-index forget failed: {e}")
    return n


# ---------------------------------------------------------------------------
# Test-only: deterministic root override
# ---------------------------------------------------------------------------


def temporary_root() -> str:
    """Return a fresh temporary checkpoint root (unit-test helper).

    Used by the disk-checkpoint tests so they don't pollute
    ``~/.cache/rapid-mlx/`` and don't race against any other agent. The
    caller is responsible for ``shutil.rmtree`` cleanup; using
    ``tempfile.TemporaryDirectory`` is cleaner in test fixtures.
    """
    return tempfile.mkdtemp(prefix="rapid-mlx-kv-checkpoint-")


__all__ = [
    "CheckpointStats",
    "DEFAULT_CHECKPOINT_INTERVAL",
    "DEFAULT_MAX_DISK_BYTES",
    "RESTORE_REJECT_REASONS",
    "DiskCheckpointIndex",
    "LoadedCheckpoint",
    "MODELS_REQUIRING_FULL_CHECKPOINT",
    "RequestCheckpointState",
    "build_content_index",
    "checkpoint_path",
    "cleanup_request",
    "enforce_disk_cap",
    "get_content_index",
    "get_default_root",
    "get_stats",
    "load_checkpoint",
    "maybe_write_checkpoint",
    "metadata_path",
    "model_requires_full_checkpoint",
    "record_hook_error",
    "record_restore_reject",
    "request_hash",
    "reset_content_index_for_tests",
    "reset_stats_for_tests",
    "resolve_max_disk_bytes",
    "scan_checkpoints",
    "should_checkpoint",
    "temporary_root",
    "tokens_path",
    "write_checkpoint",
]
