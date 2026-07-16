# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disk-backed KV checkpointing (R15-P1 task #296).

These tests pin the contract for :mod:`vllm_mlx.runtime.disk_kv_checkpoint`,
the disk-backed long-context partner of the in-process radix prefix cache.

The on-disk format is ``mlx_lm.save_prompt_cache`` /
``load_prompt_cache``, so the round-trip guards both:

* Plain ``KVCache`` (bf16) — the legacy default before R15 #300.
* ``QuantizedKVCache`` (int4) — the new R15 #300 default after PR #910.

Trigger / atomicity / eviction tests do not touch MLX at all; they exercise
the gating + filesystem layer with tiny on-disk blobs so the suite stays
fast and CPU-only (the Stage B PonyExl3 Viterbi conversion is currently
holding the GPU; the agent run cannot boot ``qmlx serve``).

Run with::

    pytest tests/test_disk_kv_checkpoint.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402

from vllm_mlx.runtime import disk_kv_checkpoint as _dkc  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> str:
    """Return a fresh checkpoint root and reset the module counters."""
    _dkc.reset_stats_for_tests()
    return str(tmp_path / "ckpt-root")


def _seed_kv_cache(num_tokens: int = 32) -> list[KVCache]:
    """Return a one-layer prompt cache prefilled with ``num_tokens`` rows.

    The keys / values are random under a fixed seed so a byte-identical
    round-trip assertion is well-defined.
    """
    cache = KVCache()
    k = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(0))
    v = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(1))
    cache.update_and_fetch(k, v)
    return [cache]


def _seed_quant_cache(num_tokens: int = 32) -> list[QuantizedKVCache]:
    """Return a one-layer QuantizedKVCache (int4) prefilled with ``num_tokens``.

    Matches the post-R15 #300 default. ``mlx_lm.save_prompt_cache``
    handles the quantized cache via the same metadata-driven loader; the
    round-trip is byte-identical at the packed-uint32 layer.
    """
    cache = QuantizedKVCache(group_size=64, bits=4)
    k = mx.random.normal((1, 2, num_tokens, 64), key=mx.random.key(2))
    v = mx.random.normal((1, 2, num_tokens, 64), key=mx.random.key(3))
    cache.update_and_fetch(k, v)
    return [cache]


# ---------------------------------------------------------------------------
# Boundary trigger logic — 256-tok intervals
# ---------------------------------------------------------------------------


def test_should_checkpoint_token_offsets_0_to_255_do_not_fire():
    """Below the first boundary, ``should_checkpoint`` must return False.

    Locks the 0-255 → no checkpoint, 256+ → first checkpoint, 512+ →
    second checkpoint progression the task brief calls out.
    """
    for n in (0, 1, 128, 255):
        assert not _dkc.should_checkpoint(n, last_checkpoint_at=0)


def test_should_checkpoint_fires_at_first_256_boundary():
    """At token offset 256 the first boundary fires; 257..511 stay quiet."""
    assert _dkc.should_checkpoint(256, last_checkpoint_at=0)
    # After the first checkpoint at offset 256, the next 255 tokens are
    # silent again.
    for n in (257, 400, 511):
        assert not _dkc.should_checkpoint(n, last_checkpoint_at=256)


def test_should_checkpoint_fires_at_second_512_boundary():
    """At token offset 512 the second boundary fires."""
    assert _dkc.should_checkpoint(512, last_checkpoint_at=256)


def test_should_checkpoint_interval_zero_disables():
    """``interval=0`` is the disable sentinel and must never fire."""
    for n in (0, 256, 1024, 9999):
        assert not _dkc.should_checkpoint(n, last_checkpoint_at=0, interval=0)


def test_should_checkpoint_handles_skip_tokens_in_spec_decode():
    """Spec decode advances by N>1 tokens per step. The trigger must fire
    once when the step crosses the boundary, then stay quiet until the
    NEXT boundary even if the gap was larger than ``interval``.
    """
    # Step advances from 200 → 320 (jumped past 256). Trigger must fire.
    assert _dkc.should_checkpoint(320, last_checkpoint_at=0)
    # After the writer snaps the watermark to 256 (largest multiple ≤
    # num_tokens), the next 256 tokens must NOT re-fire.
    assert not _dkc.should_checkpoint(320, last_checkpoint_at=256)
    assert not _dkc.should_checkpoint(500, last_checkpoint_at=256)


def test_should_checkpoint_negative_tokens_are_safe():
    """Negative / non-int tokens must not raise; should_checkpoint returns
    False so a buggy caller can't crash the decode path.
    """
    assert not _dkc.should_checkpoint(-1, last_checkpoint_at=0)
    assert not _dkc.should_checkpoint("not-an-int", last_checkpoint_at=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Roundtrip — KVCache (bf16) byte-identical
# ---------------------------------------------------------------------------


def test_roundtrip_bf16_kv_cache_byte_identical(root: str):
    """Write a KVCache to disk, reload it, assert byte-identical state.

    This is the headline correctness test — the prefix cache already
    proves ``save_prompt_cache`` round-trips, but the new module sits
    in front of it so the integration must also pass.
    """
    cache_in = _seed_kv_cache(num_tokens=64)
    req_hash = _dkc.request_hash("req-rt-bf16", model_name="qwen3-test")

    path = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=64,
        kv_dtype="bf16",
        model_name="qwen3-test",
    )
    assert path is not None
    assert os.path.isfile(path)

    loaded = _dkc.load_checkpoint(path)
    assert loaded is not None
    assert loaded.token_offset == 64
    assert loaded.kv_dtype == "bf16"
    assert len(loaded.cache) == 1

    # Byte-identical keys/values check on the underlying mx.arrays.
    k_in = cache_in[0].state[0]
    v_in = cache_in[0].state[1]
    k_out = loaded.cache[0].state[0]
    v_out = loaded.cache[0].state[1]
    assert mx.array_equal(k_in, k_out).item()
    assert mx.array_equal(v_in, v_out).item()


def test_roundtrip_int4_quantized_kv_cache(root: str):
    """Write a QuantizedKVCache to disk, reload, assert state matches.

    R15 #300 / PR #910 made ``--kv-cache-dtype int4`` the default; the
    disk checkpoint module must round-trip the packed (uint32, scales,
    biases) triple losslessly. Equality is checked on the packed
    representation — dequantizing introduces small numerical noise that
    isn't part of the disk contract.
    """
    cache_in = _seed_quant_cache(num_tokens=64)
    req_hash = _dkc.request_hash("req-rt-int4", model_name="qwen3-int4")

    path = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=64,
        kv_dtype="int4",
        model_name="qwen3-int4",
    )
    assert path is not None

    loaded = _dkc.load_checkpoint(path)
    assert loaded is not None
    assert loaded.kv_dtype == "int4"
    assert isinstance(loaded.cache[0], QuantizedKVCache)

    # Quantized cache stores .state as a 2-tuple of (K, V) where each is
    # itself a (packed_uint32, scales, biases) 3-tuple. Walk both levels
    # and assert byte-identical on every leaf array.
    s_in = cache_in[0].state
    s_out = loaded.cache[0].state
    assert len(s_in) == len(s_out)
    for kv_in, kv_out in zip(s_in, s_out):
        assert len(kv_in) == len(kv_out)
        for a, b in zip(kv_in, kv_out):
            assert mx.array_equal(a, b).item()


# ---------------------------------------------------------------------------
# Atomic write semantics — partial files must be ignored on rescan
# ---------------------------------------------------------------------------


def test_atomic_write_partial_tmp_is_ignored(root: str):
    """A leftover .tmp file from a torn write must not be loadable.

    Simulates SIGKILL between the safetensors write and rename: the
    .tmp file is on disk but the .safetensors target doesn't exist.
    ``scan_checkpoints`` must skip it AND clean it up so the next pass
    doesn't see a stale tmp.
    """
    req_hash = _dkc.request_hash("req-atomic", model_name="m")
    dst_dir = os.path.join(root, req_hash)
    os.makedirs(dst_dir, exist_ok=True)
    # Tmp filename shape matches the writer (see ``_TMP_INFIX`` doc on
    # the runtime module — mlx.core.save_safetensors auto-appends
    # ``.safetensors`` so the tmp file must end in that suffix).
    tmp_path = os.path.join(dst_dir, "checkpoint-256.tmp.safetensors")
    # Write a "partial" body that obviously isn't a valid safetensors.
    with open(tmp_path, "wb") as fh:
        fh.write(b"\x00" * 64)

    # scan_checkpoints should not return the tmp.
    rows = _dkc.scan_checkpoints(root)
    assert rows == []
    # And the cleanup should have removed it.
    assert not os.path.exists(tmp_path)


def test_atomic_write_committed_file_survives_scan(root: str):
    """A fully-committed checkpoint survives a scan and is loadable."""
    cache_in = _seed_kv_cache(num_tokens=16)
    req_hash = _dkc.request_hash("req-survive", model_name="m")
    path = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    rows = _dkc.scan_checkpoints(root)
    assert len(rows) == 1
    assert rows[0][0] == path
    # Load round-trips.
    loaded = _dkc.load_checkpoint(path)
    assert loaded is not None and loaded.token_offset == 256


# ---------------------------------------------------------------------------
# Sliding-window model handling (Gemma 4)
# ---------------------------------------------------------------------------


def test_sliding_window_model_detection_by_name():
    """Gemma 4 substring detection (the alias / HF path family glob)."""
    assert _dkc.model_requires_full_checkpoint("gemma-4-12b-int4")
    assert _dkc.model_requires_full_checkpoint("mlx-community/Gemma-4-2B-it")
    # Gemma 3 must NOT trip the full-checkpoint flag — it's covered by
    # the kv_cache_dtype safelist (auto-downgrade to bf16), not by this
    # registry. Mixing the two would over-checkpoint and slow long-run
    # serve.
    assert not _dkc.model_requires_full_checkpoint("gemma-3-27b-4bit")
    # A model with no name and no signals defaults to False.
    assert not _dkc.model_requires_full_checkpoint(None)


def test_sliding_window_model_detection_by_hf_config():
    """A model that doesn't match the name registry can still trip the
    full-checkpoint policy via ``hf_config['sliding_window']``. Catches
    new community uploads before an aliases.json entry lands.
    """
    assert _dkc.model_requires_full_checkpoint(
        "some-future-arch", hf_config={"sliding_window": 4096}
    )


def test_sliding_window_alias_metadata_explicit_override():
    """An aliases.json entry that sets ``requires_full_checkpoint: true``
    must win over the default substring match (escape hatch for
    verified-tier aliases whose family doesn't match a substring).
    """
    assert _dkc.model_requires_full_checkpoint(
        "boring-name-no-glob",
        alias_metadata={"requires_full_checkpoint": True},
    )


def test_sliding_window_checkpoint_records_full_flag(root: str):
    """When the model requires full checkpoints, the metadata sidecar
    records that flag so the loader can refuse a partial restore.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    req_hash = _dkc.request_hash("req-sw", model_name="gemma-4-12b")
    path = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=256,
        kv_dtype="bf16",
        requires_full_checkpoint=True,
        model_name="gemma-4-12b",
    )
    assert path is not None
    loaded = _dkc.load_checkpoint(path)
    assert loaded is not None
    assert loaded.requires_full_checkpoint is True


# ---------------------------------------------------------------------------
# Hybrid attention model handling (Qwen3.5)
# ---------------------------------------------------------------------------


def test_hybrid_attention_qwen35_is_delta_sliceable_not_full():
    """Qwen3.5 is reclassified OUT of the full-checkpoint-only gate.

    Its full-attention layers store K post-RoPE, token-indexed on axis 2, so
    the attention history is sliceable into deltas (the recurrent layers are
    still snapshotted whole by the delta write path). ``model_requires_full_
    checkpoint`` must therefore return False for Qwen3.5 — by name AND under a
    ``hybrid_attention`` hf_config, the two signals the old gate tripped on.
    """
    assert not _dkc.model_requires_full_checkpoint("qwen3.5-9b-4bit")
    assert not _dkc.model_requires_full_checkpoint("Qwen/Qwen3.5-Coder-32B")
    # The hybrid_attention hf_config gate must not force full for Qwen3.5...
    assert not _dkc.model_requires_full_checkpoint(
        "qwen3.5-9b", hf_config={"hybrid_attention": True}
    )
    # ...but an UNVALIDATED future hybrid still stays on the safe full path
    # (gate on model identity, not the raw flag — design §6).
    assert _dkc.model_requires_full_checkpoint(
        "some-future-hybrid", hf_config={"hybrid_attention": True}
    )


def test_hybrid_attention_checkpoint_records_flag(root: str):
    """The full-checkpoint flag must round-trip through the metadata so
    the radix index can refuse a partial restore for hybrid attention
    models.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    req_hash = _dkc.request_hash("req-hyb", model_name="qwen3.5-9b")
    path = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=512,
        kv_dtype="int4",
        requires_full_checkpoint=True,
        model_name="qwen3.5-9b",
    )
    assert path is not None
    loaded = _dkc.load_checkpoint(path)
    assert loaded is not None
    assert loaded.requires_full_checkpoint is True
    assert loaded.kv_dtype == "int4"


# ---------------------------------------------------------------------------
# Disk-cap eviction (oldest-first)
# ---------------------------------------------------------------------------


def test_disk_cap_evicts_oldest_first(root: str, monkeypatch):
    """Two checkpoints written with distinct mtimes; the older one is
    evicted first when the cap is hit. Mirrors the LMCache eviction
    pattern PR #326 calls out as oldest-first across all records.
    """
    # Write two small checkpoints; spread the mtime so the LRU sort
    # has a deterministic order.
    cache_in = _seed_kv_cache(num_tokens=8)
    p1 = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-old", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    p2 = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-new", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    assert p1 is not None and p2 is not None

    # Force the older file's mtime back in time.
    older_mtime = os.path.getmtime(p2) - 60.0
    os.utime(p1, (older_mtime, older_mtime))

    rows = _dkc.scan_checkpoints(root)
    total = sum(s for _, _, s in rows)
    # Cap at half the total. low_water_fraction=1.0 disables hysteresis so this
    # verifies the ORDERING contract (oldest-first) in isolation: evict exactly
    # one, the older. (The low-water drain is exercised separately below.)
    evicted, remaining = _dkc.enforce_disk_cap(
        root, max_bytes=total // 2, low_water_fraction=1.0
    )
    assert evicted == 1
    assert remaining <= total // 2
    # The older one (p1) must be gone; the newer one (p2) must remain.
    assert not os.path.exists(p1)
    assert os.path.exists(p2)


def test_disk_cap_low_water_drains_below_cap(root: str):
    """When the cap triggers, eviction drains to the low-water mark (a
    fraction of the cap), not just back under the cap. This is the
    anti-thrash contract: without it, every write sits at the boundary and
    re-triggers a single eviction. Write several checkpoints, set the cap so
    eviction fires, and assert the survivors sit at/under low-water, i.e.
    strictly further down than a plain evict-to-cap would leave them.
    """
    sizes = []
    for i in range(6):
        p = _dkc.write_checkpoint(
            _seed_kv_cache(num_tokens=8),
            root=root,
            req_hash=_dkc.request_hash(f"lw-{i}", model_name="m"),
            token_offset=256,
            kv_dtype="bf16",
            model_name="m",
        )
        assert p is not None
        os.utime(p, (os.path.getmtime(p) + i, os.path.getmtime(p) + i))
        sizes.append(os.path.getsize(p))
    rows = _dkc.scan_checkpoints(root)
    total = sum(s for _, _, s in rows)
    cap = int(total * 0.9)  # cap below current total so eviction fires
    _, remaining = _dkc.enforce_disk_cap(root, max_bytes=cap, low_water_fraction=0.5)
    # Drained to <= 50% of cap, well under the cap itself.
    assert remaining <= int(cap * 0.5)


def test_disk_cap_enforced_through_write_path(root: str, monkeypatch):
    """Regression guard: the cap must be enforced by ``write_checkpoint``
    ITSELF, not only by a manual ``enforce_disk_cap`` call. The live store
    mirror and the interval hook both funnel through ``write_checkpoint``,
    so a cap that only fired from the (now-disabled) generation-time hook
    would let the checkpoint dir grow unbounded. This writes several
    checkpoints under a tiny env cap and asserts the total stays bounded
    without ever calling ``enforce_disk_cap`` directly.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    # Size the cap to roughly two checkpoints, then write five with
    # increasing mtimes so eviction has a deterministic oldest-first order.
    first = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-0", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    assert first is not None
    one = os.path.getsize(first)
    cap = int(one * 2.5)
    monkeypatch.setenv(_dkc._DISK_CAP_ENV, str(cap))
    # Write several more under the cap. Each write must enforce the cap
    # inline; we assert only the invariant (bounded total + eviction
    # happened), not WHICH files survive: sub-millisecond writes can tie on
    # mtime, so the oldest-first pick among ties is not deterministic. The
    # contract F1 guards is "the dir can't grow past the cap through the live
    # write path", and that holds regardless of the tie-break.
    for i in range(1, 6):
        p = _dkc.write_checkpoint(
            cache_in,
            root=root,
            req_hash=_dkc.request_hash(f"req-{i}", model_name="m"),
            token_offset=256,
            kv_dtype="bf16",
            model_name="m",
        )
        assert p is not None
    rows = _dkc.scan_checkpoints(root)
    total = sum(s for _, _, s in rows)
    # Bounded within one checkpoint of the cap (the last write lands before
    # its own enforce trims back), and at least one eviction fired.
    assert total <= cap + one
    assert len(rows) < 6


def test_disk_cap_zero_disables_eviction(root: str):
    """``max_bytes=0`` is the operator escape hatch — no eviction even
    when the disk is full of checkpoints.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-1", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    evicted, _ = _dkc.enforce_disk_cap(root, max_bytes=0)
    assert evicted == 0


def test_disk_cap_under_limit_is_noop(root: str):
    """When the on-disk total is already under the cap, nothing is
    evicted. Guards against an over-eager evict-everything bug.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-stay", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    evicted, remaining = _dkc.enforce_disk_cap(root, max_bytes=10**12)
    assert evicted == 0
    assert remaining > 0


def test_disk_cap_nan_max_bytes_falls_back_to_default(root: str):
    """NaN / Inf max_bytes is coerced to the default (NaN-safety rule:
    Pydantic ``Field(ge=)`` does not reject NaN, so we have to here).
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-nan", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    # math.nan / math.inf must not crash; both fall back to the default.
    import math

    evicted, _ = _dkc.enforce_disk_cap(root, max_bytes=math.nan)
    assert evicted == 0
    evicted, _ = _dkc.enforce_disk_cap(root, max_bytes=math.inf)
    assert evicted == 0


# ---------------------------------------------------------------------------
# Metrics counters — writes / loads / bytes / evictions
# ---------------------------------------------------------------------------


def test_metrics_counters_tick_on_write_load_evict(root: str):
    """Every committed write/load/eviction bumps the corresponding
    counter so /metrics renders meaningful series.
    """
    before = _dkc.get_stats()

    cache_in = _seed_kv_cache(num_tokens=8)
    p = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-metric", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    after_write = _dkc.get_stats()
    assert after_write["writes"] == before["writes"] + 1
    assert after_write["bytes"] > 0

    loaded = _dkc.load_checkpoint(p)
    assert loaded is not None
    after_load = _dkc.get_stats()
    assert after_load["loads"] == after_write["loads"] + 1

    # Force eviction by setting cap to 1 byte; the single file is evicted.
    _dkc.enforce_disk_cap(root, max_bytes=1)
    after_evict = _dkc.get_stats()
    assert after_evict["evictions"] == after_load["evictions"] + 1
    assert after_evict["bytes"] == 0


# ---------------------------------------------------------------------------
# maybe_write_checkpoint — wrapper exercises gate + write together
# ---------------------------------------------------------------------------


def test_maybe_write_checkpoint_below_boundary_is_noop(root: str):
    """Below the first boundary the wrapper must NOT write."""
    cache_in = _seed_kv_cache(num_tokens=8)
    new_offset, path = _dkc.maybe_write_checkpoint(
        cache_in,
        root=root,
        req_hash="rh-noop",
        num_tokens=128,
        last_checkpoint_at=0,
    )
    assert new_offset == 0
    assert path is None


def test_maybe_write_checkpoint_above_boundary_snaps_to_multiple(root: str):
    """A step that overshoots the boundary (320 tokens) must snap the
    watermark to the largest multiple of ``interval`` that is ≤
    ``num_tokens`` (256), not just bump by ``interval``. Without this
    the next step would re-fire because the gap shrank under interval.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    new_offset, path = _dkc.maybe_write_checkpoint(
        cache_in,
        root=root,
        req_hash="rh-snap",
        num_tokens=320,
        last_checkpoint_at=0,
    )
    assert new_offset == 256
    assert path is not None


# ---------------------------------------------------------------------------
# Metadata sidecar — schema + JSON shape
# ---------------------------------------------------------------------------


def test_metadata_sidecar_shape(root: str):
    """The JSON sidecar must carry the fields the loader / radix
    hand-off depend on so a future operator can sanity-check the on-
    disk layout by reading the JSON alone.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    req_hash = _dkc.request_hash("req-meta", model_name="m")
    path = _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=512,
        kv_dtype="int4",
        requires_full_checkpoint=True,
        model_name="some/model",
        extra_metadata={"tokens_key": [1, 2, 3, 4, 5]},
    )
    assert path is not None
    sidecar = path.replace(".safetensors", ".json")
    assert os.path.isfile(sidecar)
    with open(sidecar) as fh:
        data = json.load(fh)
    assert data["schema_version"] == 2
    assert data["token_offset"] == 512
    assert data["kv_dtype"] == "int4"
    assert data["requires_full_checkpoint"] is True
    assert data["model_name"] == "some/model"
    assert data["size_bytes"] > 0
    # extra_metadata must merge.
    assert data["tokens_key"] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def test_cleanup_request_removes_all_checkpoints(root: str):
    """When a request completes, cleanup_request must remove every
    checkpoint under ``<root>/<req_hash>``. Otherwise long-running
    servers accumulate per-request dirs forever.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    req_hash = _dkc.request_hash("req-cleanup", model_name="m")
    _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    _dkc.write_checkpoint(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=512,
        kv_dtype="bf16",
        model_name="m",
    )
    rows = _dkc.scan_checkpoints(root)
    assert len(rows) == 2
    n = _dkc.cleanup_request(root, req_hash)
    assert n >= 2  # 2 safetensors + 2 sidecars = 4; lower bound is 2
    assert _dkc.scan_checkpoints(root) == []


# ---------------------------------------------------------------------------
# Env override for the disk cap — operator escape hatch
# ---------------------------------------------------------------------------


def test_env_override_max_bytes(monkeypatch):
    """``QMLX_KV_CHECKPOINT_MAX_BYTES`` overrides the default cap.

    Covers the integer parse + the explicit-0-disables shape. An
    invalid value falls back to the default.
    """
    monkeypatch.setenv("QMLX_KV_CHECKPOINT_MAX_BYTES", "12345")
    assert _dkc.resolve_max_disk_bytes() == 12345

    monkeypatch.setenv("QMLX_KV_CHECKPOINT_MAX_BYTES", "0")
    assert _dkc.resolve_max_disk_bytes() == 0

    monkeypatch.setenv("QMLX_KV_CHECKPOINT_MAX_BYTES", "not-an-int")
    assert _dkc.resolve_max_disk_bytes() == _dkc.DEFAULT_MAX_DISK_BYTES


# ---------------------------------------------------------------------------
# Bench checkpoint hook — wiring sanity
# ---------------------------------------------------------------------------


def test_request_checkpoint_state_default_interval():
    """``RequestCheckpointState`` defaults must match the module
    constants so the scheduler hook stays coherent with the public
    contract.
    """
    state = _dkc.RequestCheckpointState(req_hash="abc")
    assert state.interval == _dkc.DEFAULT_CHECKPOINT_INTERVAL
    assert state.last_checkpoint_at == 0
    assert state.requires_full_checkpoint is False
    assert state.kv_dtype == "bf16"


# ---------------------------------------------------------------------------
# Matchable-aware disk-cap eviction (#9)
# ---------------------------------------------------------------------------


def _write_matchable(root: str, req: str, tokens: list[int]) -> str:
    """Write a checkpoint WITH a tokens blob (matchable / restorable)."""
    offset = len(tokens)
    path = _dkc.write_checkpoint(
        _seed_kv_cache(num_tokens=offset),
        root=root,
        req_hash=_dkc.request_hash(req, model_name="m"),
        token_offset=offset,
        kv_dtype="bf16",
        model_name="m",
        extra_metadata={"tokens_key": list(tokens), "save_uuid": "u-" + req},
    )
    assert path is not None
    # Sanity: the paired tokens blob (the matchable predicate) exists.
    assert os.path.exists(path.replace(".safetensors", ".tokens.bin"))
    return path


def _write_tokensless(root: str, req: str, offset: int = 256) -> str:
    """Write an interval-hook-style checkpoint WITHOUT a tokens blob."""
    path = _dkc.write_checkpoint(
        _seed_kv_cache(num_tokens=8),
        root=root,
        req_hash=_dkc.request_hash(req, model_name="m"),
        token_offset=offset,
        kv_dtype="bf16",
        model_name="m",
    )
    assert path is not None
    assert not os.path.exists(path.replace(".safetensors", ".tokens.bin"))
    return path


def test_disk_cap_evicts_unmatchable_before_matchable_regardless_of_age(root: str):
    """Class beats age: a tokens-less (interval-hook) checkpoint is evicted
    before a matchable (tokens-blob-bearing) boundary checkpoint even when
    the matchable one is OLDER.

    This is the core of issue #9 — the interval-write flood must never
    reclaim the received-prompt boundary checkpoints the next turn's restore
    depends on. Before the fix, ``enforce_disk_cap`` evicted strictly
    oldest-mtime-first, so an old-but-precious boundary checkpoint was the
    FIRST thing dropped.
    """
    _dkc.reset_content_index_for_tests()
    matchable = _write_matchable(root, "boundary", tokens=list(range(1, 33)))
    tokensless = _write_tokensless(root, "interval", offset=256)

    # Make the MATCHABLE one strictly OLDER so a naive oldest-first policy
    # would evict it first. The class rule must override the age rule.
    old = os.path.getmtime(tokensless) - 120.0
    os.utime(matchable, (old, old))

    rows = _dkc.scan_checkpoints(root)
    total = sum(s for _, _, s in rows)
    # Cap one byte under the total; low_water=1.0 disables the drain so
    # exactly one checkpoint is evicted — proving WHICH class goes first.
    evicted, remaining = _dkc.enforce_disk_cap(
        root, max_bytes=total - 1, low_water_fraction=1.0
    )
    assert evicted == 1
    # The tokens-less (newer) one is gone; the matchable (older) one survives.
    assert not os.path.exists(tokensless)
    assert os.path.exists(matchable)
    assert os.path.exists(matchable.replace(".safetensors", ".tokens.bin"))


def test_scan_checkpoints_folds_tokens_blob_bytes_into_size(root: str):
    """The disk-cap accounting must count the paired ``.tokens.bin`` bytes,
    not just the safetensors body — otherwise the cap under-reports true disk
    use and a matchable checkpoint looks smaller than it is.
    """
    _dkc.reset_content_index_for_tests()
    path = _write_matchable(root, "acct", tokens=list(range(1, 65)))
    body = os.path.getsize(path)
    tok = os.path.getsize(path.replace(".safetensors", ".tokens.bin"))
    rows = _dkc.scan_checkpoints(root)
    assert len(rows) == 1
    # Reported size == body + tokens blob.
    assert rows[0][2] == body + tok


def test_content_index_lookup_prefers_prompt_boundary_over_think_stripped(
    root: str,
):
    """Think-strip divergence: a generated-output checkpoint keyed on
    ``prompt + <think></think> + <tool_call>`` is NEVER a prefix of the next
    prompt (the client strips ``<think>...</think>`` before echoing it back),
    so a lookup for ``prompt + <tool_call>`` must return the prompt-boundary
    checkpoint (a true prefix), not the generated-output one. This is why
    keying checkpoints on generated output does not help (#9).
    """
    _dkc.reset_content_index_for_tests()

    prompt = [1, 2, 3, 4]
    think = [90, 91]  # <think></think>
    tool_call = [50, 51]

    # Boundary checkpoint: keyed on the received prompt only.
    _write_matchable(root, "prompt-boundary", tokens=prompt)
    # Generated-output checkpoint: prompt + think + tool_call.
    gen_key = prompt + think + tool_call
    _write_matchable(root, "gen-output", tokens=gen_key)

    # Next turn's prompt: the client stripped the think block, so the tool
    # result follows the prompt directly.
    query = prompt + tool_call + [77, 78]  # + tool_result tokens

    loaded = _dkc.get_content_index().lookup(query)
    assert loaded is not None
    # The prompt-boundary checkpoint (offset == len(prompt)) wins — the
    # generated-output key is not a prefix of the think-stripped query.
    assert loaded.token_offset == len(prompt)

    # And the diagnostic divergence lands exactly at the assistant-content
    # boundary (first token past the shared prompt).
    div = _dkc.get_content_index().nearest_divergence(query)
    assert div is not None
    _best_off, divergence_index, _best_key = div
    assert divergence_index == len(prompt)


# ---------------------------------------------------------------------------
# Content-index lookup — partial-restore fallback + corrupt-checkpoint
# quarantine, with a hard-vs-transient split (regression for the 184k-token
# tokens_blob_verify_fail cold-prefill poison loop)
# ---------------------------------------------------------------------------


def _write_indexed_checkpoint(root: str, req_id: str, tokens: list[int]) -> str:
    """Write a checkpoint whose tokens blob equals ``tokens`` and register it
    in the process-wide content index. Returns the ``req_hash``.

    ``token_offset`` is tied to ``len(tokens)`` so the tokens-blob length, the
    claimed offset, and the seeded cache length all agree (the invariant
    :meth:`DiskCheckpointIndex.lookup` byte-verifies).
    """
    n = len(tokens)
    cache = _seed_kv_cache(num_tokens=n)
    req_hash = _dkc.request_hash(req_id, model_name="m")
    path = _dkc.write_checkpoint(
        cache,
        root=root,
        req_hash=req_hash,
        token_offset=n,
        kv_dtype="bf16",
        model_name="m",
        extra_metadata={"tokens_key": list(tokens)},
    )
    assert path is not None
    return req_hash


def test_lookup_hot_path_returns_longest_valid_checkpoint(root: str):
    """When the longest prefix verifies, lookup returns it unchanged — the hot
    path must behave byte-for-byte as before the fallback rewrite.
    """
    idx = _dkc.get_content_index()
    idx.clear()
    query = list(range(1, 65))
    _write_indexed_checkpoint(root, "hot-short", query[:32])
    _write_indexed_checkpoint(root, "hot-long", query[:64])

    loaded = idx.lookup(query)
    assert loaded is not None
    assert loaded.token_offset == 64
    # Nothing was evicted: both keys survive.
    assert tuple(query[:64]) in idx._by_key
    assert tuple(query[:32]) in idx._by_key


def test_lookup_falls_back_to_shorter_valid_prefix_when_longest_corrupt(root: str):
    """A corrupt longest-prefix checkpoint must NOT force a full cold prefill.
    lookup walks to the next-shorter verified prefix and returns it.
    """
    idx = _dkc.get_content_index()
    idx.clear()
    query = list(range(1, 65))
    _write_indexed_checkpoint(root, "fb-short", query[:32])
    req_hash_long = _write_indexed_checkpoint(root, "fb-long", query[:64])

    # Corrupt the LONG checkpoint's tokens blob with a HARD mismatch (8 bytes:
    # magic absent while the index declared a save_uuid) — the exact
    # production failure mode.
    tok_path = _dkc.tokens_path(root, req_hash_long, 64)
    assert os.path.isfile(tok_path)
    with open(tok_path, "wb") as fh:
        fh.write(b"\x00" * 8)

    loaded = idx.lookup(query)
    assert loaded is not None
    # Fell back to the shorter, still-valid checkpoint.
    assert loaded.token_offset == 32


def test_lookup_quarantines_corrupt_checkpoint(root: str):
    """After a lookup hits a HARD-corrupt longest checkpoint, that checkpoint
    is evicted from the index (a later lookup can't re-select it) AND its
    on-disk artifacts are renamed aside with a ``.corrupt`` suffix so a future
    ``build_content_index`` scan won't re-add it.
    """
    idx = _dkc.get_content_index()
    idx.clear()
    query = list(range(1, 65))
    _write_indexed_checkpoint(root, "qn-short", query[:32])
    req_hash_long = _write_indexed_checkpoint(root, "qn-long", query[:64])

    tok_path = _dkc.tokens_path(root, req_hash_long, 64)
    with open(tok_path, "wb") as fh:
        fh.write(b"\x00" * 8)

    # Trigger the quarantine.
    idx.lookup(query)

    # In-memory: the corrupt (len-64) key is gone; the shorter one stays.
    assert tuple(query[:64]) not in idx._by_key
    assert tuple(query[:32]) in idx._by_key
    # A second lookup no longer even sees the corrupt candidate — it returns
    # the shorter one directly (no re-fail on the same corrupt entry).
    again = idx.lookup(query)
    assert again is not None and again.token_offset == 32

    # On-disk: the safetensors + tokens artifacts were renamed aside.
    orig_body = _dkc.checkpoint_path(root, req_hash_long, 64)
    assert not os.path.exists(orig_body)
    assert os.path.exists(orig_body + ".corrupt")
    assert not os.path.exists(tok_path)
    assert os.path.exists(tok_path + ".corrupt")


def _corrupt_tokens_blob_hard(root: str, req_hash: str, offset: int) -> str:
    """Overwrite a checkpoint's tokens blob with bytes that read as a HARD
    content mismatch (no v3 magic while the index declared a save_uuid).
    Returns the tokens path.
    """
    tok_path = _dkc.tokens_path(root, req_hash, offset)
    assert os.path.isfile(tok_path)
    with open(tok_path, "wb") as fh:
        fh.write(b"\x00" * 8)
    return tok_path


def test_lookup_returns_none_when_longest_and_fallback_both_corrupt(root: str):
    """Verify-on-fallback contract.

    When the LONGEST prefix is corrupt and the only shorter fallback candidate
    is ALSO corrupt, ``lookup`` must walk past both and return None rather than
    hand back an unverified cache. This pins that the byte-level gates re-run
    on every fallback candidate, not just the first.
    """
    idx = _dkc.get_content_index()
    idx.clear()
    query = list(range(1, 65))
    req_hash_short = _write_indexed_checkpoint(root, "bc-short", query[:32])
    req_hash_long = _write_indexed_checkpoint(root, "bc-long", query[:64])

    # Both candidates are corrupt (HARD mismatch on re-read).
    _corrupt_tokens_blob_hard(root, req_hash_long, 64)
    _corrupt_tokens_blob_hard(root, req_hash_short, 32)

    # No verified prefix survives → genuine miss (cold prefill), never an
    # unverified cache.
    assert idx.lookup(query) is None
    # Both were quarantined out of the in-memory index.
    assert tuple(query[:64]) not in idx._by_key
    assert tuple(query[:32]) not in idx._by_key


def test_transient_load_failure_soft_evicts_without_renaming(root: str, monkeypatch):
    """A TRANSIENT failure (``load_checkpoint`` returns None, e.g. a momentary
    mmap / OOM) must NOT rename the checkpoint aside.

    The key is dropped from the in-memory index so the fallback loop advances
    this call, but the on-disk artifacts stay put — a later ``build_from_root``
    rescan re-adds the entry once the glitch clears. Contrast with a HARD
    mismatch, which DOES rename aside.
    """
    idx = _dkc.get_content_index()
    idx.clear()
    query = list(range(1, 33))
    req_hash = _write_indexed_checkpoint(root, "tr-load", query)

    # Simulate a transient failure to materialise the (otherwise valid) cache.
    monkeypatch.setattr(_dkc, "load_checkpoint", lambda path: None)

    assert idx.lookup(query) is None  # nothing else to fall back to
    # In-memory: soft-evicted.
    assert tuple(query) not in idx._by_key

    # On-disk: artifacts were NOT renamed aside — the file is untouched.
    body = _dkc.checkpoint_path(root, req_hash, 32)
    tok_path = _dkc.tokens_path(root, req_hash, 32)
    assert os.path.exists(body)
    assert os.path.exists(tok_path)
    assert not os.path.exists(body + ".corrupt")
    assert not os.path.exists(tok_path + ".corrupt")

    # A rescan re-adds the transiently-evicted entry (the glitch was purely
    # in-memory; the on-disk checkpoint is still good).
    idx.build_from_root(root)
    assert tuple(query) in idx._by_key


def test_transient_verify_io_error_soft_evicts_without_renaming(root: str):
    """A TRANSIENT verify failure classified ``io_error`` (a truncated / short
    tokens blob, the signature of a racing cleanup or half-written file) must
    soft-evict, not quarantine: the file is left in place, not renamed aside.
    """
    idx = _dkc.get_content_index()
    idx.clear()
    query = list(range(1, 33))
    req_hash = _write_indexed_checkpoint(root, "tr-io", query)

    # Truncate the tokens blob below the magic length → _read_tokens_bin
    # reports "shorter than magic length" → _verify classifies io_error
    # (transient), NOT a content mismatch.
    tok_path = _dkc.tokens_path(root, req_hash, 32)
    with open(tok_path, "wb") as fh:
        fh.write(b"\x00" * 4)

    assert idx.lookup(query) is None
    assert tuple(query) not in idx._by_key
    # Not renamed aside — a soft evict leaves the on-disk artifact alone.
    body = _dkc.checkpoint_path(root, req_hash, 32)
    assert os.path.exists(tok_path)
    assert not os.path.exists(tok_path + ".corrupt")
    assert not os.path.exists(body + ".corrupt")


def test_hard_mismatch_renames_aside_and_is_not_re_added_by_rescan(root: str):
    """Direct contrast to the transient cases: a HARD byte mismatch renames the
    artifacts aside, so a subsequent ``build_from_root`` rescan can NOT re-add
    the poisoned checkpoint (the real poison-loop fix stays intact).
    """
    idx = _dkc.get_content_index()
    idx.clear()
    query = list(range(1, 33))
    req_hash = _write_indexed_checkpoint(root, "hd-mm", query)

    tok_path = _corrupt_tokens_blob_hard(root, req_hash, 32)
    body = _dkc.checkpoint_path(root, req_hash, 32)

    assert idx.lookup(query) is None
    # Renamed aside.
    assert not os.path.exists(tok_path)
    assert os.path.exists(tok_path + ".corrupt")
    assert not os.path.exists(body)
    assert os.path.exists(body + ".corrupt")

    # A rescan finds nothing to re-add (the poison was moved out of the way).
    idx.clear()
    idx.build_from_root(root)
    assert tuple(query) not in idx._by_key
