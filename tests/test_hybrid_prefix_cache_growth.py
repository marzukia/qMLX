# SPDX-License-Identifier: Apache-2.0
"""Hybrid (GatedDeltaNet / Mamba MoE) recurrent-state cache policy.

History
-------
Issue #214 (oldriverno1, michaelasper) originally asked for hybrid multi-turn
conversations to hit the prefix cache the way dense models do, so TTFT would
not grow linearly with conversation length. We shipped that: stored
``[P + R1]`` was reused as a strict prefix of turn-2's ``[P + R1 + M2]`` (no
trim required — the RNN state at end-of-stored is exactly the state needed at
start-of-M2-prefill).

Issues #1025 / #1058 then showed the OTHER edge of the same behavior: those
per-request recurrent-state (``ArraysCache``) entries are stored by reference
and are NEVER a prefix of the *next* request across DIFFERENT conversations
(each request's output differs → every key is a unique superset), so
prefix-subset eviction never reclaims them. They only drop under the cache's
own byte budget (independent of ``--gpu-memory-utilization``), so Metal
``active`` ratchets up holding leaked recurrent state → D-METAL-CAP wedges /
OOM.

Resolution (direction 1, raullen 2026-07-09)
--------------------------------------------
Stop caching non-trimmable recurrent-state entries entirely. ``store`` now
DROPS any cache that carries an ``ArraysCache`` / ``CacheList``-wrapping-one
layer (``is_trimmable() == False``). This trades away the #214 within-
conversation multi-turn speedup (hybrid turns re-prefill) to stop the
cross-conversation leak. The tests below encode the NEW policy; the previous
#214 "must hit" assertions are intentionally inverted.

Dense (all-``KVCache``) models are unaffected — their state is trimmable and
still cached/reused normally.
"""

from unittest.mock import MagicMock

import pytest

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig


class _MockArray:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class TrimmableLayer:
    """Stands in for KVCache (transformer attention layer)."""

    def __init__(self, nbytes: int = 200, offset: int = 0):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)
        self._offset = offset

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, val: int) -> None:
        self._offset = val

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:  # KVCache-like: defines trim
        return n


class NonTrimmableLayer:
    """Stands in for ArraysCache (DeltaNet/Mamba RNN state)."""

    def __init__(self, nbytes: int = 200):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)

    def is_trimmable(self) -> bool:
        return False


def _dense_cache(n: int = 10):
    return [TrimmableLayer() for _ in range(n)]


def _hybrid_cache(n_trimmable: int = 10, n_non_trimmable: int = 30):
    """Mirror Qwen3.5/3.6 hybrid layout: ~25% transformer, ~75% DeltaNet."""
    return [TrimmableLayer() for _ in range(n_trimmable)] + [
        NonTrimmableLayer() for _ in range(n_non_trimmable)
    ]


@pytest.fixture
def cache():
    config = MemoryCacheConfig(max_memory_mb=10, max_entries=64)
    return MemoryAwarePrefixCache(MagicMock(), config)


# ---------------------------------------------------------------------------
# Dense (all-trimmable) is unaffected — still stored and reused.
# ---------------------------------------------------------------------------


def test_dense_growing_conversation_hits_prefix(cache):
    """Dense models still hit the prefix path on growing conversations."""
    prompt = list(range(1000, 1100))
    response_1 = [9001, 9002]
    new_msg = list(range(2000, 2050))

    assert cache.store(prompt, _dense_cache()) is True
    assert cache.store(prompt + response_1, _dense_cache()) is True

    turn_2 = prompt + response_1 + new_msg
    result, remaining = cache.fetch(turn_2)

    assert result is not None, "Dense growing conversation should hit prefix"
    assert remaining == new_msg


# ---------------------------------------------------------------------------
# #1025 / #1058: hybrid recurrent-state entries are DROPPED at store time.
# ---------------------------------------------------------------------------


def test_hybrid_store_is_dropped(cache):
    """A cache with any non-trimmable layer must NOT be stored (leak fix)."""
    prompt = list(range(1000, 1100))

    stored = cache.store(prompt, _hybrid_cache())

    assert stored is False, "Hybrid recurrent-state entry must be dropped, not stored"
    assert tuple(prompt) not in cache._entries, (
        "Non-trimmable entry leaked into _entries — this is the #1025/#1058 leak"
    )
    assert cache.get_stats()["non_trimmable_skips"] == 1


def test_hybrid_multiturn_does_not_leak(cache):
    """A multi-turn hybrid conversation leaves NO entries in the cache.

    Every turn stores a longer ``[P + ... ]`` superset; before the fix each
    one lingered forever (never a prefix of a *different* conversation's next
    key). After the fix none are retained → ``_entries`` stays empty and
    ``_current_memory`` returns to 0.
    """
    prompt = list(range(1000, 1100))
    r1, r2 = [9001, 9002], [9003, 9004]
    m2, m3 = list(range(2000, 2050)), list(range(3000, 3030))

    cache.store(prompt, _hybrid_cache())
    cache.store(prompt + r1, _hybrid_cache())
    cache.store(prompt + r1 + m2 + r2, _hybrid_cache())
    cache.store(prompt + r1 + m2 + r2 + m3, _hybrid_cache())

    assert len(cache._entries) == 0, (
        f"Hybrid conversation left {len(cache._entries)} lingering entries — "
        "this is the recurrent-state leak (#1025/#1058)"
    )
    assert cache._current_memory == 0
    assert cache.get_stats()["non_trimmable_skips"] == 4


def test_hybrid_fetch_always_misses(cache):
    """With hybrid stores dropped, every hybrid fetch is a clean miss."""
    prompt = list(range(1000, 1100))
    response_1 = [9001, 9002]
    new_msg = list(range(2000, 2050))

    cache.store(prompt, _hybrid_cache())
    cache.store(prompt + response_1, _hybrid_cache())

    turn_2 = prompt + response_1 + new_msg
    result, remaining = cache.fetch(turn_2)

    assert result is None, "Hybrid entries are never stored → fetch must miss"
    assert remaining == turn_2


# ---------------------------------------------------------------------------
# Granularity: partial hybrid (even ONE non-trimmable layer) drops the entry.
# ---------------------------------------------------------------------------


def test_single_non_trimmable_layer_drops_entry(cache):
    """One non-trimmable layer among many trimmable ones drops the whole entry.

    A half-populated entry (trimmable layers only) can't reconstruct a hybrid
    model, so we skip the whole entry rather than store a useless subset.
    """
    prompt = list(range(1000, 1100))
    mostly_dense = _dense_cache(n=39) + [NonTrimmableLayer()]

    assert cache.store(prompt, mostly_dense) is False
    assert tuple(prompt) not in cache._entries


def test_dict_form_arrayscache_dropped(cache):
    """Block-aware (dict-form) extracted states are gated on class_name too."""
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": "KVCache", "state": (1, 2), "meta_state": ("0",)},
        {"class_name": "ArraysCache", "state": (3, 4), "meta_state": ("0",)},
    ]

    assert cache.store(prompt, dict_cache) is False
    assert tuple(prompt) not in cache._entries


def test_dict_form_all_kvcache_stored(cache):
    """A dict-form entry with only KVCache layers is still stored."""
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": "KVCache", "state": (1, 2), "meta_state": ("0",)},
        {"class_name": "KVCache", "state": (3, 4), "meta_state": ("0",)},
    ]

    assert cache.store(prompt, dict_cache) is True
    assert tuple(prompt) in cache._entries


@pytest.mark.parametrize(
    "kv_class",
    ["RotatingKVCache", "ChunkedKVCache", "ConcatenateKVCache", "QuantizedKVCache"],
)
def test_dict_form_trimmable_kv_variants_still_stored(cache, kv_class):
    """Dict-form sliding-window / other trimmable KV classes must NOT be dropped.

    Regression for codex #1075 finding: an allowlist of only KVCache would
    wrongly classify RotatingKVCache & friends as non-trimmable and drop the
    entry, regressing prefix reuse for dense / sliding-window models. The
    denylist keeps them cacheable.
    """
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": kv_class, "state": (1, 2), "meta_state": ("0",)},
        {"class_name": kv_class, "state": (3, 4), "meta_state": ("0",)},
    ]

    assert cache.store(prompt, dict_cache) is True, (
        f"{kv_class} is trimmable and must remain cacheable"
    )
    assert tuple(prompt) in cache._entries


def test_dict_form_mamba_variant_dropped(cache):
    """Vendor-suffixed recurrent class names are caught by substring match."""
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": "KVCache", "state": (1, 2), "meta_state": ("0",)},
        {
            "class_name": "GatedDeltaNetArraysCache",
            "state": (3, 4),
            "meta_state": ("0",),
        },
    ]

    assert cache.store(prompt, dict_cache) is False
    assert tuple(prompt) not in cache._entries


# ---------------------------------------------------------------------------
# Guards preserved from the #214 era: trim-required matches still MISS. These
# now also never even reach fetch (store dropped them), but the fetch-side
# non-trimmable guard stays as defense-in-depth for any legacy on-disk entry.
# ---------------------------------------------------------------------------


def test_hybrid_supersequence_still_skipped(cache):
    """Even if a hybrid entry existed, a trim-required supersequence match must
    skip. We inject directly into ``_entries`` to bypass the store gate and
    exercise the fetch-side guard (legacy on-disk entry defense-in-depth).
    """
    from vllm_mlx.memory_cache import _CacheEntry

    long_stored = list(range(1000, 1200))
    entry = _CacheEntry.create(long_stored, _hybrid_cache())
    cache._entries[tuple(long_stored)] = entry
    import bisect

    bisect.insort(cache._sorted_keys, tuple(long_stored))

    short_request = list(range(1000, 1100))
    result, remaining = cache.fetch(short_request)

    assert result is None, (
        "Trim-required match on non-trimmable hybrid layers must still skip"
    )
    assert remaining == short_request
