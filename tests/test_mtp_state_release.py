# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``_release_kv_cache_fields`` (PR #47 MTP-leak fix).

The helper frees the REAL storage that a completed request's suspended MTP
generator pins. ``cache.state = None`` was a silent no-op on mlx-lm
``KVCache`` (the setter unpacks ``self.keys, self.values = v`` and raises on
``None``, which callers swallowed), so the prior cleanup freed nothing on the
attention layers. This helper nulls the concrete per-type fields instead and
identity-skips caches still owned by the live batch.
"""

from vllm_mlx.scheduler import _release_kv_cache_fields


class _FakeKVCache:
    """Stand-in for mlx-lm KVCache/QuantizedKVCache: keys/values/offset."""

    def __init__(self):
        self.keys = object()  # non-None, non-callable storage sentinel
        self.values = object()
        self.offset = 7


class _FakeArraysCache:
    """Stand-in for ArraysCache (GatedDeltaNet SSM state): a ``.cache`` list."""

    def __init__(self):
        self.cache = [object(), object()]


def test_release_nulls_storage_and_respects_skip_ids():
    kv = _FakeKVCache()
    arrays = _FakeArraysCache()
    skipped = _FakeKVCache()

    released = _release_kv_cache_fields([kv, arrays, skipped], skip_ids={id(skipped)})

    # KVCache storage dropped, offset reset.
    assert kv.keys is None
    assert kv.values is None
    assert kv.offset == 0

    # ArraysCache slots nulled in place.
    assert arrays.cache == [None, None]

    # The skipped cache (still owned by the live batch) is left untouched.
    assert skipped.keys is not None
    assert skipped.values is not None
    assert skipped.offset == 7

    # Only the two non-skipped caches are counted as released.
    assert released == 2


def test_release_drops_rollback_tape():
    c = _FakeArraysCache()
    c.rollback_state = [object(), object()]  # a K-position MTP rollback tape

    released = _release_kv_cache_fields([c])

    assert c.rollback_state is None
    assert c.cache == [None, None]
    assert released == 1


def test_release_skip_ids_can_protect_every_cache():
    kv = _FakeKVCache()
    released = _release_kv_cache_fields([kv], skip_ids={id(kv)})
    assert kv.keys is not None  # untouched
    assert kv.offset == 7
    assert released == 0
