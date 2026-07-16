# SPDX-License-Identifier: Apache-2.0
"""Delta / incremental KV checkpoint tests (design §7).

These pin the correctness contract for the delta-checkpoint write/restore/evict
machinery in :mod:`vllm_mlx.runtime.disk_kv_checkpoint`:

  (a) a chained restore is BIT-IDENTICAL to a full checkpoint at the same offset;
  (b) a broken chain evicts-on-fail and the fallback reaches the base's shorter
      prefix;
  (c) chain-aware eviction never unlinks a base with a live descendant;
  (d) the keyframe interval caps chain length;
  (e) delta ranges stay contiguous under simulated spec-decode overshoot;
  (f) two concurrent restores sharing a base both succeed without corruption;
  (g) a mixed-schema chain (v1 base + v2 delta) restores via the implicit-base rule;
  (h) quantized-layer slicing is bit-exact through a chain;
  (i) a long (50+ link) chain restores correctly via the layer-at-a-time concat.

The attention layers are real ``KVCache`` / ``QuantizedKVCache`` objects and the
recurrent layer is a real ``ArraysCache`` (the GatedDeltaNet cache class Qwen3.5
uses), so the round-trip exercises the actual mlx-lm save/load format, the class-
name classifier, and the mmap selective reader — not a mock.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache  # noqa: E402

from vllm_mlx.runtime import disk_kv_checkpoint as _dkc  # noqa: E402

_TOKEN_BASE = 20000


@pytest.fixture(autouse=True)
def _delta_env(monkeypatch):
    """Enable the delta path and reset all module-global state per test."""
    monkeypatch.setenv("DELTA_CHECKPOINTS_ENABLED", "1")
    _dkc.reset_stats_for_tests()
    _dkc.reset_content_index_for_tests()
    _dkc.reset_refcounts_for_tests()
    yield
    _dkc.reset_content_index_for_tests()
    _dkc.reset_refcounts_for_tests()


@pytest.fixture
def root(tmp_path: Path) -> str:
    return str(tmp_path / "delta-root")


# ---------------------------------------------------------------------------
# Cache builders (real mlx-lm cache objects)
# ---------------------------------------------------------------------------


def _canonical_full(
    n: int,
    *,
    n_attn: int = 2,
    with_recurrent: bool = True,
    quantized: bool = False,
    seed: int = 0,
) -> list:
    """A canonical full cache of ``n`` tokens: ``n_attn`` attention layers plus
    (optionally) one recurrent ``ArraysCache`` layer, all seeded deterministically.
    """
    layers: list = []
    for L in range(n_attn):
        if quantized:
            c = QuantizedKVCache(group_size=64, bits=4)
            k = mx.random.normal((1, 2, n, 64), key=mx.random.key(seed + L * 2))
            v = mx.random.normal((1, 2, n, 64), key=mx.random.key(seed + L * 2 + 1))
        else:
            c = KVCache()
            k = mx.random.normal((1, 2, n, 8), key=mx.random.key(seed + L * 2)).astype(
                mx.bfloat16
            )
            v = mx.random.normal(
                (1, 2, n, 8), key=mx.random.key(seed + L * 2 + 1)
            ).astype(mx.bfloat16)
        c.update_and_fetch(k, v)
        layers.append(c)
    if with_recurrent:
        rec = ArraysCache(2)
        rec[0] = mx.random.normal((1, 4, 16, 16), key=mx.random.key(seed + 991))
        rec[1] = mx.random.normal((1, 3, 12), key=mx.random.key(seed + 992))
        layers.append(rec)
    mx.eval([layer.state for layer in layers])
    return layers


def _slice_full_to(canon: list, m: int) -> list:
    """Derive a full cache at offset ``m`` from a canonical full cache: attention
    layers sliced to ``[:m]`` on the token axis, recurrent layers passed through.

    Row-stability (token k's KV row never changes when later tokens append) makes
    the sliced attention EQUAL the canonical attention over ``[:m]``, so a base at
    ``m`` plus a delta over ``[m:n]`` reassembles the canonical cache exactly.
    """
    out: list = []
    for layer in canon:
        cls = type(layer).__name__
        if cls == "KVCache":
            k, v = layer.state
            nc = KVCache()
            nc.state = (k[:, :, :m, :], v[:, :, :m, :])
            out.append(nc)
        elif cls == "QuantizedKVCache":
            keys, values = layer.state
            nc = QuantizedKVCache(group_size=layer.group_size, bits=layer.bits)
            nc.state = (
                tuple(t[:, :, :m, :] for t in keys),
                tuple(t[:, :, :m, :] for t in values),
            )
            nc.offset = m
            out.append(nc)
        else:
            out.append(layer)  # recurrent: whole snapshot
    mx.eval([layer.state for layer in out])
    return out


def _tokens(n: int, salt: int = 0) -> list[int]:
    return [_TOKEN_BASE + salt * 100000 + i for i in range(n)]


def _assert_attention_bit_identical(expected: list, got: list) -> None:
    """Every attention layer's (keys, values) must be byte-identical."""
    for i, (e, g) in enumerate(zip(expected, got)):
        cls = type(e).__name__
        if cls == "KVCache":
            ek, ev = e.state
            gk, gv = g.state
            assert gk.shape == ek.shape, f"layer {i} key shape {gk.shape}!={ek.shape}"
            assert mx.array_equal(ek, gk).item(), f"layer {i} keys differ"
            assert mx.array_equal(ev, gv).item(), f"layer {i} values differ"
        elif cls == "QuantizedKVCache":
            ek, ev = e.state
            gk, gv = g.state
            for j in range(3):
                assert mx.array_equal(
                    ek[j], gk[j]
                ).item(), f"layer {i} qkey[{j}] differ"
                assert mx.array_equal(
                    ev[j], gv[j]
                ).item(), f"layer {i} qval[{j}] differ"
            assert g.offset == e.offset


def _write_link(
    root: str,
    *,
    salt: int,
    idx: int,
    offset: int,
    cache_at_offset: list,
    tokens: list[int],
    kind: str = "full",
    delta_cache=None,
    delta_meta=None,
    save_uuid: str | None = None,
    index: bool = True,
) -> tuple[str, str]:
    """Write one checkpoint link and (optionally) register it in the content index.

    Returns ``(path, save_uuid)``. The tokens blob is written so the chain
    resolver + content index can verify prefixes.
    """
    rh = _dkc.request_hash(f"s{salt}-link{idx}", model_name="m")
    uu = save_uuid or f"uuid-{salt}-{idx}"
    path = _dkc.write_checkpoint(
        cache_at_offset,
        root=root,
        req_hash=rh,
        token_offset=offset,
        model_name="m",
        extra_metadata={"tokens_key": list(tokens[:offset]), "save_uuid": uu},
        kind=kind,
        delta_cache=delta_cache,
        delta_meta=delta_meta,
    )
    assert path is not None
    if index:
        _dkc.get_content_index().index_checkpoint(
            list(tokens[:offset]),
            root=root,
            req_hash=rh,
            token_offset=offset,
            save_uuid=uu,
        )
    return path, uu


def _build_chain(
    root: str,
    offsets: list[int],
    *,
    salt: int = 0,
    quantized: bool = False,
    with_recurrent: bool = True,
    seed: int = 0,
) -> tuple[list, list[int], list[dict]]:
    """Grow a session across ``offsets``, planning delta-vs-keyframe exactly like
    the scheduler does (longest strict-prefix parent + ``should_write_keyframe``).

    Returns ``(canonical_full_cache, tokens, links)`` where each link dict has
    ``path``/``offset``/``kind``/``req_hash``.
    """
    canon = _canonical_full(
        offsets[-1], with_recurrent=with_recurrent, quantized=quantized, seed=seed
    )
    tokens = _tokens(offsets[-1], salt=salt)
    index = _dkc.get_content_index()
    links: list[dict] = []
    for i, off in enumerate(offsets):
        cache_at = _slice_full_to(canon, off)
        kind, dcache, dmeta = "full", None, None
        parent = index.longest_strict_prefix(tokens[:off])
        if parent is not None:
            pside = _dkc.read_sidecar(root, parent.req_hash, parent.token_offset)
            pdepth = int((pside or {}).get("chain_depth", 0) or 0)
            b_off = int(parent.token_offset)
            if (
                pside is not None
                and not _dkc.should_write_keyframe(pdepth)
                and 0 < b_off < off
            ):
                built = _dkc.build_delta_cache(cache_at, b_off, off)
                if built is not None:
                    kind = "delta"
                    dcache = built
                    dmeta = {
                        "base_hash": parent.req_hash,
                        "base_offset": b_off,
                        "base_save_uuid": parent.save_uuid,
                        "delta_range": [b_off, off],
                        "chain_depth": pdepth + 1,
                    }
        rh = _dkc.request_hash(f"s{salt}-link{i}", model_name="m")
        uu = f"uuid-{salt}-{i}"
        path = _dkc.write_checkpoint(
            cache_at,
            root=root,
            req_hash=rh,
            token_offset=off,
            model_name="m",
            extra_metadata={"tokens_key": list(tokens[:off]), "save_uuid": uu},
            kind=kind,
            delta_cache=dcache,
            delta_meta=dmeta,
        )
        assert path is not None
        index.index_checkpoint(
            list(tokens[:off]), root=root, req_hash=rh, token_offset=off, save_uuid=uu
        )
        links.append({"path": path, "offset": off, "kind": kind, "req_hash": rh})
    return canon, tokens, links


# ---------------------------------------------------------------------------
# (a) chained restore is bit-identical to a full checkpoint
# ---------------------------------------------------------------------------


def test_a_chained_restore_bit_identical_to_full(root: str):
    canon = _canonical_full(400, with_recurrent=True)
    tokens = _tokens(400)
    # base (full) at 300, delta over [300, 400).
    _write_link(
        root,
        salt=0,
        idx=0,
        offset=300,
        cache_at_offset=_slice_full_to(canon, 300),
        tokens=tokens,
    )
    parent = _dkc.get_content_index().longest_strict_prefix(tokens)
    assert parent is not None and parent.token_offset == 300
    dcache = _dkc.build_delta_cache(canon, 300, 400)
    assert dcache is not None
    leaf_path, _ = _write_link(
        root,
        salt=0,
        idx=1,
        offset=400,
        cache_at_offset=canon,
        tokens=tokens,
        kind="delta",
        delta_cache=dcache,
        delta_meta={
            "base_hash": parent.req_hash,
            "base_offset": 300,
            "base_save_uuid": parent.save_uuid,
            "delta_range": [300, 400],
            "chain_depth": 1,
        },
    )
    side = _dkc.read_sidecar(root, _dkc.request_hash("s0-link1", "m"), 400)
    assert side["kind"] == "delta" and side["delta_range"] == [300, 400]

    loaded = _dkc.load_checkpoint_chain(leaf_path)
    assert loaded is not None and loaded.token_offset == 400
    _assert_attention_bit_identical(canon, loaded.cache)
    # recurrent layer comes from the newest link == canonical.
    assert mx.array_equal(canon[-1].state[0], loaded.cache[-1].state[0]).item()
    # full-checkpoint cross-check: the assembled attention equals a full write.
    assert _dkc._cache_offset_matches(loaded.cache, 400)


def test_a2_recurrent_taken_from_newest_link_only(root: str):
    """The recurrent (ArraysCache) state must come from the NEWEST link, never a
    base/intermediate link. Give the base a DISTINCT recurrent snapshot from the
    leaf and assert the restore matches the leaf's, not the base's.
    """
    canon = _canonical_full(400, with_recurrent=True, seed=0)
    tokens = _tokens(400, salt=20)
    leaf_recurrent = [x for x in canon[-1].state]
    # Base carries a DIFFERENT recurrent snapshot (as it would in a real session,
    # where recurrent state evolves token to token).
    base_cache = _slice_full_to(canon, 200)
    distinct_rec = ArraysCache(2)
    distinct_rec[0] = mx.random.normal((1, 4, 16, 16), key=mx.random.key(4242))
    distinct_rec[1] = mx.random.normal((1, 3, 12), key=mx.random.key(4243))
    mx.eval(distinct_rec.state)
    base_cache[-1] = distinct_rec
    # sanity: the two recurrent snapshots really differ.
    assert not mx.array_equal(distinct_rec.state[0], leaf_recurrent[0]).item()

    _write_link(
        root, salt=20, idx=0, offset=200, cache_at_offset=base_cache, tokens=tokens
    )
    parent = _dkc.get_content_index().longest_strict_prefix(tokens)
    dcache = _dkc.build_delta_cache(canon, 200, 400)
    leaf_path, _ = _write_link(
        root,
        salt=20,
        idx=1,
        offset=400,
        cache_at_offset=canon,
        tokens=tokens,
        kind="delta",
        delta_cache=dcache,
        delta_meta={
            "base_hash": parent.req_hash,
            "base_offset": 200,
            "base_save_uuid": parent.save_uuid,
            "delta_range": [200, 400],
            "chain_depth": 1,
        },
    )
    loaded = _dkc.load_checkpoint_chain(leaf_path)
    assert loaded is not None
    restored_rec = loaded.cache[-1].state
    # Must equal the LEAF's recurrent, and must NOT equal the base's.
    assert mx.array_equal(restored_rec[0], leaf_recurrent[0]).item()
    assert mx.array_equal(restored_rec[1], leaf_recurrent[1]).item()
    assert not mx.array_equal(restored_rec[0], distinct_rec.state[0]).item()


def test_a_lookup_routes_delta_through_chain(root: str):
    """The content-index lookup must assemble a delta leaf via the chain (its own
    short attention would otherwise fail the offset guard)."""
    canon = _canonical_full(512, with_recurrent=True)
    tokens = _tokens(512)
    _write_link(
        root,
        salt=1,
        idx=0,
        offset=256,
        cache_at_offset=_slice_full_to(canon, 256),
        tokens=tokens,
    )
    parent = _dkc.get_content_index().longest_strict_prefix(tokens)
    dcache = _dkc.build_delta_cache(canon, 256, 512)
    _write_link(
        root,
        salt=1,
        idx=1,
        offset=512,
        cache_at_offset=canon,
        tokens=tokens,
        kind="delta",
        delta_cache=dcache,
        delta_meta={
            "base_hash": parent.req_hash,
            "base_offset": 256,
            "base_save_uuid": parent.save_uuid,
            "delta_range": [256, 512],
            "chain_depth": 1,
        },
    )
    loaded = _dkc.get_content_index().lookup(tokens)
    assert loaded is not None
    assert loaded.token_offset == 512
    _assert_attention_bit_identical(canon, loaded.cache)


# ---------------------------------------------------------------------------
# (b) evict-on-fail: broken chain falls back to the base's shorter prefix
# ---------------------------------------------------------------------------


def test_b_broken_chain_falls_back_to_base_prefix(root: str):
    canon, tokens, links = _build_chain(root, [128, 256, 384], salt=2)
    # link0 full base @128, link1 delta @256, link2 delta @384.
    assert [x["kind"] for x in links] == ["full", "delta", "delta"]
    # Break the MIDDLE link (delta @256): remove its body + sidecar + tokens.
    mid = links[1]
    for ext in (".safetensors", ".json", ".tokens.bin"):
        p = mid["path"].replace(".safetensors", ext)
        if os.path.exists(p):
            os.unlink(p)

    # Lookup for the full prompt: the @384 leaf can't assemble (mid gone), the
    # @256 link is unreadable, so the loop must fall back to the @128 base.
    loaded = _dkc.get_content_index().lookup(tokens)
    assert loaded is not None
    assert loaded.token_offset == 128
    _assert_attention_bit_identical(_slice_full_to(canon, 128), loaded.cache)
    # The broken leaf was dropped from the index (fallback loop advanced).
    assert 384 not in {
        r.token_offset for r in _dkc.get_content_index()._by_key.values()
    }


def test_b2_quarantining_a_delta_leaf_decrefs_its_base(root: str):
    """A HARD-corrupt delta leaf that gets quarantined during lookup must release
    the +1 it held on its base — otherwise the base is pinned against eviction
    for the whole process lifetime (the quarantine refcount-leak path)."""
    canon, tokens, links = _build_chain(root, [200, 400], salt=21)
    base, delta = links[0], links[1]
    assert delta["kind"] == "delta"
    assert _dkc._persistent_refcount.get(f"{base['req_hash']}:200", 0) == 1
    # Corrupt the delta's tokens blob (flip a token byte → HARD content mismatch,
    # not an io_error) so lookup quarantines the leaf.
    tok = delta["path"].replace(".safetensors", ".tokens.bin")
    data = bytearray(Path(tok).read_bytes())
    data[-1] ^= 0xFF
    Path(tok).write_bytes(bytes(data))

    loaded = _dkc.get_content_index().lookup(tokens)
    # Falls back to the base's shorter prefix after quarantining the leaf.
    assert loaded is not None and loaded.token_offset == 200
    # The quarantined delta released its hold on the base.
    assert _dkc._persistent_refcount.get(f"{base['req_hash']}:200", 0) == 0
    assert f"{delta['req_hash']}:400" not in _dkc._child_base_edge


# ---------------------------------------------------------------------------
# (c) eviction never orphans a base with a live descendant
# ---------------------------------------------------------------------------


def test_c_eviction_never_orphans_live_base(root: str):
    canon, tokens, links = _build_chain(root, [256, 512], salt=3)
    base, delta = links[0], links[1]
    assert base["kind"] == "full" and delta["kind"] == "delta"
    # Make the base the OLDEST file so a naive oldest-first eviction would take
    # it first; the refcount guard must protect it while the delta is alive.
    old = 1_000.0
    for ext in (".safetensors", ".json", ".tokens.bin"):
        for link in (base, delta):
            p = link["path"].replace(".safetensors", ext)
            if os.path.exists(p):
                os.utime(p, (old, old) if link is base else (old + 500, old + 500))
    assert _dkc._persistent_refcount.get(f"{base['req_hash']}:256", 0) == 1

    # Cap below the total so eviction must free something.
    total = sum(s for _, _, s in _dkc.scan_checkpoints(root))
    _dkc.enforce_disk_cap(root, max_bytes=max(1, total // 2), low_water_fraction=0.5)

    # Invariant 3: the base is never unlinked while its delta is on disk.
    base_present = os.path.exists(base["path"])
    delta_present = os.path.exists(delta["path"])
    if delta_present:
        assert base_present, "base evicted while a live descendant remained"
    # The evictable leaf went; the protected base stayed.
    assert base_present
    assert not delta_present
    # After the delta is evicted its base's refcount is decremented back to 0
    # (edge dropped) — otherwise the base would be pinned forever and never
    # reclaimable. Guards the decref path (a leak here would pass without this).
    assert _dkc._persistent_refcount.get(f"{base['req_hash']}:256", 0) == 0
    assert f"{delta['req_hash']}:512" not in _dkc._child_base_edge
    # A follow-up eviction pass can now reclaim the (unprotected) base.
    _dkc.enforce_disk_cap(root, max_bytes=1, low_water_fraction=1.0)
    assert not os.path.exists(base["path"])


# ---------------------------------------------------------------------------
# (d) keyframe interval caps chain length
# ---------------------------------------------------------------------------


def test_d_keyframe_caps_chain_length(root: str, monkeypatch):
    monkeypatch.setenv("DELTA_CHECKPOINTS_KEYFRAME_INTERVAL", "3")
    assert _dkc.delta_keyframe_interval() == 3
    # should_write_keyframe fires when child depth reaches N.
    assert not _dkc.should_write_keyframe(0)  # child depth 1 -> delta
    assert not _dkc.should_write_keyframe(1)  # child depth 2 -> delta
    assert _dkc.should_write_keyframe(2)  # child depth 3 -> keyframe (full)

    offsets = [64 * (i + 1) for i in range(10)]
    _canon, _tokens_, links = _build_chain(root, offsets, salt=4)
    depths = []
    kinds = []
    for link in links:
        side = _dkc.read_sidecar(root, link["req_hash"], link["offset"])
        depths.append(int(side.get("chain_depth", 0)))
        kinds.append(side.get("kind"))
    # No on-disk chain deeper than N-1.
    assert max(depths) <= 2, depths
    # Keyframes (full bases) recur: depth resets to 0 at every third link.
    assert kinds[0] == "full" and depths[0] == 0
    assert "full" in kinds[3:], kinds  # at least one later keyframe
    # The deepest leaf still restores and walks at most N links.
    leaf = links[-1]
    chain = _dkc._resolve_chain(leaf["path"])
    assert chain is not None and len(chain) <= 3


# ---------------------------------------------------------------------------
# (e) delta ranges stay contiguous under spec-decode overshoot
# ---------------------------------------------------------------------------


def test_e_delta_ranges_contiguous_under_overshoot(root: str):
    # Irregular boundaries (spec decode advances several tokens at once): 258,515.
    canon = _canonical_full(515, with_recurrent=True)
    tokens = _tokens(515)
    _write_link(
        root,
        salt=5,
        idx=0,
        offset=258,
        cache_at_offset=_slice_full_to(canon, 258),
        tokens=tokens,
    )
    parent = _dkc.get_content_index().longest_strict_prefix(tokens)
    assert parent.token_offset == 258
    dcache = _dkc.build_delta_cache(canon, 258, 515)
    leaf_path, _ = _write_link(
        root,
        salt=5,
        idx=1,
        offset=515,
        cache_at_offset=canon,
        tokens=tokens,
        kind="delta",
        delta_cache=dcache,
        delta_meta={
            "base_hash": parent.req_hash,
            "base_offset": 258,
            "base_save_uuid": parent.save_uuid,
            "delta_range": [258, 515],
            "chain_depth": 1,
        },
    )
    side = _dkc.read_sidecar(root, _dkc.request_hash("s5-link1", "m"), 515)
    # Range taken from the parent's ACTUAL stamped offset, not offset-256.
    assert side["delta_range"] == [258, 515]
    loaded = _dkc.load_checkpoint_chain(leaf_path)
    assert loaded is not None and loaded.token_offset == 515
    _assert_attention_bit_identical(canon, loaded.cache)


# ---------------------------------------------------------------------------
# (f) two concurrent restores sharing a base
# ---------------------------------------------------------------------------


def test_f_concurrent_restore_shared_base(root: str):
    # Shared base @200; two distinct leaves branch off it.
    shared = _tokens(200, salt=6)
    canonA = _canonical_full(360, with_recurrent=True, seed=10)
    canonB = _canonical_full(360, with_recurrent=True, seed=77)
    # Make both canon share the SAME first-200 attention so a single base serves
    # both (branching conversation): rebuild A/B from a shared 200-prefix.
    base_cache = _canonical_full(200, with_recurrent=True, seed=5)
    for canon in (canonA, canonB):
        for i, bl in enumerate(base_cache):
            if type(bl).__name__ == "KVCache":
                k, v = canon[i].state
                bk, bv = bl.state
                nk = mx.concatenate([bk, k[:, :, 200:, :]], axis=2)
                nv = mx.concatenate([bv, v[:, :, 200:, :]], axis=2)
                canon[i].state = (nk, nv)
        mx.eval([layer.state for layer in canon])
    tokensA = shared + _tokens(160, salt=61)
    tokensB = shared + _tokens(160, salt=62)

    # One base, two deltas (one per leaf) pointing at it.
    rh_base = _dkc.request_hash("s6-base", "m")
    _dkc.write_checkpoint(
        _slice_full_to(base_cache, 200),
        root=root,
        req_hash=rh_base,
        token_offset=200,
        model_name="m",
        extra_metadata={"tokens_key": shared, "save_uuid": "uuid-base6"},
    )

    def _leaf(canon, tokens, salt):
        dcache = _dkc.build_delta_cache(canon, 200, 360)
        rh = _dkc.request_hash(f"s6-leaf{salt}", "m")
        p = _dkc.write_checkpoint(
            canon,
            root=root,
            req_hash=rh,
            token_offset=360,
            model_name="m",
            extra_metadata={"tokens_key": tokens, "save_uuid": f"uuid-leaf{salt}"},
            kind="delta",
            delta_cache=dcache,
            delta_meta={
                "base_hash": rh_base,
                "base_offset": 200,
                "base_save_uuid": "uuid-base6",
                "delta_range": [200, 360],
                "chain_depth": 1,
            },
        )
        return p

    pathA = _leaf(canonA, tokensA, "A")
    pathB = _leaf(canonB, tokensB, "B")
    assert _dkc._persistent_refcount.get(f"{rh_base}:200", 0) == 2

    results: dict[str, object] = {}

    def _restore(name, path):
        results[name] = _dkc.load_checkpoint_chain(path)

    tA = threading.Thread(target=_restore, args=("A", pathA))
    tB = threading.Thread(target=_restore, args=("B", pathB))
    tA.start()
    tB.start()
    tA.join()
    tB.join()

    assert results["A"] is not None and results["B"] is not None
    _assert_attention_bit_identical(canonA, results["A"].cache)
    _assert_attention_bit_identical(canonB, results["B"].cache)
    # Transient read locks fully released after both restores.
    assert not _dkc._transient_refcount


# ---------------------------------------------------------------------------
# (g) mixed-schema chain: v1 base + v2 delta
# ---------------------------------------------------------------------------


def test_g_mixed_schema_v1_base_v2_delta(root: str):
    canon = _canonical_full(400, with_recurrent=True)
    tokens = _tokens(400, salt=7)
    base_path, base_uuid = _write_link(
        root,
        salt=7,
        idx=0,
        offset=200,
        cache_at_offset=_slice_full_to(canon, 200),
        tokens=tokens,
    )
    # Rewrite the base sidecar as a LEGACY v1 checkpoint: schema_version 1, no
    # ``kind`` key (implicit full base). Keep save_uuid so the delta can pin it.
    side_path = base_path.replace(".safetensors", ".json")
    with open(side_path) as fh:
        base_side = json.load(fh)
    base_side["schema_version"] = 1
    base_side.pop("kind", None)
    base_side.pop("chain_depth", None)
    with open(side_path, "w") as fh:
        json.dump(base_side, fh)

    parent = _dkc.get_content_index().longest_strict_prefix(tokens)
    assert parent.token_offset == 200
    dcache = _dkc.build_delta_cache(canon, 200, 400)
    leaf_path, _ = _write_link(
        root,
        salt=7,
        idx=1,
        offset=400,
        cache_at_offset=canon,
        tokens=tokens,
        kind="delta",
        delta_cache=dcache,
        delta_meta={
            "base_hash": parent.req_hash,
            "base_offset": 200,
            "base_save_uuid": base_uuid,
            "delta_range": [200, 400],
            "chain_depth": 1,
        },
    )
    # The chain resolver treats the v1 (kind-less) sidecar as an implicit base.
    loaded = _dkc.load_checkpoint_chain(leaf_path)
    assert loaded is not None and loaded.token_offset == 400
    _assert_attention_bit_identical(canon, loaded.cache)


# ---------------------------------------------------------------------------
# (h) quantized-layer slice correctness through a chain
# ---------------------------------------------------------------------------


def test_h_quantized_slice_chain_bit_exact(root: str):
    canon = _canonical_full(384, with_recurrent=True, quantized=True)
    tokens = _tokens(384, salt=8)
    _write_link(
        root,
        salt=8,
        idx=0,
        offset=192,
        cache_at_offset=_slice_full_to(canon, 192),
        tokens=tokens,
    )
    parent = _dkc.get_content_index().longest_strict_prefix(tokens)
    dcache = _dkc.build_delta_cache(canon, 192, 384)
    assert dcache is not None
    # The delta's quantized attention layer carries exactly the [192:384] rows.
    q_delta = dcache[0]
    assert q_delta.offset == 192
    assert q_delta.state[0][0].shape[2] == 192  # packed keys, token axis
    leaf_path, _ = _write_link(
        root,
        salt=8,
        idx=1,
        offset=384,
        cache_at_offset=canon,
        tokens=tokens,
        kind="delta",
        delta_cache=dcache,
        delta_meta={
            "base_hash": parent.req_hash,
            "base_offset": 192,
            "base_save_uuid": parent.save_uuid,
            "delta_range": [192, 384],
            "chain_depth": 1,
        },
    )
    loaded = _dkc.load_checkpoint_chain(leaf_path)
    assert loaded is not None and loaded.token_offset == 384
    # Bit-exact on the packed uint32 / scales / biases — slicing+concat of a
    # quantized cache on the token axis must not perturb any component.
    _assert_attention_bit_identical(canon, loaded.cache)


# ---------------------------------------------------------------------------
# (i) long chain (50+ links) restores via layer-at-a-time concat
# ---------------------------------------------------------------------------


def test_i_long_chain_restore_bounded(root: str, monkeypatch):
    # Disable keyframing so the whole run is ONE deep chain (55 links).
    monkeypatch.setenv("DELTA_CHECKPOINTS_KEYFRAME_INTERVAL", "100000")
    offsets = [16 * (i + 1) for i in range(55)]
    canon, tokens, links = _build_chain(root, offsets, salt=9)
    assert links[0]["kind"] == "full"
    assert sum(1 for x in links if x["kind"] == "delta") == 54

    leaf = links[-1]
    chain = _dkc._resolve_chain(leaf["path"])
    assert chain is not None and len(chain) == 55  # base + 54 deltas

    loaded = _dkc.load_checkpoint_chain(leaf["path"])
    assert loaded is not None
    assert loaded.token_offset == offsets[-1]
    _assert_attention_bit_identical(canon, loaded.cache)
    # Transient locks released; the deep read never leaked a refcount.
    assert not _dkc._transient_refcount


# ---------------------------------------------------------------------------
# Observability (design §7): delta-checkpoint metrics
# ---------------------------------------------------------------------------


def test_obs_delta_bytes_saved_ticks_on_delta_write(root: str):
    """A committed delta write bumps ``delta_bytes_saved`` by the omitted
    attention slice; a full base write leaves it untouched."""
    before = _dkc.get_stats()["delta_bytes_saved"]
    canon = _canonical_full(400, with_recurrent=True)
    tokens = _tokens(400, salt=30)
    # Full base @200 saves nothing.
    _write_link(
        root,
        salt=30,
        idx=0,
        offset=200,
        cache_at_offset=_slice_full_to(canon, 200),
        tokens=tokens,
    )
    assert _dkc.get_stats()["delta_bytes_saved"] == before
    # Delta @400 over [200, 400) saves the [0, 200) attention rows.
    parent = _dkc.get_content_index().longest_strict_prefix(tokens)
    dcache = _dkc.build_delta_cache(canon, 200, 400)
    _write_link(
        root,
        salt=30,
        idx=1,
        offset=400,
        cache_at_offset=canon,
        tokens=tokens,
        kind="delta",
        delta_cache=dcache,
        delta_meta={
            "base_hash": parent.req_hash,
            "base_offset": 200,
            "base_save_uuid": parent.save_uuid,
            "delta_range": [200, 400],
            "chain_depth": 1,
        },
    )
    saved = _dkc.get_stats()["delta_bytes_saved"] - before
    # Two attention layers, 2 kv-heads, head_dim 8, bf16 (2 bytes), 200 omitted
    # rows, keys + values: 2*(1*2*200*8*2)*2 = 25600 bytes.
    assert saved == 2 * (1 * 2 * 200 * 8 * 2) * 2


def test_obs_chain_metrics_tick_on_assembly(root: str):
    """A successful chain assembly sets ``chain_length`` and adds to
    ``restore_link_count``."""
    canon, tokens, links = _build_chain(root, [128, 256, 384], salt=31)
    assert [x["kind"] for x in links] == ["full", "delta", "delta"]
    before_links = _dkc.get_stats()["restore_link_count"]
    loaded = _dkc.load_checkpoint_chain(links[-1]["path"])
    assert loaded is not None
    stats = _dkc.get_stats()
    assert stats["chain_length"] == 3  # base + 2 deltas
    assert stats["restore_link_count"] == before_links + 3


def test_obs_orphan_event_ticks_on_broken_chain(root: str):
    """A chain whose base is missing bumps ``orphan_events`` and warns."""
    canon, tokens, links = _build_chain(root, [128, 256], salt=32)
    base, delta = links[0], links[1]
    assert delta["kind"] == "delta"
    # Remove the base body/sidecar/tokens so the delta can't resolve its chain.
    for ext in (".safetensors", ".json", ".tokens.bin"):
        p = base["path"].replace(".safetensors", ext)
        if os.path.exists(p):
            os.unlink(p)
    before = _dkc.get_stats()["orphan_events"]
    assert _dkc.load_checkpoint_chain(delta["path"]) is None
    assert _dkc.get_stats()["orphan_events"] == before + 1


# ---------------------------------------------------------------------------
# MAJOR 2: delta write is gated on model identity, not just the flag
# ---------------------------------------------------------------------------


def _run_persist_boundary(root, monkeypatch, model_name, tokens, canon):
    """Drive Scheduler._disk_persist_boundary with a stub ``self`` for two
    boundaries (256 then full len) and return the second checkpoint's ``kind``.

    Writes to ``root`` (via a monkeypatched default-root) so a full-prefix
    parent exists for the second write; the delta-vs-full decision is then made
    entirely by the gate under test.
    """
    from types import SimpleNamespace

    from vllm_mlx.scheduler import Scheduler

    monkeypatch.setattr(_dkc, "get_default_root", lambda: root)
    cfg = SimpleNamespace(
        kv_disk_checkpoint_interval=256,
        kv_cache_turboquant=False,
        kv_cache_quantization=False,
        kv_cache_min_quantize_tokens=10**9,
        kv_cache_dtype="bf16",
    )
    stub = SimpleNamespace(config=cfg, _model_name=model_name)
    n = len(tokens)
    # Boundary 1: the base (full) at 256.
    Scheduler._disk_persist_boundary(stub, tokens[:256], _slice_full_to(canon, 256))
    # Boundary 2: the full-length write, which discovers the 256 base as parent.
    Scheduler._disk_persist_boundary(stub, tokens, canon)
    import array as _arr
    import hashlib as _hl

    raw = _arr.array("i", (int(t) for t in tokens)).tobytes()
    req_hash = _hl.sha256(str(model_name).encode() + raw).hexdigest()[:16]
    side = _dkc.read_sidecar(root, req_hash, n)
    assert side is not None, "second boundary did not write a checkpoint"
    return side.get("kind")


def test_major2_full_only_model_never_deltas(root: str, monkeypatch):
    """A model in MODELS_REQUIRING_FULL_CHECKPOINT (gemma-4) must never emit a
    delta even with DELTA_CHECKPOINTS_ENABLED on and a valid parent present."""
    assert _dkc.model_requires_full_checkpoint("gemma-4")
    canon = _canonical_full(512, with_recurrent=True, seed=40)
    tokens = _tokens(512, salt=40)
    kind = _run_persist_boundary(root, monkeypatch, "gemma-4", tokens, canon)
    assert kind == "full"


def test_major2_sliceable_model_does_delta(root: str, monkeypatch):
    """Control for the gate: the SAME setup with a delta-sliceable model
    (qwen3.5) DOES produce a delta, proving parent discovery works and the gate
    is the only reason the full-only model above wrote full."""
    assert not _dkc.model_requires_full_checkpoint("qwen3.5")
    canon = _canonical_full(512, with_recurrent=True, seed=41)
    tokens = _tokens(512, salt=41)
    kind = _run_persist_boundary(root, monkeypatch, "qwen3.5-9b", tokens, canon)
    assert kind == "delta"


# ---------------------------------------------------------------------------
# MINOR 4: restore-consistency guard rejects a partial checkpoint for a
# full-only model (scheduler.py guard (d): expected_full and not requires_full)
# ---------------------------------------------------------------------------


class _HonestStub:
    def __init__(self):
        self.calls = []

    def record_disk_restore(self, hit):
        self.calls.append(hit)


def _write_full_only_delta_leaf(root, model_name):
    """Write a base + delta chain for ``model_name`` whose leaf sidecar is
    stamped ``requires_full_checkpoint=False`` (a genuinely partial checkpoint),
    indexed so the content-index lookup finds it. Returns the query tokens."""
    canon = _canonical_full(400, with_recurrent=True, seed=50)
    tokens = _tokens(400, salt=50)
    rh_base = _dkc.request_hash("guard-base", model_name)
    _dkc.write_checkpoint(
        _slice_full_to(canon, 200),
        root=root,
        req_hash=rh_base,
        token_offset=200,
        model_name=model_name,
        requires_full_checkpoint=False,
        extra_metadata={"tokens_key": list(tokens[:200]), "save_uuid": "u-gbase"},
    )
    _dkc.get_content_index().index_checkpoint(
        list(tokens[:200]),
        root=root,
        req_hash=rh_base,
        token_offset=200,
        save_uuid="u-gbase",
    )
    dcache = _dkc.build_delta_cache(canon, 200, 400)
    rh_leaf = _dkc.request_hash("guard-leaf", model_name)
    _dkc.write_checkpoint(
        canon,
        root=root,
        req_hash=rh_leaf,
        token_offset=400,
        model_name=model_name,
        requires_full_checkpoint=False,
        extra_metadata={"tokens_key": list(tokens), "save_uuid": "u-gleaf"},
        kind="delta",
        delta_cache=dcache,
        delta_meta={
            "base_hash": rh_base,
            "base_offset": 200,
            "base_save_uuid": "u-gbase",
            "delta_range": [200, 400],
            "chain_depth": 1,
        },
    )
    _dkc.get_content_index().index_checkpoint(
        list(tokens),
        root=root,
        req_hash=rh_leaf,
        token_offset=400,
        save_uuid="u-gleaf",
    )
    return tokens


def test_minor4_partial_rejected_for_full_only_model(root: str, monkeypatch):
    """A delta (partial) checkpoint stamped requires_full=False is REJECTED
    for a full-only model (gemma-4): guard (d) bumps full_checkpoint_mismatch
    and the request stays a miss. Reverting the guard line lets the partial
    install (cache_hit_type becomes 'disk'), which flips both assertions."""
    from types import SimpleNamespace

    from vllm_mlx.scheduler import Scheduler

    monkeypatch.setattr(_dkc, "get_default_root", lambda: root)
    tokens = _write_full_only_delta_leaf(root, "gemma-4")

    cfg = SimpleNamespace(kv_disk_restore_enabled=True, kv_cache_dtype="bf16")
    stub = SimpleNamespace(
        config=cfg,
        _model_name="gemma-4",
        _disk_restore_index_built=True,
        honest_metrics=_HonestStub(),
        _resolve_metal_cap_bytes=lambda: 0,
        _current_metal_active_bytes=lambda: 0,
    )
    request = SimpleNamespace(
        request_id="req-guard-0001",
        cache_hit_type="miss",
        prompt_cache=None,
        prompt_token_ids=list(tokens),
        cached_tokens=0,
        remaining_tokens=None,
    )

    before = _dkc.get_stats()["restore_rejects"].get("full_checkpoint_mismatch", 0)
    Scheduler._maybe_disk_restore(stub, request, pflash_compressed=False)

    after = _dkc.get_stats()["restore_rejects"].get("full_checkpoint_mismatch", 0)
    assert after == before + 1, "guard (d) did not reject the partial checkpoint"
    # The request was NOT restored: it stays a miss and re-prefills.
    assert request.cache_hit_type == "miss"
    assert request.prompt_cache is None
    # Exactly one attempt recorded, as a miss.
    assert stub.honest_metrics.calls == [False]
