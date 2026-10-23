# SPDX-License-Identifier: Apache-2.0
"""Image-aware disk KV checkpoint keying (issue #56).

A multimodal prompt's image content does not survive in the token stream —
every image expands into a run of the same ``image_token_id`` placeholder,
so token-only keying would let a CHANGED image reuse the previous image's
KV. These tests pin the fix:

* the stable media-content key (vllm_mlx.media_digest) is mixed into the
  boundary-checkpoint hash — text-only hash bytes stay pre-#56 exact, a
  media key changes them;
* the content index is media-bucketed — same tokens + same image hit,
  same tokens + changed image never match, text-only and image prompts
  never cross-match;
* the on-disk sidecar carries the media key so a restart's
  ``build_content_index`` re-adds image checkpoints to the SAME bucket
  (the round-trip contract from the acceptance draft);
* the scheduler store site threads ``request.media_key`` end to end.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import KVCache  # noqa: E402

from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.runtime import disk_kv_checkpoint as _dkc  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

# Two distinct image contents -> two distinct stable media keys.
IMG_A = "data:image/png;base64,aW1nLWEtYnl0ZXM="  # base64 of "img-a-bytes"
IMG_B = "data:image/png;base64,aW1nLWItdHJ1ZQ=="  # base64 of "img-b-true"


def _media_key(ref: str) -> str:
    from vllm_mlx.media_digest import media_key_for_media

    return media_key_for_media([ref])


def _seed_kv_cache(num_tokens: int = 256) -> list[KVCache]:
    cache = KVCache()
    k = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(0))
    v = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(1))
    cache.update_and_fetch(k, v)
    return [cache]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    root = tmp_path / "kv-ckpt"
    root.mkdir()
    monkeypatch.setattr(_dkc, "get_default_root", lambda: str(root))
    _dkc.reset_stats_for_tests()
    _dkc.reset_content_index_for_tests()
    return str(root)


def _write_matchable(root: str, req_id: str, tokens: list[int], media_key: str = "") -> str:
    """Write a boundary-style checkpoint (tokens blob + sidecar) and index it."""
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
        extra_metadata={
            "tokens_key": list(tokens),
            "save_uuid": "test-uuid",
            **({"media_key": media_key} if media_key else {}),
        },
    )
    assert path is not None
    return req_hash


# ---------------------------------------------------------------------------
# Content-index media bucketing
# ---------------------------------------------------------------------------


def test_same_tokens_same_image_is_a_hit(root: str):
    """Acceptance: same image + extended conversation = prefix hit."""
    tokens = list(range(1, 257))  # 256 prompt tokens (image already expanded)
    mk = _media_key(IMG_A)
    _write_matchable(root, "req-a1", tokens, media_key=mk)

    # Next turn: the SAME prompt prefix extended with new conversation tokens.
    query = tokens + [900, 901, 902]
    loaded = _dkc.get_content_index().lookup(query, mk)
    assert loaded is not None
    assert loaded.token_offset == len(tokens)


def test_changed_image_never_reuses_stale_kv(root: str):
    """Acceptance: image change -> no stale reuse, even at identical tokens."""
    tokens = list(range(1, 257))
    mk_a = _media_key(IMG_A)
    mk_b = _media_key(IMG_B)
    assert mk_a != mk_b
    _write_matchable(root, "req-a", tokens, media_key=mk_a)

    # Identical token sequence, different image content: the image-A
    # checkpoint must NOT match.
    assert _dkc.get_content_index().lookup(tokens, mk_b) is None
    # ...and a text-only query for the same tokens must not match either.
    assert _dkc.get_content_index().lookup(tokens) is None
    # ...while the matching image still hits.
    assert _dkc.get_content_index().lookup(tokens, mk_a) is not None


def test_text_only_lookup_unchanged_for_text_checkpoints(root: str):
    """Pre-#56 behaviour: text-only checkpoints (no media key) keep working
    for text-only queries and never cross-match image buckets."""
    tokens = list(range(1, 257))
    _write_matchable(root, "req-text", tokens)  # media_key=""
    idx = _dkc.get_content_index()
    assert idx.lookup(tokens) is not None
    # An image query for the same tokens must not borrow the text KV.
    mk = _media_key(IMG_A)
    assert idx.lookup(tokens, mk) is None


def test_same_tokens_both_images_coexist(root: str):
    tokens = list(range(1, 257))
    mk_a = _media_key(IMG_A)
    mk_b = _media_key(IMG_B)
    _write_matchable(root, "req-a", tokens, media_key=mk_a)
    _write_matchable(root, "req-b", tokens, media_key=mk_b)
    idx = _dkc.get_content_index()
    # Both buckets resolve; they point at different request hashes.
    la = idx.lookup(tokens, mk_a)
    lb = idx.lookup(tokens, mk_b)
    assert la is not None and lb is not None
    # Different request hashes (and therefore different on-disk dirs).
    assert la.path != lb.path


# ---------------------------------------------------------------------------
# Sidecar round-trip: restart re-indexes into the same media bucket
# ---------------------------------------------------------------------------


def test_sidecar_carries_media_key_and_round_trips(root: str):
    """Acceptance: disk KV checkpoints round-trip with image-aware keys.

    A process restart rebuilds the content index from disk; the media
    bucket must be reconstructed from the sidecar so the same image still
    hits and a changed image still misses.
    """
    tokens = list(range(1, 257))
    mk_a = _media_key(IMG_A)
    req_hash = _write_matchable(root, "req-a", tokens, media_key=mk_a)

    # Sidecar on disk carries the media key.
    sidecar = _dkc.read_sidecar(root, req_hash, len(tokens))
    assert sidecar is not None
    assert sidecar.get("media_key") == mk_a

    # Simulate a restart: wipe the in-memory index, rebuild from disk.
    _dkc.reset_content_index_for_tests()
    n = _dkc.build_content_index(root)
    assert n == 1

    idx = _dkc.get_content_index()
    # Same image -> restored from the rebuilt index (round-trip hit).
    assert idx.lookup(tokens, mk_a) is not None
    # Changed image -> still no stale reuse after the rebuild.
    assert idx.lookup(tokens, _media_key(IMG_B)) is None
    # Text-only query for image tokens -> no cross-match after rebuild.
    assert idx.lookup(tokens) is None


def test_text_sidecar_has_no_media_key_field(root: str):
    """Text-only checkpoints keep the exact pre-#56 sidecar shape (no
    media_key field) so old writers and readers interoperate."""
    tokens = list(range(1, 257))
    req_hash = _write_matchable(root, "req-text", tokens, media_key="")
    sidecar = _dkc.read_sidecar(root, req_hash, len(tokens))
    assert sidecar is not None
    assert "media_key" not in sidecar


# ---------------------------------------------------------------------------
# Scheduler store site: req_hash mixing + request threading
# ---------------------------------------------------------------------------


def _make_scheduler(interval: int = 256) -> Scheduler:
    cfg = SchedulerConfig()
    cfg.kv_disk_checkpoint_interval = interval
    cfg.kv_cache_dtype = "bf16"
    cfg.kv_disk_restore_enabled = False
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


def test_disk_persist_boundary_text_hash_unchanged(root: str):
    """Backward compatibility: a text-only boundary checkpoint's req_hash is
    byte-identical to the pre-#56 formula (sha256(model + token bytes))."""
    sched = _make_scheduler()
    tokens = list(range(1, 257))
    cache = _seed_kv_cache(num_tokens=len(tokens))
    sched._disk_persist_boundary(tokens, cache)

    raw = b"".join(int(t).to_bytes(4, "little", signed=True) for t in tokens)
    expected = hashlib.sha256(b"test-model" + raw).hexdigest()[:16]
    body = os.path.join(root, expected, f"checkpoint-{len(tokens)}.safetensors")
    assert os.path.exists(body)
    # And the text bucket is indexed (no media key in the sidecar).
    sidecar = _dkc.read_sidecar(root, expected, len(tokens))
    assert sidecar is not None and "media_key" not in sidecar


def test_disk_persist_boundary_media_key_changes_hash_and_sidecar(root: str):
    sched = _make_scheduler()
    tokens = list(range(1, 257))
    cache = _seed_kv_cache(num_tokens=len(tokens))
    mk = _media_key(IMG_A)
    sched._disk_persist_boundary(tokens, cache, media_key=mk)

    raw = b"".join(int(t).to_bytes(4, "little", signed=True) for t in tokens)
    text_hash = hashlib.sha256(b"test-model" + raw).hexdigest()[:16]
    media_hash = hashlib.sha256(
        b"test-model" + raw + b"\x00" + mk.encode()
    ).hexdigest()[:16]
    assert media_hash != text_hash
    body = os.path.join(root, media_hash, f"checkpoint-{len(tokens)}.safetensors")
    assert os.path.exists(body)
    sidecar = _dkc.read_sidecar(root, media_hash, len(tokens))
    assert sidecar is not None and sidecar.get("media_key") == mk

    # The changed image lands in a different directory with its own sidecar.
    mk_b = _media_key(IMG_B)
    sched._disk_persist_boundary(tokens, cache, media_key=mk_b)
    media_hash_b = hashlib.sha256(
        b"test-model" + raw + b"\x00" + mk_b.encode()
    ).hexdigest()[:16]
    assert os.path.exists(
        os.path.join(
            root, media_hash_b, f"checkpoint-{len(tokens)}.safetensors"
        )
    )


def test_disk_persist_boundary_threads_request_media_key(root: str):
    """The store-site wiring: a Request carrying a media_key (set by the
    engine from the media digest of its images) drives an image-aware
    boundary write — the request's image, not the prompt text, keys the
    checkpoint."""
    sched = _make_scheduler()
    tokens = list(range(1, 257))
    cache = _seed_kv_cache(num_tokens=len(tokens))
    req = Request(
        request_id="req-mm",
        prompt="ignored",
        sampling_params=SamplingParams(max_tokens=32),
        images=[IMG_A],
        media_key=_media_key(IMG_A),
    )
    sched._disk_persist_boundary(
        tokens, cache, media_key=getattr(req, "media_key", "") or ""
    )
    # The content index resolves this exact (tokens, image) pair and only it.
    idx = _dkc.get_content_index()
    assert idx.lookup(tokens, req.media_key) is not None
    assert idx.lookup(tokens, _media_key(IMG_B)) is None


def test_request_media_key_defaults_empty():
    """Text-only requests (no images) keep the default empty media key so
    nothing about their checkpointing changes."""
    req = Request(
        request_id="r1",
        prompt="hi",
        sampling_params=SamplingParams(),
    )
    assert req.media_key == ""
