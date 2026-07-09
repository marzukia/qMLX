# SPDX-License-Identifier: Apache-2.0
"""End-to-end round-trip regression for hybrid-cache disk checkpointing.

This encodes Phase 0 of the disk-KV restore feature (R15 #303): the proof
that ``mlx_lm.save_prompt_cache`` / ``load_prompt_cache`` round-trips this
project's hybrid model cache (ArraysCache recurrent layers + KVCache
attention layers) so faithfully that generation *resumed* from a reloaded
checkpoint is TOKEN-IDENTICAL to generation continued in-process.

Unlike ``tests/test_disk_kv_checkpoint.py`` (which byte-compares synthetic
one-layer caches without a model), this test drives a real forward pass:
prefill > 256 tokens, snapshot the cache with ``write_checkpoint``, then
compare greedy continuation from the live cache against greedy continuation
from the cache ``load_checkpoint`` reads back. The assertion is on the
decoded token *ids* (a Python ``list[int]`` equality), NOT ``mx.array_equal``
on the KV tensors — a byte-identical cache that still decoded differently
would be the real regression, and only comparing emitted tokens catches it.

Requires a real model, so it SKIPS unless:

* ``mlx`` and ``mlx_lm`` import, and
* ``RAPID_MLX_HYBRID_TEST_MODEL`` points at a small (ideally hybrid) MLX
  model dir/repo that ``mlx_lm.load`` can open.

CI does not set the env var, so this never demands the 122B. On an Apple
box, run with e.g.::

    RAPID_MLX_HYBRID_TEST_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        pytest tests/test_disk_kv_hybrid_roundtrip.py -v

(Any small model exercises the round-trip; point it at a hybrid checkpoint
to cover the ArraysCache + KVCache mix specifically.)
"""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from vllm_mlx.runtime import disk_kv_checkpoint as _dkc  # noqa: E402

_MODEL_ENV = "RAPID_MLX_HYBRID_TEST_MODEL"
_MIN_PROMPT_TOKENS = 257  # strictly > 256 so a checkpoint boundary is crossed
_RESUME_TOKENS = 24  # greedy tokens to compare across the round-trip


def _load_model():
    model_id = os.environ.get(_MODEL_ENV)
    if not model_id:
        pytest.skip(
            f"{_MODEL_ENV} not set; skipping hybrid round-trip "
            "(needs a real MLX model, not the 122B)"
        )
    try:
        model, tokenizer = mlx_lm.load(model_id)
    except Exception as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"could not load {model_id!r} via mlx_lm.load: {exc!r}")
    return model, tokenizer


def _build_prompt_ids(tokenizer) -> list[int]:
    """Encode a prompt of at least ``_MIN_PROMPT_TOKENS`` tokens."""
    seed = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
    )
    text = seed
    ids = tokenizer.encode(text)
    # Grow deterministically until the prompt crosses the 256-token boundary.
    while len(ids) < _MIN_PROMPT_TOKENS:
        text += seed
        ids = tokenizer.encode(text)
    return list(ids)


def _prefill(model, prompt_ids: list[int], cache) -> int:
    """Run the prompt through ``model`` into ``cache``; return the greedy next id."""
    y = mx.array([prompt_ids])
    logits = model(y, cache=cache)
    mx.eval(logits)
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _greedy_continue(model, cache, first_id: int, k: int) -> list[int]:
    """Greedy-decode ``k`` tokens from ``cache`` starting at ``first_id``.

    Mutates ``cache`` in place (the normal decode contract). Returns the
    emitted token ids as plain Python ints.
    """
    out = [first_id]
    cur = first_id
    for _ in range(k - 1):
        y = mx.array([[cur]])
        logits = model(y, cache=cache)
        mx.eval(logits)
        cur = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        out.append(cur)
    return out


def test_hybrid_checkpoint_resume_is_token_identical(tmp_path) -> None:
    """write_checkpoint -> load_checkpoint -> resume must match live resume."""
    _dkc.reset_stats_for_tests()
    model, tokenizer = _load_model()
    prompt_ids = _build_prompt_ids(tokenizer)
    assert len(prompt_ids) > 256

    root = str(tmp_path / "ckpt-root")
    req_hash = _dkc.request_hash("hybrid-roundtrip", model_name="test-model")
    offset = len(prompt_ids)

    # 1. Prefill into cache A and capture the first greedy token.
    cache_a = make_prompt_cache(model)
    first_id = _prefill(model, prompt_ids, cache_a)

    # 2. Snapshot cache A to disk BEFORE any decode mutates it.
    path = _dkc.write_checkpoint(
        cache_a,
        root=root,
        req_hash=req_hash,
        token_offset=offset,
        kv_dtype="bf16",
        requires_full_checkpoint=True,
        model_name="test-model",
    )
    assert path is not None, "write_checkpoint returned None (write failed)"
    assert os.path.isfile(path)

    # 3. Reference: continue greedily from the live (in-process) cache.
    reference = _greedy_continue(model, cache_a, first_id, _RESUME_TOKENS)

    # 4. Reload the checkpoint and continue from the reconstructed cache.
    loaded = _dkc.load_checkpoint(path)
    assert loaded is not None, "load_checkpoint refused a checkpoint it just wrote"
    assert loaded.token_offset == offset
    restored = _greedy_continue(model, loaded.cache, first_id, _RESUME_TOKENS)

    # 5. The Phase 0 contract: resumed generation is token-identical. Compare
    #    the decoded ids directly (NOT mx.array_equal on the KV tensors).
    assert restored == reference, (
        "resumed generation diverged from live generation:\n"
        f"  reference={reference}\n"
        f"  restored ={restored}"
    )

    # The successful load must have moved the loads counter.
    assert _dkc.get_stats()["loads"] >= 1
