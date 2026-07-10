# SPDX-License-Identifier: Apache-2.0
"""Correctness guard for ``_dequantize_cache``, the restore-path fix.

A restored int4 ``QuantizedKVCache`` can't have the remaining tail tokens
prefilled on top of it: this mlx-lm raises "QuantizedKVCache does not yet
support batching with history", which aborts a request that would re-prefill
fine. ``_maybe_disk_restore`` therefore runs the loaded cache through
``_dequantize_cache`` before installing it (scheduler.py). This test pins that
function's behaviour so a future edit can't silently break resume:

* a quantized attention layer comes back as a plain ``KVCache`` (so batching
  works) whose values match the pre-quantization tensor within 4-bit error,
* the ``offset`` is preserved (a dropped offset would truncate the prefix),
* non-attention layers (the hybrid model's recurrent ``ArraysCache``) and
  ``None`` slots pass through untouched.

Needs ``mlx`` + ``mlx_lm`` but NO model, so unlike the hardware round-trip
test it runs in any Apple CI without an env var.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402

from vllm_mlx.memory_cache import _dequantize_cache  # noqa: E402


def _quantized_layer(num_tokens: int, bits: int = 4, group_size: int = 64):
    """Build a KVCache, capture its bf16 tensors, return (quantized, k, v)."""
    kv = KVCache()
    # head_dim must be a multiple of group_size for the packed quant.
    k = mx.random.normal((1, 2, num_tokens, group_size), key=mx.random.key(0))
    v = mx.random.normal((1, 2, num_tokens, group_size), key=mx.random.key(1))
    kv.update_and_fetch(k, v)
    q = kv.to_quantized(group_size=group_size, bits=bits)
    return q, kv.keys, kv.values


def test_quantized_attention_layer_becomes_kvcache():
    q, _, _ = _quantized_layer(num_tokens=8)
    assert isinstance(q, QuantizedKVCache)

    out = _dequantize_cache([q])

    assert len(out) == 1
    assert isinstance(out[0], KVCache)
    assert not isinstance(out[0], QuantizedKVCache)


def test_dequantized_values_match_within_4bit_error():
    q, orig_k, orig_v = _quantized_layer(num_tokens=8)
    out = _dequantize_cache([q])[0]

    # 4-bit round-trip is lossy; assert it tracks the original, not exact.
    # The tolerance is generous by design: the point is "close", a wildly
    # wrong dequant (mismatched group_size/bits) would blow way past this.
    assert out.keys.shape == orig_k.shape
    assert out.values.shape == orig_v.shape
    assert mx.mean(mx.abs(out.keys - orig_k)).item() < 0.3
    assert mx.mean(mx.abs(out.values - orig_v)).item() < 0.3


def test_offset_is_preserved():
    q, _, _ = _quantized_layer(num_tokens=13)
    assert q.offset == 13
    out = _dequantize_cache([q])[0]
    assert out.offset == 13


def test_non_quantized_and_none_layers_pass_through_by_identity():
    # Stand-in for a recurrent ArraysCache layer: any object that isn't a
    # QuantizedKVCache must come back as the SAME object, untouched.
    class _FakeRecurrent:
        pass

    recurrent = _FakeRecurrent()
    plain = KVCache()
    plain.update_and_fetch(
        mx.zeros((1, 1, 4, 8)), mx.zeros((1, 1, 4, 8))
    )

    out = _dequantize_cache([recurrent, None, plain])

    assert out[0] is recurrent
    assert out[1] is None
    assert out[2] is plain


def test_mixed_hybrid_cache_only_touches_quantized_layers():
    class _FakeRecurrent:
        pass

    q, _, _ = _quantized_layer(num_tokens=8)
    recurrent = _FakeRecurrent()

    out = _dequantize_cache([recurrent, q])

    assert out[0] is recurrent  # recurrent layer untouched
    assert isinstance(out[1], KVCache) and not isinstance(out[1], QuantizedKVCache)
