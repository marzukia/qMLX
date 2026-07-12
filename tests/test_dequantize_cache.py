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
    plain.update_and_fetch(mx.zeros((1, 1, 4, 8)), mx.zeros((1, 1, 4, 8)))

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


# --- Streaming variant (fix/restore-headroom-streaming-dequant) ---------------
# ``_dequantize_cache_streaming`` is the restore-install-path dequant. It must
# produce a cache numerically IDENTICAL to ``_dequantize_cache`` (same values,
# just materialized eagerly), but frees each int4 layer as it goes by nulling
# the input slot (``cache[i] = None``), so the transient peak is ~4x + one
# layer instead of the ~5x simultaneous int4+bf16 peak. That freeing is what
# lets the headroom guard drop its multiplier 5x -> 4x.

from vllm_mlx.memory_cache import _dequantize_cache_streaming  # noqa: E402


def _mixed_fixture():
    """A couple of QuantizedKVCache layers + a None + a plain KVCache."""
    q0, _, _ = _quantized_layer(num_tokens=8, group_size=64)
    q1, _, _ = _quantized_layer(num_tokens=13, group_size=64)
    plain = KVCache()
    plain.update_and_fetch(mx.zeros((1, 1, 4, 8)), mx.ones((1, 1, 4, 8)))
    return [q0, None, q1, plain]


def test_streaming_matches_dequantize_cache_numerically():
    # Two independent-but-equal input lists: one for the reference, one for the
    # streaming (mutating) variant. Same quantized tensors in both.
    ref_in = _mixed_fixture()
    stream_in = _mixed_fixture()

    ref = _dequantize_cache(ref_in)
    out = _dequantize_cache_streaming(stream_in)

    assert len(out) == len(ref) == 4
    # None slot passes through.
    assert out[1] is None
    # Quantized layers: identical values (bit-exact, same dequantize op).
    for i in (0, 2):
        assert isinstance(out[i], KVCache) and not isinstance(out[i], QuantizedKVCache)
        assert out[i].offset == ref[i].offset
        assert mx.array_equal(out[i].keys, ref[i].keys).item()
        assert mx.array_equal(out[i].values, ref[i].values).item()
    # Plain KVCache passes through by identity.
    assert out[3] is stream_in[3]


def test_streaming_frees_int4_input_slots():
    stream_in = _mixed_fixture()
    _dequantize_cache_streaming(stream_in)

    # Consumed int4 layers nulled; None stays None; plain layer untouched.
    assert stream_in[0] is None
    assert stream_in[1] is None
    assert stream_in[2] is None
    assert isinstance(stream_in[3], KVCache)


def test_streaming_offset_preserved():
    q, _, _ = _quantized_layer(num_tokens=21)
    out = _dequantize_cache_streaming([q])[0]
    assert out.offset == 21


def test_streaming_non_quantized_and_none_pass_through():
    class _FakeRecurrent:
        pass

    recurrent = _FakeRecurrent()
    plain = KVCache()
    plain.update_and_fetch(mx.zeros((1, 1, 4, 8)), mx.zeros((1, 1, 4, 8)))

    cache = [recurrent, None, plain]
    out = _dequantize_cache_streaming(cache)

    assert out[0] is recurrent
    assert out[1] is None
    assert out[2] is plain
    # Non-int4 slots are NOT nulled (only consumed int4 layers are freed).
    assert cache[0] is recurrent
    assert cache[2] is plain
