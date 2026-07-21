# SPDX-License-Identifier: Apache-2.0
"""PFlash v2: KV pattern compression for attention layers during prefill.

Compresses the KV for full-attention layers: full KV for sink + recent
tokens, averaged/pooled KV for the middle. Applied during PREFILL when
the accumulated KV cache is large enough.

During chunked prefill, each chunk's attention sees the full accumulated
KV. We compress it before SDPA, reducing the quadratic attention cost.

DeltaNet layers are unaffected - they use recurrent state, not KV cache.

Gated behind QMLX_PFLASH_V2_ENABLED env var (default OFF).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _is_enabled() -> bool:
    val = os.environ.get("QMLX_PFLASH_V2_ENABLED", "").lower()
    if val in ("0", "false", "no"):
        return False
    return True


def _int_env(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, ""))
    except (ValueError, TypeError):
        return default


@dataclass(frozen=True)
class PFlashV2Config:
    enabled: bool = True
    sink_tokens: int = 256
    tail_tokens: int = 2048
    pool_window: int = 64
    min_seq_length: int = 4096


def config_from_env() -> PFlashV2Config:
    return PFlashV2Config(
        enabled=_is_enabled(),
        sink_tokens=_int_env("QMLX_PFLASH_V2_SINK", 256),
        tail_tokens=_int_env("QMLX_PFLASH_V2_TAIL", 2048),
        pool_window=_int_env("QMLX_PFLASH_V2_POOL", 64),
        min_seq_length=_int_env("QMLX_PFLASH_V2_MIN_SEQ", 4096),
    )


_config: PFlashV2Config | None = None


def get_config() -> PFlashV2Config:
    global _config
    if _config is None:
        _config = config_from_env()
    return _config


def compress_kv_pattern(keys, values, sink_tokens, tail_tokens, pool_window):
    """Compress K/V: sink + pooled middle + tail.

    Args:
        keys: [B, n_kv_heads, L, head_dim]
        values: [B, n_kv_heads, L, head_dim]
        sink_tokens: leading tokens at full resolution
        tail_tokens: trailing tokens at full resolution
        pool_window: middle tokens averaged into 1

    Returns:
        (compressed_keys, compressed_values, compressed_length)
    """
    import mlx.core as mx

    B, H, L, D = keys.shape

    sink = min(sink_tokens, L)
    tail = min(tail_tokens, L - sink)
    middle_start = sink
    middle_end = L - tail

    if middle_end <= middle_start or sink + tail >= L:
        return keys, values, L

    middle_len = middle_end - middle_start
    pw = min(pool_window, middle_len)
    n_pools = (middle_len + pw - 1) // pw

    middle_k = keys[:, :, middle_start:middle_end, :]
    middle_v = values[:, :, middle_start:middle_end, :]
    pad_len = n_pools * pw - middle_len
    if pad_len > 0:
        pad_k = mx.zeros((B, H, pad_len, D), dtype=middle_k.dtype)
        pad_v = mx.zeros((B, H, pad_len, D), dtype=middle_v.dtype)
        middle_k = mx.concatenate([middle_k, pad_k], axis=2)
        middle_v = mx.concatenate([middle_v, pad_v], axis=2)

    pooled_k = middle_k.reshape(B, H, n_pools, pw, D).mean(axis=3)
    pooled_v = middle_v.reshape(B, H, n_pools, pw, D).mean(axis=3)

    comp_k = mx.concatenate(
        [keys[:, :, :sink, :], pooled_k, keys[:, :, middle_end:, :]], axis=2
    )
    comp_v = mx.concatenate(
        [values[:, :, :sink, :], pooled_v, values[:, :, middle_end:, :]], axis=2
    )

    return comp_k, comp_v, sink + n_pools + tail


def install_pflash_v2(model: Any) -> bool:
    """Install PFlash v2 on attention layers. Returns True if patched."""
    cfg = get_config()
    if not cfg.enabled:
        return False

    layers = getattr(model, "layers", [])
    if not layers:
        return False

    sentinel = "_qmlx_pflash_v2_patched"
    patched_classes: set[str] = set()

    for i, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            continue

        attn = layer.self_attn
        cls = type(attn)

        if getattr(cls, sentinel, False):
            continue

        if not hasattr(attn, "q_proj") or not hasattr(attn, "k_proj"):
            continue

        original_call = cls.__call__
        v2_cfg = cfg

        def make_wrapper(orig, cfg):
            def wrapper(self_inner, x, mask=None, cache=None, **kwargs):
                import mlx.core as mx

                B, L, D = x.shape

                # During prefill (L > 1), compress KV before attention
                # once the accumulated cache is large enough
                if L > 1 and cache is not None and cfg.enabled:
                    total_seq = cache.offset + L
                    if total_seq >= cfg.min_seq_length:
                        try:
                            # Run projections
                            q_out = self_inner.q_proj(x)
                            num_heads = self_inner.num_attention_heads
                            queries, gate = mx.split(
                                q_out.reshape(B, L, num_heads, -1), 2, axis=-1
                            )
                            gate = gate.reshape(B, L, -1)

                            keys = self_inner.k_proj(x)
                            values = self_inner.v_proj(x)

                            queries = self_inner.q_norm(queries).transpose(0, 2, 1, 3)
                            keys = self_inner.k_norm(
                                keys.reshape(B, L, self_inner.num_key_value_heads, -1)
                            ).transpose(0, 2, 1, 3)
                            values = values.reshape(
                                B, L, self_inner.num_key_value_heads, -1
                            ).transpose(0, 2, 1, 3)

                            queries = self_inner.rope(queries, offset=cache.offset)
                            keys = self_inner.rope(keys, offset=cache.offset)

                            # Update cache with full K/V
                            full_keys, full_values = cache.update_and_fetch(
                                keys, values
                            )

                            # Compress full KV for attention
                            comp_k, comp_v, comp_len = compress_kv_pattern(
                                full_keys,
                                full_values,
                                cfg.sink_tokens,
                                cfg.tail_tokens,
                                cfg.pool_window,
                            )

                            from mlx_lm.models.base import scaled_dot_product_attention

                            # No mask needed for prefill with compressed KV
                            # SDPA handles causal masking internally
                            output = scaled_dot_product_attention(
                                queries,
                                comp_k,
                                comp_v,
                                cache=None,
                                scale=self_inner.scale,
                                mask=None,
                            )
                            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

                            logger.info(
                                "[PFlash v2] prefill chunk L=%d total=%d comp=%d (%.1f%%)",
                                L,
                                total_seq,
                                comp_len,
                                (1 - comp_len / total_seq) * 100,
                            )

                            return self_inner.o_proj(output * mx.sigmoid(gate))

                        except Exception as e:
                            logger.error("[PFlash v2] error: %s", e, exc_info=True)
                            return orig(self_inner, x, mask=mask, cache=cache, **kwargs)

                return orig(self_inner, x, mask=mask, cache=cache, **kwargs)

            return wrapper

        cls.__call__ = make_wrapper(original_call, v2_cfg)
        setattr(cls, sentinel, True)
        patched_classes.add(cls.__name__)

        logger.info(
            "[PFlash v2] Layer %d: patched %s.%s",
            i,
            cls.__module__,
            cls.__name__,
        )

    if patched_classes:
        logger.info(
            "[PFlash v2] Installed on %d attention classes "
            "(sink=%d, tail=%d, pool=%d, min_seq=%d)",
            len(patched_classes),
            cfg.sink_tokens,
            cfg.tail_tokens,
            cfg.pool_window,
            cfg.min_seq_length,
        )

    return bool(patched_classes)
