# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-lm's ``ArraysCache`` to carry a ``rollback_state`` slot.

mlx-lm PR #990 adds a ``rollback_state: Optional[tuple] = None`` class
attribute to ``mlx_lm.models.cache.ArraysCache``. It is set by the
GatedDeltaNet layer's ``_process_chunk`` split (saves the
``(conv_state, ssm_state)`` snapshot at position ``n_confirmed``) and
read by the MTP generator's ``_rollback_draft`` (restores the snapshot
on draft rejection). Both writers and readers run under
``mx.stream(generation_stream)`` so the lock-free attribute access is
safe.

Until upstream merges, our installed ``mlx_lm 0.31.3`` does not have
the attribute. Setting it on a per-instance basis from the patched
model's ``_process_chunk`` would work, but Python's attribute lookup
falls back to the class only after the instance miss, so the FIRST
write succeeds — but the ``hasattr(cache, "rollback_state")`` guard in
the generator's ``_clear_rollback`` runs against the CLASS first and
would return ``False`` on a fresh cache, skipping the clear. That's
fine in isolation (nothing to clear) but the same guard is used to
gate ``rollback_state is not None`` checks; without the class slot we
have to fall back to ``getattr(c, "rollback_state", None)`` everywhere
which is fragile.

Patching the class once at import time is the simple fix. The patch is:

* Idempotent — calling :func:`patch_arrays_cache_rollback_state` twice
  is a no-op.
* Reversible only via process restart — the patch is intentionally
  one-way. There is no test path that needs to un-patch (mlx-lm's
  ``ArraysCache`` is a behaviorally-pure attribute slot; adding it
  doesn't change anything for callers that don't touch it).
* Safe under future mlx-lm versions that add the slot themselves —
  the guard checks ``"rollback_state" in cls.__dict__`` before
  patching, so once upstream lands the change this becomes a no-op.

The patch is applied automatically the first time
:func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step` is
imported (the import in the generator module forces the side-effect).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# Module-level guard so concurrent threads importing the generator
# don't race on the class attribute install. Without the lock, two
# threads could both see ``"rollback_state" not in cls.__dict__`` and
# both setattr — harmless for an attribute set to ``None`` (the writes
# are identical) but conceptually racy. The lock keeps the install
# atomic.
_install_lock = threading.Lock()
_PATCHED = False
_GATED_DELTA_PATCHED = False
_ORIG_GATED_DELTA_CALL = None


def patch_arrays_cache_rollback_state() -> bool:
    """Install ``rollback_state = None`` on ``mlx_lm.models.cache.ArraysCache``.

    Returns ``True`` if the patch was applied, ``False`` if the slot
    was already present (either from a previous call or from a future
    mlx-lm version that lands the change upstream).

    Raises:
        ImportError: If ``mlx_lm.models.cache`` cannot be imported.
            The MTP path is fundamentally unusable without mlx-lm so
            we let the import error propagate rather than silently
            falling back.
    """
    global _PATCHED

    with _install_lock:
        if _PATCHED:
            return False

        # Defer the import so a static analyzer can't trip on the
        # mlx_lm dependency before the package is installed (the
        # generator module itself imports mlx_lm at the top, so by the
        # time this patch fires, the import must already work — but
        # we still keep it lazy for symmetry with the rest of the MTP
        # package).
        from mlx_lm.models.cache import ArraysCache

        # ``cls.__dict__`` check (not ``hasattr``) so a future mlx-lm
        # that ships the slot wins over our patch — we don't want to
        # shadow an upstream rename or type change.
        if "rollback_state" in ArraysCache.__dict__:
            _PATCHED = True
            logger.debug(
                "[mtp.cache_patch] ArraysCache.rollback_state already present "
                "(upstream version or prior patch); skipping install."
            )
            return False

        # The class attribute default is ``None``; instance writes
        # shadow it transparently. This mirrors the upstream PR #990
        # patch verbatim (``ArraysCache`` is a ``_BaseCache`` subclass
        # built via ``__new__``, so class-level defaults are the right
        # shape — there is no ``__init__`` that would otherwise
        # initialize the slot).
        ArraysCache.rollback_state = None  # type: ignore[attr-defined]
        _PATCHED = True
        logger.info(
            "[mtp.cache_patch] Installed rollback_state slot on "
            "ArraysCache (vendored from mlx-lm PR #990)."
        )
        return True


def patch_gated_delta_net_for_mtp() -> bool:
    """Wrap ``GatedDeltaNet.__call__`` with TAPE-based chunk-split for K>=2 rollback.

    TAPE ROLLBACK (PR-C+): Instead of storing a single snapshot tuple at
    the confirmed boundary, we now record a TAPE of snapshots — one per
    position from 1 to n_confirmed. This unlocks K>=2 on SSM-hybrid
    targets by allowing rollback to ANY position in the draft chain.

    TAPE FORMAT: ``rollback_state`` is a list of tuples:
    ``[(conv_snap_1, ssm_snap_1), (conv_snap_2, ssm_snap_2), ...]``
    where index i corresponds to the state AFTER processing i+1 tokens.
    Rolling back to position N means restoring ``rollback_state[N-1]``.

    Without this patch the verify forward advances the SSM by 2 steps
    and there is no way to roll back to position 1 on rejection — the
    LOSSLESS contract breaks on the linear-attention layers (only;
    full-attention's ``KVCache.trim(1)`` already handles its rollback).
    Output diverges from the non-spec-decode baseline within ~10
    tokens at 90% accept rate.

    The patch:

    * Is idempotent — calling twice is a no-op.
    * Is transparent — when ``cache.n_confirmed_for_mtp`` is 0 (the
      class default), the wrapped call falls through to the original
      ``__call__`` unchanged. Production non-MTP code paths are
      unaffected.
    * Reads the chunk boundary from ``cache.n_confirmed_for_mtp``,
      which the MTP-wrapped ``TextModel.__call__`` sets before each
      ``layer.linear_attn`` invocation. Threading via a cache attr
      avoids changing the layer's call signature (and so avoids
      touching ``DecoderLayer.__call__`` / ``Qwen3_5TextModel.__call__``
      upstream).

    Returns ``True`` when the patch was applied (or already in place),
    ``False`` if mlx-lm cannot be imported.
    """
    global _GATED_DELTA_PATCHED, _ORIG_GATED_DELTA_CALL

    with _install_lock:
        if _GATED_DELTA_PATCHED:
            return True

        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.models.cache import ArraysCache
            from mlx_lm.models.gated_delta import gated_delta_update
            from mlx_lm.models.qwen3_5 import GatedDeltaNet
        except ImportError:  # pragma: no cover — mlx_lm always available
            logger.warning(
                "[mtp.cache_patch] Could not import GatedDeltaNet; "
                "skipping rollback-state install."
            )
            return False

        # Add a class-default ``n_confirmed_for_mtp`` slot to
        # ArraysCache so the layer can read it without an
        # ``AttributeError`` on caches the wrapper hasn't tagged.
        if "n_confirmed_for_mtp" not in ArraysCache.__dict__:
            ArraysCache.n_confirmed_for_mtp = 0  # type: ignore[attr-defined]

        _ORIG_GATED_DELTA_CALL = GatedDeltaNet.__call__

        def _patched_call(self, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            n_conf = 0
            if cache is not None:
                n_conf = int(getattr(cache, "n_confirmed_for_mtp", 0) or 0)

            # Fast path — no MTP boundary signaled, or chunk has 1
            # token, or boundary is outside the range, or NO cache at
            # all. Defer to the original implementation (byte-equal
            # behavior). rollback_state is NOT touched on this path.
            if cache is None or n_conf <= 0 or n_conf >= S or S < 2:
                return _ORIG_GATED_DELTA_CALL(self, inputs, mask=mask, cache=cache)

            # --- Chunk-split path (n_confirmed in (0, S)) ---
            if self.sharding_group is not None:
                # The verify cycle only runs single-device; bail back
                # to the unsplit path under tensor parallel.
                return _ORIG_GATED_DELTA_CALL(self, inputs, mask=mask, cache=cache)

            # Clear rollback_state before chunk-split (see round-7
            # ordering fix in the original patch).
            cache.rollback_state = None

            # Steps 1-3: projections + conv prefix — identical to the
            # original call. Build all derived tensors once.
            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)

            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )

            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            n_keep = self.conv_kernel_size - 1

            # Conv state AT END (after processing all S tokens):
            # last n_keep entries of conv_input.
            conv_post = mx.contiguous(conv_input[:, -n_keep:, :])
            cache[0] = conv_post

            conv_out = nn.silu(self.conv1d(conv_input))

            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                )
            ]

            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            # TAPE ROLLBACK: Build snapshots at EVERY position from 1
            # to n_confirmed. This allows rolling back to ANY position
            # in the draft chain, not just position 1.
            tape = []
            out1_list = []
            current_state = state

            for idx in range(n_conf):
                q_pos = q[:, idx : idx + 1]
                k_pos = k[:, idx : idx + 1]
                v_pos = v[:, idx : idx + 1]
                a_pos = a[:, idx : idx + 1]
                b_pos = b[:, idx : idx + 1]
                mask_pos = mask[:, idx : idx + 1] if mask is not None else None

                # Conv state at this position: last n_keep entries of
                # conv_input[:, : idx+1 + n_keep].
                conv_snap_pos = mx.contiguous(
                    conv_input[:, idx + 1 : idx + 1 + n_keep, :]
                )

                out_pos, current_state = gated_delta_update(
                    q_pos,
                    k_pos,
                    v_pos,
                    a_pos,
                    b_pos,
                    self.A_log,
                    self.dt_bias,
                    current_state,
                    mask_pos,
                    use_kernel=not self.training,
                )
                out1_list.append(out_pos)

                # Record tape entry for this position
                tape.append((conv_snap_pos, current_state))

                # Also record via tape_rollback module if available
                try:
                    from .tape_rollback import get_tape_recorder
                    recorder = get_tape_recorder()
                    if recorder.is_recording:
                        # Find layer index from cache
                        # This is a best-effort lookup; layer_idx is needed for tape
                        recorder.record_entry(
                            layer_idx=0,  # Will be set by caller
                            conv_state=conv_snap_pos,
                            ssm_state=current_state
                        )
                except (ImportError, RuntimeError):
                    pass  # tape_rollback not initialized or not recording

            out1 = mx.concatenate(out1_list, axis=1) if out1_list else mx.zeros(
                (B, 0, self.num_v_heads, self.head_v_dim), dtype=inputs.dtype
            )

            state_at_boundary = tape[-1][1] if tape else state

            # Chunk 2: [n_conf:S] — process remaining tokens in batch
            if n_conf < S:
                q2 = q[:, n_conf:]
                k2 = k[:, n_conf:]
                v2 = v[:, n_conf:]
                a2 = a[:, n_conf:]
                b2 = b[:, n_conf:]
                mask2 = mask[:, n_conf:] if mask is not None else None
                out2, state_final = gated_delta_update(
                    q2,
                    k2,
                    v2,
                    a2,
                    b2,
                    self.A_log,
                    self.dt_bias,
                    state_at_boundary,
                    mask2,
                    use_kernel=not self.training,
                )
            else:
                out2 = mx.zeros((B, 0, self.num_v_heads, self.head_v_dim), dtype=inputs.dtype)
                state_final = state_at_boundary

            # TAPE FORMAT: list of (conv_snap, ssm_snap) tuples, one per
            # position from 1 to n_conf. Rolling back to position N means
            # restoring tape[N-1].
            cache.rollback_state = tape

            out = mx.concatenate([out1, out2], axis=1)
            cache[1] = state_final
            cache.advance(S)

            out = self.norm(out, z)
            out = self.out_proj(out.reshape(B, S, -1))
            return out

        GatedDeltaNet.__call__ = _patched_call  # type: ignore[assignment]
        _GATED_DELTA_PATCHED = True
        logger.info(
            "[mtp.cache_patch] Installed GatedDeltaNet TAPE rollback "
            "(snapshots at all positions 1..n_confirmed for K>=2 support)."
        )
        return True


def _is_patched_for_tests() -> bool:
    """Test-only — inspect the install flag."""
    return _PATCHED


def _unpatch_for_tests() -> None:
    """Test-only — clear the install flag and remove the class attr.

    Allows tests to verify the install side-effect by toggling the
    install state. Never called from production.
    """
    global _PATCHED, _GATED_DELTA_PATCHED, _ORIG_GATED_DELTA_CALL

    with _install_lock:
        try:
            from mlx_lm.models.cache import ArraysCache

            if "rollback_state" in ArraysCache.__dict__:
                delattr(ArraysCache, "rollback_state")
            if "n_confirmed_for_mtp" in ArraysCache.__dict__:
                delattr(ArraysCache, "n_confirmed_for_mtp")
        except ImportError:
            pass
        if _GATED_DELTA_PATCHED and _ORIG_GATED_DELTA_CALL is not None:
            try:
                from mlx_lm.models.qwen3_5 import GatedDeltaNet

                GatedDeltaNet.__call__ = _ORIG_GATED_DELTA_CALL  # type: ignore[assignment]
            except ImportError:
                pass
        _PATCHED = False
        _GATED_DELTA_PATCHED = False
        _ORIG_GATED_DELTA_CALL = None
