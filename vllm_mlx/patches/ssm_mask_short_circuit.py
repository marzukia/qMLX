# SPDX-License-Identifier: Apache-2.0
"""Conditional SSM-mask short-circuit for hybrid (GDN/Mamba) models.

Background
----------
Upstream ``mlx_lm.models.base.create_ssm_mask(h, cache)`` returns
``cache.make_mask(T)`` whenever the cache exposes ``make_mask``. For the
native BatchGenerator caches (``BatchMambaCache`` / ``ArraysCache`` family
with ``left_padding = [0]``) that mask is an all-True ``(1, T)`` array:
numerically a no-op, but it feeds ``mx.fast.gated_delta_update`` (ROCm lib
kernel) through the MASKED path, measured 9.6-15.1x slower than the
unmasked path on gfx1201 (0.16-0.19 vs 0.011-0.019 ms/tok, T-flat
512->4095). On a 4K prefill that was ~17.8s of the 24.2s serve wall (24
GDN layers x 2 x 2048 steps) and a real T=1 decode tax as well.

What this module patches
------------------------
``create_ssm_mask`` -- when the cache carries no real left padding
(``left_padding`` absent or all offsets <= 0) and no length truncation
(``lengths`` absent or all >= T), return ``None`` so the kernel stays on
the unmasked fast path. Any real padding (offset > 0) or truncation
(length < T) keeps the original mask unchanged, so padded batches are
bit-identical to the stock behavior.

Verified (gfx1201, Qwen3.5-9B hybrid, 2026-11-19): FNV1A token
streams bit-identical on the padded arm; 4K prefill 26.4s -> 2.17s,
steady 24.4s -> 1.2s (inside the direct class); T=1 decode 29.6 -> 37.8
tps.

Install
-------
Idempotent, thread-safe, no-op when ``mlx_lm`` is unavailable. Model files
``from .base import create_ssm_mask`` (name binding at import time), so the
install also rebinds the function in every already-loaded
``mlx_lm.models.*`` namespace that holds the original.
"""

import logging
import sys
import threading

logger = logging.getLogger(__name__)

_INSTALLED = False
_LOCK = threading.Lock()
_orig: object = None


def _short_circuit_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        T = h.shape[1]
        lp = getattr(cache, "left_padding", None)
        ln = getattr(cache, "lengths", None)
        # Conditional skip: with no real left padding (all offsets <= 0)
        # and no length truncation (all lengths >= T), the mask would be
        # all-True (a numeric no-op); returning None keeps the kernel on
        # the fast path. Any real padding or truncation keeps the base
        # mask unchanged.
        if (lp is None or lp.max().item() <= 0) and (
            ln is None or ln.min().item() >= T
        ):
            return None
        return cache.make_mask(T)
    return None


def install_ssm_mask_short_circuit() -> None:
    """Install the conditional skip. Idempotent and thread-safe."""
    global _INSTALLED, _orig

    with _LOCK:
        if _INSTALLED:
            return

        # Import lazily so this module is cheap to import even when mlx_lm
        # is unavailable (e.g. tests that patch import paths).
        try:
            from mlx_lm.models import base as _base
        except ImportError:  # pragma: no cover - mlx_lm always present in our env
            logger.debug(
                "[ssm_mask_short_circuit] mlx_lm not importable; skipping install"
            )
            return

        # Stash the upstream original the FIRST time we install so a later
        # install after a module reload cannot self-reference the patched
        # callable.
        if _orig is None:
            _orig = _base.create_ssm_mask

        _base.create_ssm_mask = _short_circuit_ssm_mask

        # Rebind every already-loaded model module that from-imported the
        # original (all mlx_lm hybrid model files bind the name at import
        # time; a base-module patch alone would not reach them).
        rebound = 0
        for mod in list(sys.modules.values()):
            name = getattr(mod, "__name__", "")
            if (
                name.startswith("mlx_lm.models.")
                and getattr(mod, "create_ssm_mask", None) is _orig
            ):
                mod.create_ssm_mask = _short_circuit_ssm_mask
                rebound += 1

        _INSTALLED = True
        logger.debug(
            "[ssm_mask_short_circuit] installed (rebound %s model modules)",
            rebound,
        )
