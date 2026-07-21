# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP injection for Qwen3.5 / Qwen3.6 models (vendor PR #990).

mlx-lm 0.31.3 does not yet ship PR #990, so its
``mlx_lm.models.qwen3_5.TextModel.__call__`` does not accept
``return_hidden`` or ``n_confirmed`` and the class has no
``mtp_forward`` / ``make_mtp_cache`` methods. Without those four
surfaces, :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`
can't drive the model.

This module mirrors the pattern from
:mod:`vllm_mlx.patches.qwen3_next_mtp` (the Qwen3-Next runtime injection):

1. Construct the MTP module that PR #990 adds to ``TextModel`` —
   delegated to :func:`vllm_mlx.spec_decode.mtp.head.build_mtp_module`.
2. Quantize the MTP module to match the base model's quantization (so
   the weight tensors land in the right shape for ``load_weights``).
3. Load the MTP weights from a separate ``mtp_sidecar`` checkpoint —
   ``mlx-community/Qwen3.5-9B-MTP-4bit`` ships the head as a 131 MB
   standalone safetensors file with top-level keys (``fc.*``,
   ``layers.0.*``, ``norm.weight``, ``pre_fc_norm_{hidden,embedding}.weight``).
4. Monkey-patch the ``TextModel`` instance's ``__class__`` to a
   subclass that adds the four MTP surfaces (``__call__`` with
   ``return_hidden``/``n_confirmed``, ``mtp_forward``,
   ``make_mtp_cache``).

Coverage scope
--------------

In-scope: the dense ``TextModel`` (``mlx_lm.models.qwen3_5.TextModel``),
its MoE subclass (``mlx_lm.models.qwen3_5_moe.Model``), and the VLM
wrapper (``mlx_lm.models.qwen3_5.Model``) where the text model is
nested under ``model.language_model``. The patch always targets the
inner ``TextModel`` — never the outer VLM wrapper (whose ``__call__``
just delegates).

``n_confirmed`` rollback: implemented as of this PR. ``__call__``
accepts ``n_confirmed`` and threads it through to each
``ArraysCache`` via ``n_confirmed_for_mtp`` before the forward, so
the patched ``GatedDeltaNet.__call__`` (installed by
``patch_gated_delta_net_for_mtp``) can snapshot ``(conv_state,
ssm_state)`` AT the confirmed-token boundary. On draft rejection the
generator's ``_rollback_draft`` restores the snapshot per cache
instance. Lossless contract confirmed byte-equal × 3 profiles on
mlx-community/Qwen3.5-9B-4bit + mlx-community/Qwen3.5-9B-MTP-4bit
(see ``tests/test_mtp_real_weights.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_inner_text_model(model: Any) -> Any:
    """Return the ``TextModel`` instance the patch must monkey-patch.

    For mlx-lm 0.31.3's Qwen3.5 architecture, ``mlx_lm.load(...)``
    returns the VLM-style ``Model`` wrapper whose ``language_model``
    field is the actual ``TextModel`` (carrying ``embed_tokens``,
    ``lm_head``, the ``model.layers`` backbone, and ``args``). The
    wrapper itself only has ``args = ModelArgs(model_type,
    text_config)`` and a delegating ``__call__`` — patching it would
    leave ``self.model.embed_tokens`` undefined for the injected
    ``mtp_forward``.

    Three shapes are accepted:

    * The outer VLM-style ``Model`` with ``model.language_model`` (real
      runtime path).
    * The inner ``TextModel`` itself (the test path constructs this
      directly to avoid the heavy VLM init).
    * A custom shell that exposes ``args`` + ``model`` and where
      ``args`` has either ``hidden_size`` (the inner-TextModel-like
      shape) or ``mtp_num_hidden_layers`` (the explicit-test shape).
      Used by ``test_inject_mtp_support_rejects_*`` paths.
    """
    # Case 1: VLM wrapper — text model lives under ``language_model``.
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "args") and hasattr(lm, "model"):
        return lm

    # Case 2: Already the inner TextModel (or a test shell). The inner
    # TextModel exposes both ``model`` (the backbone) and ``args``.
    if hasattr(model, "model") and hasattr(model, "args"):
        return model

    return None


def _detect_base_quantization(inner: Any) -> dict | None:
    """Detect the quantization params used by the base model.

    Walks the inner ``TextModel`` looking for a ``QuantizedLinear``
    instance and reads its ``bits`` / ``group_size``. The MTP module
    must be quantized with the same params so its weight shapes match
    the sidecar's safetensors layout (4-bit / group_size 64 / affine
    for ``mlx-community/Qwen3.5-9B-MTP-4bit``).

    Returns ``None`` for FP base models — the caller skips quantize
    in that case.

    NOTE: only ``bits`` + ``group_size`` are returned. ``nn.quantize``
    in the mlx-lm versions we target does not accept a ``mode`` arg —
    it always applies the affine mode. The mlx-community sidecars
    similarly assume affine. Returning ``mode`` would be dead data
    that callers cannot pass through, so it's dropped. If/when
    ``nn.quantize`` grows mode support, extend the dict here and
    pipe it through at the inject call-site.
    """
    try:
        from mlx.nn import QuantizedEmbedding, QuantizedLinear
    except ImportError:  # pragma: no cover — mlx.nn always available
        return None

    backbone = getattr(inner, "model", None)
    if backbone is None:
        return None

    # Try a full-attention layer's q_proj first (always present + quantized).
    for layer in getattr(backbone, "layers", []):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            qp = layer.self_attn.q_proj
            if isinstance(qp, QuantizedLinear):
                return {
                    "bits": int(qp.bits),
                    "group_size": int(qp.group_size),
                }

    # Fall back: embed_tokens (QuantizedEmbedding has bits/group_size too).
    embed = getattr(backbone, "embed_tokens", None)
    if isinstance(embed, QuantizedEmbedding):
        return {
            "bits": int(embed.bits),
            "group_size": int(embed.group_size),
        }

    return None


def _resolve_sidecar_file(mtp_sidecar: str | Path) -> Path | None:
    """Resolve a sidecar reference to a concrete safetensors file path.

    Accepts:

    * An absolute / relative path to a directory containing a
      ``model.safetensors`` or ``model-mtp.safetensors`` file
      (operators with a pre-downloaded HF snapshot).
    * An absolute / relative path to a ``*.safetensors`` file
      directly (operators with a hand-assembled sidecar; the
      filename does NOT have to be one of the two well-known
      names).
    * An HF Hub repo name like ``mlx-community/Qwen3.5-9B-MTP-4bit``
      (downloaded via ``snapshot_download`` to the HF cache, then
      probed for ``model.safetensors`` / ``model-mtp.safetensors``).

    Returns ``None`` if the reference cannot be resolved — caller
    treats this as a soft failure and logs.
    """
    if mtp_sidecar is None:
        return None

    path = Path(mtp_sidecar)
    if path.is_file():
        # Explicit file path — use it verbatim. Supports operator
        # workflows where the sidecar lives at a custom filename
        # (``mtp-q4-g64.safetensors``, ``qwen3_5_mtp_head.safetensors``,
        # …). Skipping the well-known-name probe avoids the silent
        # "file is at a non-default name → fall back to None" trap
        # codex flagged on PR #954 review.
        return path
    if path.is_dir():
        return _find_mtp_weights_file(path)

    # Treat as HF repo id.
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar))
        return _find_mtp_weights_file(Path(local))
    except Exception as exc:  # pragma: no cover — network failure path
        logger.warning(
            "[mtp.inject] could not resolve sidecar %r: %s",
            mtp_sidecar,
            exc,
        )
        return None


def _find_mtp_weights_file(sidecar_dir: Path) -> Path | None:
    """Pick the safetensors file inside ``sidecar_dir`` that holds the MTP head.

    The mlx-community ``Qwen3.5-9B-MTP-4bit`` repo ships
    ``model.safetensors`` (single shard, 131 MB, 31 keys, no ``mtp.``
    prefix). Other vendors may ship ``model-mtp.safetensors`` (the
    Qwen3-Next convention used by ``add_mtp_weights.py``). Try both.
    """
    candidates = (
        sidecar_dir / "model-mtp.safetensors",
        sidecar_dir / "model.safetensors",
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def _fix_mtp_norm_double_shift(model: Any) -> None:
    """Fix norm weights that were incorrectly double-shifted by mlx-lm.

    stock mlx-lm's ``TextModel.sanitize`` adds +1.0 to ALL norm weights
    when MTP keys are present (``has_mtp_weights=True``). But models like
    oQ4-mtp have norms already in MLX convention (mean > 0.5). The
    double-shift corrupts normalization (mean goes from ~2 to ~3),
    producing garbage output.

    This function checks each norm weight and removes the extra +1.0
    if the weight is already in MLX convention (mean > 0.5 after the
    shift, meaning the original mean was > -0.5 which is always true
    for RMSNorm weights).

    The fix is safe: RMSNorm weights in MLX convention have mean ~= 1-3.
    If they were incorrectly shifted, subtracting 1.0 restores them.
    If they were correctly shifted (raw-HF, mean was ~0), subtracting
    1.0 would make them negative — which we guard against.
    """
    import mlx.core as mx

    _NORM_SUFFIXES = (
        "input_layernorm",
        "post_attention_layernorm",
        "model.norm",
        "q_norm",
        "k_norm",
    )

    fixed = 0
    for name, mod in model.named_modules():
        if not hasattr(mod, "weight") or mod.weight is None:
            continue
        if mod.weight.ndim != 1:
            continue
        if not any(name.endswith(s) for s in _NORM_SUFFIXES):
            continue
        mean_val = float(mx.mean(mod.weight.astype(mx.float32)).item())
        # RMSNorm weights in MLX convention: mean ~= 1.0
        # RMSNorm weights in raw-HF convention: mean ~= 0.0
        # stock sanitize adds +1.0 when MTP keys present:
        #   MLX convention + 1.0 = mean ~2.0 (incorrect, double-shift)
        #   raw-HF + 1.0 = mean ~1.0 (correct)
        # Threshold: mean > 1.5 indicates double-shift.
        if mean_val > 1.5:
            mod.weight = mod.weight - 1.0
            fixed += 1

    if fixed > 0:
        logger.info(
            "[mtp.inject] Fixed %d double-shifted norm weights "
            "(subtracted 1.0 from norms with mean > 1.5)",
            fixed,
        )


def _load_mtp_weights_from_repo(model_repo: str) -> dict[str, Any] | None:
    """Try to load MTP weights from a model's own HF repo safetensors."""
    import glob as _glob
    import os

    import mlx.core as mx
    from huggingface_hub import snapshot_download

    try:
        local = snapshot_download(model_repo, allow_patterns=["*.safetensors"])
    except Exception as exc:
        logger.warning(
            "[mtp.inject] Could not download model repo %r: %s",
            model_repo,
            exc,
        )
        return None

    mtp_weights: dict[str, Any] = {}
    for sf_path in sorted(_glob.glob(os.path.join(local, "*.safetensors"))):
        raw = mx.load(sf_path)
        for k, v in raw.items():
            # Match keys like ``language_model.mtp.layers.0.*``
            # or ``mtp.layers.0.*`` or ``model.mtp.*``
            if ".mtp." in k or k.startswith("mtp."):
                # Strip ``language_model.`` prefix if present
                clean = k.removeprefix("language_model.")
                # Strip ``mtp.`` prefix if present (some converters)
                clean = clean.removeprefix("mtp.")
                mtp_weights[clean] = v

    if not mtp_weights:
        return None
    return mtp_weights


def _infer_mtp_config(mtp_weights: dict[str, Any]) -> dict[str, Any]:
    """Infer MTP head architecture from weight shapes.

    Models like oQ4-mtp may have MTP heads with different dimensions
    than the backbone. When the model's config.json lacks an
    ``mtp_config`` section, we infer the MTP head's architecture from
    the actual weight tensor shapes.

    Returns a dict of overrides to merge into the backbone's args.
    """
    overrides: dict[str, Any] = {}

    # head_dim: from k_norm.weight shape (one value per head)
    k_norm = mtp_weights.get("layers.0.self_attn.k_norm.weight")
    if k_norm is not None:
        overrides["head_dim"] = int(k_norm.shape[0])

    head_dim = overrides.get("head_dim")

    # num_attention_heads: infer from o_proj (which always takes
    # num_attention_heads * head_dim as input, regardless of Q gating).
    # q_proj may have a *2 factor for gated attention, making it
    # unreliable for inferring num_heads.
    o_proj = mtp_weights.get("layers.0.self_attn.o_proj.weight")
    o_scales = mtp_weights.get("layers.0.self_attn.o_proj.scales")
    if (
        o_proj is not None
        and o_scales is not None
        and head_dim is not None
        and head_dim > 0
        and len(o_proj.shape) >= 2
        and len(o_scales.shape) >= 2
    ):
        o_k_packed = o_proj.shape[-1]
        o_n_groups = o_scales.shape[-1]
        for o_gs in [64, 128, 32]:
            o_k = o_n_groups * o_gs
            if o_k > 0 and (o_k_packed * 32) % o_k == 0:
                o_bits = (o_k_packed * 32) // o_k
                if o_bits in [2, 3, 4, 5, 6, 8]:
                    overrides["num_attention_heads"] = o_k // head_dim
                    break

    # num_key_value_heads: from k_proj weight shape
    k_proj = mtp_weights.get("layers.0.self_attn.k_proj.weight")
    if k_proj is not None and head_dim is not None and head_dim > 0:
        overrides["num_key_value_heads"] = int(k_proj.shape[0]) // head_dim

    # intermediate_size: from shared_expert gate_proj or dense MLP gate_proj
    shared_gate = mtp_weights.get("layers.0.mlp.shared_expert.gate_proj.weight")
    if shared_gate is not None:
        overrides["intermediate_size"] = int(shared_gate.shape[0])
    else:
        gate = mtp_weights.get("layers.0.mlp.gate_proj.weight")
        if gate is not None:
            overrides["intermediate_size"] = int(gate.shape[0])

    # num_experts: from switch_mlp gate_proj weight shape
    switch_gate = mtp_weights.get("layers.0.mlp.switch_mlp.gate_proj.weight")
    if switch_gate is not None and len(switch_gate.shape) >= 3:
        overrides["num_experts"] = int(switch_gate.shape[0])

    # Detect gated vs standard attention: check if q_proj has the *2
    # factor vs the (now-correct) num_attention_heads from o_proj.
    q_proj = mtp_weights.get("layers.0.self_attn.q_proj.weight")
    if q_proj is not None and head_dim is not None and head_dim > 0:
        q_out = int(q_proj.shape[0])
        num_heads = overrides.get("num_attention_heads", 32)
        expected_gated = num_heads * head_dim * 2
        overrides["_use_gated_attention"] = q_out == expected_gated

    if overrides:
        logger.info(
            "[mtp.inject] Inferred MTP config from weight shapes: %s",
            overrides,
        )
    return overrides


def _infer_mtp_quantization(
    mtp_weights: dict[str, Any],
) -> dict[str, Any] | None:
    """Infer quantization params from the first quantized MTP weight.

    Reads bits and group_size from the first weight tensor that has
    associated ``.scales`` and ``.biases`` tensors. Returns None if
    no quantized weights are found (model is bf16/fp16).
    """
    # Find the first weight that has scales + biases
    scale_keys = [k for k in mtp_weights if k.endswith(".scales")]
    for scale_key in sorted(scale_keys):
        base = scale_key.removesuffix(".scales")
        weight_key = base + ".weight"
        biases_key = base + ".biases"
        if weight_key in mtp_weights and biases_key in mtp_weights:
            w = mtp_weights[weight_key]
            s = mtp_weights[scale_key]
            if len(w.shape) >= 2 and len(s.shape) >= 2:
                k_packed = w.shape[1] if len(w.shape) == 2 else w.shape[-1]
                n_groups = s.shape[-1] if len(s.shape) >= 2 else s.shape[0]
                for gs in [64, 128, 32]:
                    k = n_groups * gs
                    if k > 0 and (k_packed * 32) % k == 0:
                        bits = (k_packed * 32) // k
                        if bits in [2, 3, 4, 5, 6, 8]:
                            return {"bits": bits, "group_size": gs}
                return {"bits": 4, "group_size": 64}
    return None


def _build_per_layer_quant_map(
    mtp_weights: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a per-layer quantization map from MTP weight shapes.

    Returns a dict mapping module path (e.g. ``layers.0.self_attn.q_proj``)
    to its quantization config (``bits``, ``group_size``). Layers without
    scales/biases are omitted (they remain bf16/fp16).
    """
    quant_map: dict[str, dict[str, Any]] = {}
    scale_keys = [k for k in mtp_weights if k.endswith(".scales")]

    for scale_key in sorted(scale_keys):
        base = scale_key.removesuffix(".scales")
        weight_key = base + ".weight"
        biases_key = base + ".biases"
        if weight_key not in mtp_weights or biases_key not in mtp_weights:
            continue

        w = mtp_weights[weight_key]
        s = mtp_weights[scale_key]
        if len(w.shape) < 2 or len(s.shape) < 2:
            continue

        k_packed = w.shape[1] if len(w.shape) == 2 else w.shape[-1]
        n_groups = s.shape[-1] if len(s.shape) >= 2 else s.shape[0]

        bits = None
        group_size = None
        for gs in [64, 128, 32]:
            k = n_groups * gs
            if k > 0 and (k_packed * 32) % k == 0:
                candidate_bits = (k_packed * 32) // k
                if candidate_bits in [2, 3, 4, 5, 6, 8]:
                    bits = candidate_bits
                    group_size = gs
                    break

        if bits is not None and group_size is not None:
            # Map weight key to module path:
            # ``layers.0.self_attn.q_proj.weight`` -> ``layers.0.self_attn.q_proj``
            module_path = base
            quant_map[module_path] = {"bits": bits, "group_size": group_size}

    return quant_map


def inject_mtp_support(
    model: Any,
    mtp_sidecar: str | Path | None = None,
    *,
    allow_random_init: bool = False,
    model_repo: str | None = None,
) -> bool:
    """Inject MTP support into a loaded Qwen3.5 / Qwen3.6 model.

    Args:
        model: A model loaded via ``mlx_lm.load()``. Either the VLM
            wrapper ``Model`` (with ``model.language_model``) or the
            inner ``TextModel`` directly (tests pass this shape).
        mtp_sidecar: Optional reference to a separate checkpoint
            holding the MTP head's safetensors. Accepts an HF Hub
            repo id (``mlx-community/Qwen3.5-9B-MTP-4bit``), a local
            directory path, or a direct path to a ``.safetensors``
            file.
        allow_random_init: When ``True``, permit ``mtp_sidecar=None``
            and ship the MTP head with its RANDOM INIT weights (the
            patched ``mtp_forward`` produces useless drafts, accept
            rate ~0%). Test-only. Codex flagged on PR #954 that
            allowing this by default lets production callers silently
            enable a useless/slow draft model, so the default is
            ``False`` — a missing sidecar in production now returns
            ``False`` from this function and the model is left
            unmodified. The bench, server boot, and the qmlx
            spec_decode pipeline MUST pass a sidecar.
        model_repo: Optional HF repo id (e.g.
            ``mlx-community/Qwen3.5-122B-A10B-oQ4-mtp``). When
            ``mtp_sidecar`` is ``None`` and ``model_repo`` is
            provided, the inject code tries to load MTP weights from
            the model's own safetensors (for models like oQ4-mtp
            where MTP weights are embedded in the main checkpoint
            but stripped by ``mlx_lm.load()``).

    Returns:
        ``True`` when the patch landed and the model now exposes
        ``mtp_forward``, ``make_mtp_cache``, ``return_hidden``, and
        ``n_confirmed`` — the four contract surfaces
        :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`
        depends on. ``False`` when the model is not Qwen3.5 / 3.6,
        the config lacks ``mtp_num_hidden_layers``, the sidecar
        cannot be resolved, or ``mtp_sidecar`` is ``None`` and
        ``allow_random_init`` is ``False``.

    Notes:
        This function is NEW in this PR (Qwen3.5 native MTP). It is
        NOT the legacy ``vllm_mlx.patches.qwen3_next_mtp.inject_mtp_support``
        used by the scheduler (different signature, different model
        family, different load path). The only production caller of
        this function is ``bench/bench_spec_decode_mtp.py`` (which
        already passes ``mtp_sidecar``). There are no pre-existing
        bare ``inject_mtp_support(model)`` call-sites to break with
        the new ``allow_random_init=False`` default.

        ``n_confirmed`` rollback is implemented as of this PR: it
        threads through to each ``ArraysCache`` via
        ``n_confirmed_for_mtp`` before forward, so the patched
        ``GatedDeltaNet.__call__`` (installed by
        ``patch_gated_delta_net_for_mtp``) can snapshot
        ``(conv_state, ssm_state)`` AT the confirmed-token boundary.
    """
    import mlx.core as mx
    import mlx.nn as nn

    # NOTE: the global ``ArraysCache`` rollback_state class-default and
    # the ``GatedDeltaNet.__call__`` chunk-split patches are deferred
    # until AFTER every can-fail validation completes (see ``# --- Step
    # 4`` below). Codex flagged on PR #954 that installing these
    # monkey-patches up-front meant a failed sidecar load left
    # process-global behavior mutated even though inject_mtp_support
    # returned False. The patches are now strictly post-validation.

    inner = _resolve_inner_text_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject] model %s has neither model.language_model nor "
            "(model + args); skipping MTP injection.",
            type(model).__name__,
        )
        return False

    args = inner.args

    # 1. Resolve num_mtp_layers. Prefer the dataclass attr (which
    # tests set via object.__setattr__); fall back to the outer
    # wrapper's text_config dict (the real runtime path — mlx-lm
    # 0.31.3's TextModelArgs lacks ``mtp_num_hidden_layers`` so the
    # field gets dropped during ``BaseModelArgs.from_dict``).
    num_mtp_layers = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
    if num_mtp_layers < 1:
        outer_args = getattr(model, "args", None)
        text_config = getattr(outer_args, "text_config", None) or {}
        if isinstance(text_config, dict):
            num_mtp_layers = int(text_config.get("mtp_num_hidden_layers", 0) or 0)
        if num_mtp_layers >= 1:
            # Surface it on the dataclass so downstream code (incl.
            # validate_mtp_support, accept_counter labels) can read it
            # off ``args.mtp_num_hidden_layers`` uniformly.
            try:
                object.__setattr__(args, "mtp_num_hidden_layers", num_mtp_layers)
            except (TypeError, AttributeError):  # pragma: no cover — frozen
                pass

    if num_mtp_layers < 1:
        logger.info(
            "[mtp.inject] config has no mtp_num_hidden_layers; skipping MTP injection."
        )
        return False

    # --- Step 0: Fix norm weight double-shift for MTP models ---
    # stock mlx-lm's sanitize adds +1.0 to ALL norm weights when MTP
    # keys are present (has_mtp_weights=True). Models like oQ4-mtp have
    # norms already in MLX convention (mean > 0.5); the double-shift
    # corrupts normalization. Run this ONLY after confirming the model is
    # a real MTP checkpoint (num_mtp_layers >= 1) so a rejected/non-MTP
    # model is never mutated.
    _fix_mtp_norm_double_shift(inner)

    # --- Step 3 (early): Load MTP weights + infer config if auto-detecting ---
    # When model_repo is provided, load weights BEFORE building the MTP
    # module so we can infer the MTP head's actual dimensions from the
    # weight shapes. Models like oQ4-mtp may have MTP heads with
    # different architecture parameters than the backbone.
    mtp_weights: dict[str, Any] | None = None
    weights_source: str = ""
    _skip_quantize = False

    if mtp_sidecar is not None:
        weights_file = _resolve_sidecar_file(mtp_sidecar)
        if weights_file is None:
            logger.warning(
                "[mtp.inject] sidecar %r could not be resolved; skipping.",
                mtp_sidecar,
            )
            return False
        raw = mx.load(str(weights_file))
        mtp_weights = {
            (k.removeprefix("mtp.") if k.startswith("mtp.") else k): v
            for k, v in raw.items()
        }
        weights_source = str(weights_file)
    elif model_repo is not None:
        mtp_weights = _load_mtp_weights_from_repo(model_repo)
        if mtp_weights is not None:
            weights_source = model_repo
            _skip_quantize = True  # already quantized by model publisher
            logger.info(
                "[mtp.inject] Auto-detected %d MTP weight tensors from %r",
                len(mtp_weights),
                model_repo,
            )
            # Infer MTP head dimensions from weight shapes and apply
            # as overrides on the backbone args.
            mtp_overrides = _infer_mtp_config(mtp_weights)
            if mtp_overrides:
                for key, val in mtp_overrides.items():
                    try:
                        object.__setattr__(args, key, val)
                    except (TypeError, AttributeError):
                        logger.debug(
                            "[mtp.inject] Could not override args.%s=%r", key, val
                        )

    # --- Step 1: Build the MTP module from the vendored head ---
    from .head import build_mtp_module

    _use_gated = getattr(args, "_use_gated_attention", True)
    mtp = build_mtp_module(args, num_mtp_layers, use_gated_attention=_use_gated)
    logger.info(
        "[mtp.inject] Built MTP module (%d layer(s), hidden_size=%d).",
        num_mtp_layers,
        getattr(args, "hidden_size", -1),
    )

    # --- Step 2: Quantize MTP to match the base model's quantization ---
    # For auto-detect (model_repo), use per-layer quantization based on
    # which weights have scales/biases. For sidecar, use uniform
    # quantization from the backbone.
    if _skip_quantize and mtp_weights is not None:
        # Per-layer quantization: only quantize layers whose weights
        # have associated scales/biases tensors.
        _mtp_quant_map = _build_per_layer_quant_map(mtp_weights)
        if _mtp_quant_map:

            def _mtp_class_predicate(path, module):
                if path in _mtp_quant_map:
                    return _mtp_quant_map[path]
                # Layer has no quantized weights in the model's
                # safetensors — leave as bf16/fp16.
                return False

            # Use backbone defaults for the top-level call
            default_info = _detect_base_quantization(inner) or {
                "bits": 4,
                "group_size": 64,
            }
            nn.quantize(
                mtp,
                group_size=default_info["group_size"],
                bits=default_info["bits"],
                class_predicate=_mtp_class_predicate,
            )
            logger.info(
                "[mtp.inject] Per-layer quantized MTP (%d overrides)",
                len(_mtp_quant_map),
            )
    else:
        quant_info = _detect_base_quantization(inner)
        if quant_info is not None:
            nn.quantize(
                mtp,
                group_size=quant_info["group_size"],
                bits=quant_info["bits"],
            )
            logger.info(
                "[mtp.inject] Quantized MTP: %d-bit, group_size=%d",
                quant_info["bits"],
                quant_info["group_size"],
            )
        else:
            # FP base model — _detect_base_quantization returns None and the
            # MTP head stays in floating point. Do NOT dereference quant_info
            # here (it is None): logging its keys was an unconditional
            # NoneType subscript crash on every unquantised model.
            logger.info(
                "[mtp.inject] Base model is unquantised (FP); "
                "MTP head left in floating point."
            )

    # --- Step 3: Load MTP weights into the module ---
    if mtp_weights is not None:
        # Pre-load coverage check
        from mlx.utils import tree_flatten

        expected_keys = {k for k, _ in tree_flatten(mtp.parameters())}
        loaded_keys = set(mtp_weights.keys())
        missing = expected_keys - loaded_keys
        if missing:
            logger.warning(
                "[mtp.inject] %s is missing %d required MTP "
                "tensor(s); refusing to ship a partially-random-init head. "
                "Missing keys (first 8): %s.",
                weights_source,
                len(missing),
                sorted(missing)[:8],
            )
            return False
        mtp.load_weights(list(mtp_weights.items()), strict=False)
        mx.eval(mtp.parameters())
        extra = loaded_keys - expected_keys
        logger.info(
            "[mtp.inject] Loaded %d/%d expected MTP weight tensors from %s%s",
            len(expected_keys),
            len(expected_keys),
            weights_source,
            f" (+{len(extra)} extra key(s) ignored)" if extra else "",
        )
    else:
        # No sidecar and no auto-detected weights.
        if not allow_random_init:
            logger.warning(
                "[mtp.inject] inject_mtp_support called without "
                "mtp_sidecar or model_repo; refusing to "
                "ship a random-init MTP head. Pass "
                "mtp_sidecar='mlx-community/Qwen3.5-9B-MTP-4bit' (or "
                "equivalent) for production use, or set "
                "allow_random_init=True for unit-test wiring probes."
            )
            return False
        # Test-only path — explicit opt-in to random-init weights for
        # wiring tests that pin the surfaces without paying the
        # 131 MB sidecar download cost.
        mx.eval(mtp.parameters())
        logger.warning(
            "[mtp.inject] inject_mtp_support called with "
            "allow_random_init=True — MTP head retains RANDOM init "
            "weights (accept rate ~0%%). This is the test-only path; "
            "do not use in production."
        )

    # --- Step 4: Install global ArraysCache + GatedDeltaNet patches ---
    # Deferred from the top of this function so a failed validation /
    # sidecar load (above) leaves the process global state untouched.
    # Both patches are idempotent + transparent at n_confirmed=0, so
    # a successful inject_mtp_support that runs after a failed one
    # still lands cleanly.
    from .cache_patch import (
        patch_arrays_cache_rollback_state,
        patch_gated_delta_net_for_mtp,
    )

    patch_arrays_cache_rollback_state()
    patch_gated_delta_net_for_mtp()

    # --- Step 5: Attach + monkey-patch ``TextModel`` class ---
    inner.mtp = mtp
    original_class = type(inner)

    class _Qwen3_5WithMTP(original_class):  # type: ignore[valid-type, misc]
        """``TextModel`` + MTP surfaces injected by R15 #302 (vendor PR #990).

        The forward is inlined from
        ``mlx_lm.models.qwen3_5.Qwen3_5TextModel.__call__`` so that:

        * ``return_hidden=True`` can return the pre-norm hidden state
          the MTP head consumes (the upstream forward returns only the
          post-norm output).
        * ``n_confirmed`` is accepted on the signature for ABI parity
          with PR #990 (the generator passes ``n_confirmed=1`` during
          verify forwards). It is currently a no-op below this layer
          — the GatedDeltaNet rollback patch is tracked separately.
        """

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden: bool = False,
            n_confirmed: int = 0,
        ):
            from mlx_lm.models.base import create_attention_mask, create_ssm_mask

            inner_m = self.model
            if input_embeddings is not None:
                hidden_states = input_embeddings
            else:
                hidden_states = inner_m.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(inner_m.layers)

            # Tag each ArraysCache (linear-attention) with the
            # confirmed boundary so the patched GatedDeltaNet splits
            # ``gated_delta_update`` into two chunks and writes
            # ``(conv_snap, ssm_snap)`` to ``cache.rollback_state``.
            # KVCache slots ignore the tag — their rollback is the
            # existing ``c.trim(1)`` path. Tagged values are cleared
            # in the ``finally`` block so a later non-MTP forward
            # (mtp_forward, prefill, etc.) on the same cache list
            # doesn't accidentally re-trigger a split.
            if n_confirmed > 0:
                for c in cache:
                    if c is not None and hasattr(c, "rollback_state"):
                        c.n_confirmed_for_mtp = n_confirmed

            try:
                fa_mask = create_attention_mask(hidden_states, cache[inner_m.fa_idx])
                ssm_mask = create_ssm_mask(hidden_states, cache[inner_m.ssm_idx])
                for layer, c in zip(inner_m.layers, cache):
                    mask = ssm_mask if layer.is_linear else fa_mask
                    hidden_states = layer(hidden_states, mask=mask, cache=c)
            finally:
                if n_confirmed > 0:
                    for c in cache:
                        if c is not None and hasattr(c, "n_confirmed_for_mtp"):
                            c.n_confirmed_for_mtp = 0

            # Return PRE-norm hidden so MTP can apply its own
            # ``pre_fc_norm_hidden`` — matches PR #990's contract that
            # ``mtp_forward(hidden, ...)`` consumes pre-norm hidden.
            normed = inner_m.norm(hidden_states)
            if self.args.tie_word_embeddings:
                out = inner_m.embed_tokens.as_linear(normed)
            else:
                out = self.lm_head(normed)

            if return_hidden:
                return out, hidden_states
            return out

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache,
            *,
            return_hidden: bool = False,
        ):
            """Run the MTP head and project through the shared lm_head.

            When ``return_hidden=True``, returns ``(logits, mtp_hidden)``
            where ``mtp_hidden`` is the MTP module's pre-lm_head output
            at the last predicted position. Used by the generator's
            drafter-hidden cascade path so successive chain iterations
            see the drafter's own representation instead of the frozen
            backbone hidden.
            """
            mtp_out = self.mtp(
                hidden_states,
                next_token_ids,
                self.model.embed_tokens,
                mtp_cache,
            )
            if self.args.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(mtp_out)
            else:
                logits = self.lm_head(mtp_out)
            if return_hidden:
                return logits, mtp_out[:, -1:, :]
            return logits

        def make_mtp_cache(self):
            """Return fresh ``KVCache`` entries — one per MTP layer.

            All MTP layers are full-attention by design (PR #990's
            ``MTPDecoderLayer`` is hard-coded to ``self_attn =
            Attention(...)`` — see ``vllm_mlx/spec_decode/mtp/head.py``
            line 89-115). The MTP head deliberately does NOT include
            ``GatedDeltaNet`` linear-attention layers (the backbone's
            hybrid layout via ``args.full_attention_interval`` does not
            apply here). So ``KVCache`` is correct for every MTP layer
            — there are no ``ArraysCache`` slots to maintain on this
            side of the rollback. The ``ArraysCache.rollback_state``
            machinery installed by this PR exists to handle the
            BACKBONE's linear-attention layers (where the GatedDeltaNet
            patch lives), not the MTP head.
            """
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.layers]

    inner.__class__ = _Qwen3_5WithMTP
    logger.info(
        "[mtp.inject] Patched %s with MTP surfaces "
        "(return_hidden, n_confirmed, mtp_forward, make_mtp_cache).",
        original_class.__name__,
    )
    return True


def validate_mtp_support(model: Any) -> bool:
    """Verify that ``inject_mtp_support`` succeeded on ``model``.

    Used by the CLI's boot-time MTP wiring: the operator gets a
    clear warning if the injection silently dropped MTP rather than
    discovering it mid-generation when the first ``mtp_forward`` call
    raises ``AttributeError``.

    Checks:

    1. Model has ``mtp`` attribute (or ``model.mtp`` for the dense
       variant).
    2. ``mtp_forward`` is callable.
    3. ``make_mtp_cache`` is callable.
    4. ``__call__`` accepts ``return_hidden`` and ``n_confirmed``.
    """
    import inspect

    inner = _resolve_inner_text_model(model)
    if inner is None:
        return False

    if getattr(inner, "mtp", None) is None:
        logger.warning("[mtp.validate] model.mtp is missing.")
        return False
    if not callable(getattr(inner, "mtp_forward", None)):
        logger.warning("[mtp.validate] model.mtp_forward is missing.")
        return False
    if not callable(getattr(inner, "make_mtp_cache", None)):
        logger.warning("[mtp.validate] model.make_mtp_cache is missing.")
        return False
    sig = inspect.signature(type(inner).__call__)
    if "return_hidden" not in sig.parameters:
        logger.warning("[mtp.validate] model.__call__ does not accept return_hidden.")
        return False
    if "n_confirmed" not in sig.parameters:
        logger.warning("[mtp.validate] model.__call__ does not accept n_confirmed.")
        return False
    return True
