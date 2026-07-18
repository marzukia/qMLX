# SPDX-License-Identifier: Apache-2.0
"""Prefill phase profiler for qMLX.

Instrument the model forward pass to measure where prefill time is spent.
Outputs Prometheus counters breaking down prefill time into:
- FFN/MoE: Feed-forward network and expert routing time
- Attention: Full attention score + softmax computation
- DeltaNet: Recurrent state updates (for hybrid models)
- Overhead: RoPE, normalization, residual connections

Gated behind QMLX_PREFILL_PROFILER_ENABLED env var (default OFF).

Usage:
    export QMLX_PREFILL_PROFILER_ENABLED=1
    python -m vllm_mlx.serve ...

The profiler adds minimal overhead when disabled (single env var check).
"""

import os
import time
from collections import defaultdict
from typing import Any


def _is_profiler_enabled() -> bool:
    """Check if prefill profiler is enabled via env var. Single fast check."""
    return os.environ.get("QMLX_PREFILL_PROFILER_ENABLED", "").lower() in ("1", "true", "yes")


def _get_seq_bucket(num_tokens: int) -> str:
    """Map token count to sequence bucket label for metrics."""
    if num_tokens <= 8192:
        return "8k"
    elif num_tokens <= 16384:
        return "16k"
    elif num_tokens <= 32768:
        return "32k"
    else:
        return "32k+"


# Sentinel attribute name to avoid double-wrapping a class
_PROFILER_PATCHED_ATTR = "_qmlx_profiler_patched"


class PrefillProfiler:
    """Profiles prefill by wrapping model layers to time each phase.

    IMPORTANT: Python dunder method lookup (e.g. __call__) bypasses instance
    attributes and goes straight to the class.  We MUST patch the class-level
    __call__, not the instance.  This is safe because qMLX runs with
    --max-num-seqs 1 (single request at a time).
    """

    def __init__(self):
        self._enabled = _is_profiler_enabled()
        self._phase_totals: dict[str, float] = defaultdict(float)
        self._wrapped: bool = False
        self._model_wrapped: bool = False
        self._call_count: int = 0
        self._num_layers: int = 0
        self._logger = __import__("logging").getLogger(__name__)

    @property
    def enabled(self) -> bool:
        """Fast check if profiling is enabled."""
        return self._enabled

    def analyze_and_log_model_config(self, model: Any):
        """Inspect model.layers and log real model config exactly once."""
        if not self._enabled:
            return

        layers = getattr(model, "layers", [])
        total_layers = len(layers)

        attn_count = 0
        deltanet_count = 0

        for layer in layers:
            class_name = type(layer).__name__

            if "Attention" in class_name or "Attn" in class_name:
                attn_count += 1
            elif "DeltaNet" in class_name:
                deltanet_count += 1

        # Extract model config if available
        hidden_dim = 0
        num_heads = 0
        moe_experts = 0
        moe_top_k = 0

        if hasattr(model, "config"):
            config = model.config
            hidden_dim = getattr(config, "hidden_size", 0)
            num_heads = getattr(config, "num_attention_heads", 0)

        # Check for MoE in any layer
        for layer in layers:
            if hasattr(layer, "router") or hasattr(layer, "gate"):
                if hasattr(layer, "experts"):
                    moe_experts = len(getattr(layer, "experts", []))
                if hasattr(layer, "top_k"):
                    moe_top_k = getattr(layer, "top_k", 0)

        # Calculate softmax:DeltaNet ratio
        if deltanet_count > 0:
            ratio = f"{attn_count}/{deltanet_count}={attn_count / deltanet_count:.1f}"
        elif attn_count > 0:
            ratio = "inf (pure attention)"
        else:
            ratio = "N/A"

        import logging

        logger = logging.getLogger(__name__)

        logger.info(
            "[QMLX_PREFILL_PROFILER] Model config: "
            f"layer_count={total_layers}, "
            f"softmax:delta={ratio}, "
            f"MoE: experts={moe_experts}, top_k={moe_top_k}, "
            f"hidden_dim={hidden_dim}, heads={num_heads}"
        )

    def _patch_class_call(self, obj: Any, phase: str):
        """Patch the class-level __call__ of obj's class.

        Returns True if a new patch was applied, False if already patched.
        """
        cls = type(obj)
        if getattr(cls, _PROFILER_PATCHED_ATTR, False):
            return False

        original_call = cls.__call__
        profiler = self

        def profiled_call(self_inner, *args, **kwargs):
            start = time.monotonic()
            try:
                result = original_call(self_inner, *args, **kwargs)
                # Force GPU sync to get real wall-clock time, not just
                # graph-construction time (MLX uses lazy evaluation).
                import mlx.core as mx
                mx.eval(result)
                return result
            finally:
                elapsed = time.monotonic() - start
                profiler._phase_totals[phase] += elapsed

        cls.__call__ = profiled_call
        setattr(cls, _PROFILER_PATCHED_ATTR, True)
        self._logger.debug(
            "[QMLX_PREFILL_PROFILER] Patched %s.%s.__call__ for phase=%s",
            cls.__module__, cls.__name__, phase,
        )
        return True

    def wrap_layers_for_profiling(self, model: Any):
        """Wrap model layers to profile each phase during forward pass.

        Patches the CLASS-level __call__ (not instance) because Python dunder
        method lookup bypasses instance attributes.
        """
        if not self._enabled or self._wrapped:
            return

        layers = getattr(model, "layers", [])
        if not layers:
            return

        self._num_layers = len(layers)

        patched_classes: set[str] = set()

        for layer in layers:
            # Attention layers
            if hasattr(layer, "self_attn"):
                cls_name = type(layer.self_attn).__name__
                if cls_name not in patched_classes:
                    if self._patch_class_call(layer.self_attn, "attention"):
                        patched_classes.add(cls_name)

            # DeltaNet / linear attention layers
            if hasattr(layer, "linear_attn"):
                cls_name = type(layer.linear_attn).__name__
                if cls_name not in patched_classes:
                    if self._patch_class_call(layer.linear_attn, "deltanet"):
                        patched_classes.add(cls_name)

            # FFN / MLP
            if hasattr(layer, "mlp"):
                cls_name = type(layer.mlp).__name__
                if cls_name not in patched_classes:
                    if self._patch_class_call(layer.mlp, "ffn"):
                        patched_classes.add(cls_name)

        self._wrapped = True

        self._logger.info(
            "[QMLX_PREFILL_PROFILER] Patched %d unique layer classes for profiling",
            len(patched_classes),
        )

    def wrap_model_call(self, model: Any):
        """Wrap model.__call__ to trigger profiler dump after each prefill pass.

        This is the key addition: the layer-level patches accumulate timing,
        but we need a model-level wrapper to detect prefill vs decode and
        trigger the dump at the right time.
        """
        if not self._enabled or self._model_wrapped:
            return

        model_cls = type(model)
        if getattr(model_cls, _PROFILER_PATCHED_ATTR, False):
            return

        original_call = model_cls.__call__
        profiler = self
        in_prefill = [False]  # mutable to allow closure mutation
        num_prefill_chunks = [0]
        prefill_start_time = [0.0]

        def profiled_model_call(self_inner, *args, **kwargs):
            # Detect prefill: first arg is input_ids, multi-token = prefill
            is_prefill = False
            if args:
                inputs = args[0]
                if hasattr(inputs, "shape") and len(inputs.shape) > 1:
                    if inputs.shape[1] > 1:
                        is_prefill = True

            # Transition: prefill -> decode (or end of prefill)
            if in_prefill[0] and not is_prefill:
                # Dump accumulated prefill timing
                if profiler._phase_totals:
                    wall = time.monotonic() - prefill_start_time[0]
                    total = sum(profiler._phase_totals.values())
                    parts = []
                    for phase in ("attention", "deltanet", "ffn"):
                        t = profiler._phase_totals.get(phase, 0)
                        pct = (t / total * 100) if total > 0 else 0
                        parts.append(f"{phase}={t:.3f}s ({pct:.1f}%)")
                    profiler._logger.info(
                        "[PREFILL_PROFILER] %s | measured=%.3f wall=%.3f chunks=%d",
                        " | ".join(parts), total, wall, num_prefill_chunks[0],
                    )
                    profiler._phase_totals.clear()
                in_prefill[0] = False
                num_prefill_chunks[0] = 0

            # Transition: decode -> prefill (start of new prefill)
            if is_prefill and not in_prefill[0]:
                in_prefill[0] = True
                prefill_start_time[0] = time.monotonic()
                profiler._phase_totals.clear()

            if is_prefill:
                num_prefill_chunks[0] += 1

            result = original_call(self_inner, *args, **kwargs)
            return result

        model_cls.__call__ = profiled_model_call
        setattr(model_cls, _PROFILER_PATCHED_ATTR, True)
        self._model_wrapped = True

        self._logger.info(
            "[QMLX_PREFILL_PROFILER] Wrapped model.__call__ for prefill dump trigger"
        )

    def dump_if_active(self):
        """Manual dump for debugging."""
        if self._phase_totals:
            self._dump_and_reset()

    def _dump_and_reset(self):
        """Log accumulated timing and reset counters."""
        total = sum(self._phase_totals.values())
        if total <= 0:
            self._call_count = 0
            return

        parts = []
        for phase in ("attention", "deltanet", "ffn"):
            t = self._phase_totals.get(phase, 0)
            pct = (t / total * 100) if total > 0 else 0
            parts.append(f"{phase}={t:.3f}s ({pct:.1f}%)")

        self._logger.info(
            "[PREFILL_PROFILER] %s | total=%.3f%s",
            " | ".join(parts), total, "s"
        )

        self._phase_totals.clear()
        self._call_count = 0

    def record_and_emit_prefill(self, model: Any, num_tokens: int):
        """Profile a complete prefill pass and emit metrics."""
        if not self._enabled:
            return

        # Clear previous timing data
        self._phase_totals.clear()

        # Log model config on first wrap
        if not self._wrapped:
            self.analyze_and_log_model_config(model)
            self.wrap_layers_for_profiling(model)
            self.wrap_model_call(model)

    def emit_metrics(self, num_tokens: int):
        """Emit Prometheus-formatted metrics for current prefill."""
        if not self._enabled:
            return

        seq_bucket = _get_seq_bucket(num_tokens)

        # Output in Prometheus exposition format
        for phase, total_time in self._phase_totals.items():
            print(f"# TYPE qmlx_prefill_phase_seconds counter")
            print(f'qmlx_prefill_phase_seconds{{phase="{phase}",seq_bucket="{seq_bucket}"}} {total_time}')

    def reset(self):
        """Reset accumulated timing data."""
        self._phase_totals.clear()


# Global profiler instance (lazy initialization)
_profiler: PrefillProfiler | None = None


def get_profiler() -> PrefillProfiler:
    """Get global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = PrefillProfiler()
    return _profiler


def install_prefill_profiling(batch_gen: Any):
    """Install profiling hooks on model layers for prefill measurement."""
    if not _is_profiler_enabled():
        return

    profiler = get_profiler()
    logger = __import__("logging").getLogger(__name__)

    model = batch_gen.model
    profiler.analyze_and_log_model_config(model)
    profiler.wrap_layers_for_profiling(model)
    profiler.wrap_model_call(model)

    logger.info(
        "[QMLX_PREFILL_PROFILER] Installed layer profiling on %d layers",
        len(getattr(model, "layers", [])),
    )


def dump_prefill_profile():
    """Dump current profiling results to stdout in Prometheus format."""
    if _profiler:
        _profiler.emit_metrics(0)
