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


class PrefillProfiler:
    """Profiles prefill by wrapping model layers to time each phase."""

    def __init__(self):
        self._enabled = _is_profiler_enabled()
        self._phase_totals: dict[str, float] = defaultdict(float)
        self._wrapped: bool = False

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

    def wrap_layers_for_profiling(self, model: Any):
        """Wrap model layers to profile each phase during forward pass."""
        if not self._enabled or self._wrapped:
            return

        layers = getattr(model, "layers", [])
        if not layers:
            return

        # Wrap each layer type with proper closure handling
        for layer in layers:
            class_name = type(layer).__name__

            # Wrap attention layers
            if "Attention" in class_name or "Attn" in class_name:
                if hasattr(layer, "__call__"):
                    orig_call = layer.__call__

                    def make_attn_wrapper(original):
                        def wrapped(*args, **kwargs):
                            start = time.monotonic()
                            try:
                                return original(*args, **kwargs)
                            finally:
                                self._phase_totals["attention"] += time.monotonic() - start

                        return wrapped

                    layer.__call__ = make_attn_wrapper(orig_call)

            # Wrap DeltaNet layers
            elif "DeltaNet" in class_name:
                if hasattr(layer, "__call__"):
                    orig_call = layer.__call__

                    def make_deltanet_wrapper(original):
                        def wrapped(*args, **kwargs):
                            start = time.monotonic()
                            try:
                                return original(*args, **kwargs)
                            finally:
                                self._phase_totals["deltanet"] += time.monotonic() - start

                        return wrapped

                    layer.__call__ = make_deltanet_wrapper(orig_call)

            # Wrap FFN/MLP modules within layers
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                if hasattr(mlp, "__call__"):
                    orig_mlp = mlp.__call__

                    def make_ffn_wrapper(original):
                        def wrapped(*args, **kwargs):
                            start = time.monotonic()
                            try:
                                return original(*args, **kwargs)
                            finally:
                                self._phase_totals["ffn"] += time.monotonic() - start

                        return wrapped

                    mlp.__call__ = make_ffn_wrapper(orig_mlp)

            elif hasattr(layer, "feed_forward"):
                ffn = layer.feed_forward
                if hasattr(ffn, "__call__"):
                    orig_ffn = ffn.__call__

                    def make_ffn_wrapper(original):
                        def wrapped(*args, **kwargs):
                            start = time.monotonic()
                            try:
                                return original(*args, **kwargs)
                            finally:
                                self._phase_totals["ffn"] += time.monotonic() - start

                        return wrapped

                    ffn.__call__ = make_ffn_wrapper(orig_ffn)

        self._wrapped = True

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

        # Metrics will be emitted by the wrappers during the forward pass

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
    """Install profiling hooks on BatchGenerator for prefill measurement."""
    if not _is_profiler_enabled():
        # Zero-overhead: no monkey patching when disabled
        return

    profiler = get_profiler()
    logger = __import__("logging").getLogger(__name__)

    # Wrap the model's forward pass to detect prefill operations
    orig_model_call = batch_gen.model.__call__

    def profiled_model_call(*args, **kwargs):
        inputs = args[0] if args else kwargs.get("inputs")

        # Detect prefill: multi-token input (prompt processing)
        if hasattr(inputs, "shape") and len(inputs.shape) > 1:
            num_tokens = inputs.shape[1]
            if num_tokens > 1:
                profiler.record_and_emit_prefill(batch_gen.model, num_tokens)

        result = orig_model_call(*args, **kwargs)

        # Emit metrics after forward completes
        if hasattr(inputs, "shape") and len(inputs.shape) > 1:
            profiler.emit_metrics(inputs.shape[1])

        return result

    batch_gen.model.__call__ = profiled_model_call

    logger.info("[QMLX_PREFILL_PROFILER] Installed profiling on BatchGenerator.model")


def dump_prefill_profile():
    """Dump current profiling results to stdout in Prometheus format."""
    if _profiler:
        _profiler.emit_metrics(0)