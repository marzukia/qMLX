# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for qmlx inference.

BatchedEngine is the sole engine — continuous batching for all workloads.
"""

from .base import BaseEngine, GenerationOutput

__all__ = [
    "BaseEngine",
    "GenerationOutput",
    "BatchedEngine",
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]


def __getattr__(name):
    """Lazy-load the MLX-dependent engine components.

    ``engine_core`` and ``batched`` import ``mlx.core`` at module level, so
    they are deferred to first use. This keeps ``import vllm_mlx.engine`` —
    and the ``vllm_mlx.engine.base`` / ``vllm_mlx.config`` /
    ``vllm_mlx.routes.*`` import chains that only need the MLX-free ``base``
    types — free of ``mlx`` on non-Apple-Silicon platforms (e.g. Linux CI),
    matching the lazy-import convention in ``vllm_mlx/__init__``.
    """
    if name in ("EngineCore", "AsyncEngineCore", "EngineConfig"):
        from vllm_mlx import engine_core

        return getattr(engine_core, name)

    if name == "BatchedEngine":
        from .batched import BatchedEngine

        return BatchedEngine

    if name == "batched":
        import importlib

        return importlib.import_module(".batched", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
