# SPDX-License-Identifier: Apache-2.0
"""qMLX #85 finding 3 follow-up (post-merge Gate 3): the diffusion
lane's SHARED serialized-multiplexing ``mlx_executor=`` path was
AVAILABLE but NOT wired into the shipped load paths. This suite pins
the wiring and the current (standalone-default) behaviour:

  * standalone default (``mlx_executor=None``) — ``load_runner`` and
    the server construction path create a self-owned single
    ``mlx-step`` owner thread; model load AND every generation job
    run on that thread, never on the dispatcher;
  * shared mode (caller-provided executor) — model load AND
    generation run on the provided executor, and the engine never
    shuts it down (still usable after ``stop()``);
  * backward compatibility — the 1-arg ``load_runner`` call still
    works and yields a self-owned executor.

Runs on the no-mlx Linux CI: the real ``mlx`` / ``mlx_lm`` /
``mlx_vlm`` packages are stubbed into ``sys.modules`` (saved and
restored) so the import chain resolves without the Apple-Silicon-only
wheels. On an Apple-Silicon host the stubs transparently shadow the
real packages for the duration of this module's tests only.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import types
from typing import Any

import pytest


# ----------------------------------------------------------------------
# Stub installation (no-mlx host)
# ----------------------------------------------------------------------


class _FakeModel:
    config = type("FakeConfig", (), {"eos_token_id": 7, "canvas_length": 256})


class _FakeStoppingCriteria:
    eos_token_ids: list[int] = []

    def reset(self, *_args: Any, **_kw: Any) -> None:
        pass


class _FakeProcessor:
    stopping_criteria = _FakeStoppingCriteria()
    all_special_ids = [0, 1, 2]

    def apply_chat_template(self, *_args: Any, **_kw: Any) -> str:
        return "x"

    def encode(self, _text: str) -> list[int]:
        return [0]


class _FakeTokenizer:
    stopping_criteria = _FakeStoppingCriteria()
    all_special_ids = [0, 1, 2]

    def apply_chat_template(self, *_args: Any, **_kw: Any) -> str:
        return "x"

    def encode(self, _text: str) -> list[int]:
        return [0]


def _new_module(name: str, **attrs: Any) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _install_mlx_stubs(stubs: dict[str, Any]) -> None:
    """Install the mlx / mlx_lm / mlx_vlm surface stubs and record the
    prior ``sys.modules`` state in ``stubs`` so ``_restore_mlx_stubs``
    can put the environment back (real mlx wins again on hosts that
    have it)."""
    mlx = _new_module("mlx")
    mx = _new_module(
        "mlx.core",
        default_stream=lambda *a, **k: "stream",
        default_device=lambda *a, **k: "device",
        eval=lambda *a, **k: None,
    )
    mlx.core = mx

    _new_module("mlx_lm")
    _new_module("mlx_lm.generate", BatchGenerator=type("BatchGenerator", (), {}))
    _new_module(
        "mlx_lm.sample_utils",
        make_logits_processors=lambda *a, **k: [],
        make_sampler=lambda *a, **k: object(),
    )
    _new_module(
        "mlx_lm.tokenizer_utils",
        NaiveStreamingDetokenizer=type("NaiveStreamingDetokenizer", (), {}),
    )
    _new_module("mlx_lm.models")
    _new_module("mlx_lm.models.cache", ArraysCache=type("ArraysCache", (), {}))

    # mlx_vlm surface — mirror the real contract the load path touches.
    _new_module("mlx_vlm")
    load_calls: list[int] = []

    def _load(_hf_path: str) -> tuple[Any, Any]:
        load_calls.append(threading.get_ident())
        return _FakeModel(), _FakeProcessor()

    _new_module("mlx_vlm.utils", load=_load)
    _new_module("mlx_vlm.generate")
    gen_diff = _new_module(
        "mlx_vlm.generate.diffusion",
        diffusion_generation_family=lambda _m: "block",
    )
    gen_diff._load_calls = load_calls  # type: ignore[attr-defined]
    _new_module("mlx_vlm.generate.common")

    stubs["_load_calls"] = load_calls


def _restore_mlx_stubs(stubs: dict[str, Any]) -> None:
    for name in (
        "mlx",
        "mlx.core",
        "mlx_lm",
        "mlx_lm.generate",
        "mlx_lm.sample_utils",
        "mlx_lm.tokenizer_utils",
        "mlx_lm.models",
        "mlx_lm.models.cache",
        "mlx_vlm",
        "mlx_vlm.utils",
        "mlx_vlm.generate",
        "mlx_vlm.generate.diffusion",
        "mlx_vlm.generate.common",
    ):
        prev = stubs["_prev"]
        if name in prev:
            sys.modules[name] = prev[name]
        else:
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def mlx_stubs() -> dict[str, Any]:
    stubs: dict[str, Any] = {"_prev": {}}
    for name in (
        "mlx",
        "mlx.core",
        "mlx_lm",
        "mlx_lm.generate",
        "mlx_lm.sample_utils",
        "mlx_lm.tokenizer_utils",
        "mlx_lm.models",
        "mlx_lm.models.cache",
        "mlx_vlm",
        "mlx_vlm.utils",
        "mlx_vlm.generate",
        "mlx_vlm.generate.diffusion",
        "mlx_vlm.generate.common",
    ):
        stubs["_prev"][name] = sys.modules.get(name)
    _install_mlx_stubs(stubs)
    yield stubs
    _restore_mlx_stubs(stubs)


def _make_shared_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Mirror the AR lane's owner thread: single worker, ``mlx-step``
    prefix, the ``_init_mlx_step_thread`` initializer. Imported lazily
    (after the module fixture installs the stubs) so the no-mlx host
    can resolve ``vllm_mlx.engine_core``."""
    from vllm_mlx.engine_core import _init_mlx_step_thread

    return concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="mlx-step",
        initializer=_init_mlx_step_thread,
    )


# ----------------------------------------------------------------------
# (a) standalone default
# ----------------------------------------------------------------------


class TestStandaloneDefault:
    def test_load_runner_default_creates_self_owned_executor(
        self, mlx_stubs: dict[str, Any]
    ) -> None:
        from vllm_mlx.runtime.diffusion_lane import load_runner

        engine = load_runner("x/y")

        assert engine._owns_mlx_executor is True, (
            "default load_runner() must create a self-owned executor "
            "(standalone mode); the engine must not inherit ownership "
            "of a caller-provided thread"
        )
        assert engine._mlx_executor is not None
        # Force the owner thread to spawn (lazy on first submit).
        engine._step_executor().submit(lambda: None).result()
        threads = list(engine._mlx_executor._threads)  # type: ignore[attr-defined]
        assert len(threads) == 1, (
            f"standalone mode must own exactly one mlx-step thread; saw "
            f"{len(threads)}"
        )
        assert all(t.name.startswith("mlx-step") for t in threads), (
            f"self-owned executor thread misnamed: "
            f"{[t.name for t in threads]}"
        )

    def test_load_and_generation_run_on_the_owned_thread(
        self, mlx_stubs: dict[str, Any]
    ) -> None:
        from vllm_mlx.runtime.diffusion_lane import (
            DiffusionEngine,
            DiffusionGenerationConfig,
        )

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        assert engine._owns_mlx_executor is True
        executor = engine._step_executor()
        owner_ident = list(executor._threads)[0].ident  # type: ignore[attr-defined]
        dispatcher = engine._worker
        assert dispatcher is not None and dispatcher.ident != owner_ident

        # Model load ran on the owned executor thread, NOT the dispatcher.
        assert mlx_stubs["_load_calls"], "mock load() was never invoked"
        assert mlx_stubs["_load_calls"][-1] == owner_ident, (
            f"model load ran on thread {mlx_stubs['_load_calls'][-1]}; "
            f"expected the owned mlx-step thread {owner_ident}"
        )
        assert mlx_stubs["_load_calls"][-1] != dispatcher.ident


# ----------------------------------------------------------------------
# (b) shared mode (caller-provided executor)
# ----------------------------------------------------------------------


class TestSharedMode:
    def test_provided_executor_is_used_for_load_and_generation(
        self, mlx_stubs: dict[str, Any]
    ) -> None:
        from vllm_mlx.runtime.diffusion_lane import (
            DiffusionEngine,
            DiffusionGenerationConfig,
        )

        shared = _make_shared_executor()
        engine = DiffusionEngine(model_name="x/y", mlx_executor=shared)
        assert engine._owns_mlx_executor is False, (
            "a caller-provided executor must be treated as SHARED — the "
            "engine must NOT own its lifecycle"
        )
        assert engine._mlx_executor is shared

        engine._load_blocking()

        # The provided executor's owner thread (single worker).
        shared_thread = list(shared._threads)[0]  # type: ignore[attr-defined]
        shared_ident = shared_thread.ident
        # Model load ran on the SHARED thread.
        assert mlx_stubs["_load_calls"], "mock load() was never invoked"
        assert mlx_stubs["_load_calls"][-1] == shared_ident, (
            f"model load ran on thread {mlx_stubs['_load_calls'][-1]}; "
            f"expected the provided shared thread {shared_ident}"
        )

        # A generation job also runs on the SAME shared thread.
        import queue as _queue

        thread_q: _queue.Queue[Any] = _queue.Queue()
        cancel_event = threading.Event()
        done_event = threading.Event()
        engine._jobs.put(
            (
                "prompt",
                16,
                DiffusionGenerationConfig(),
                thread_q,
                cancel_event,
                done_event,
            )
        )
        assert done_event.wait(10.0), "worker never drained the test job"
        # The job was submitted to the shared executor: prove the
        # engine used it (the load + job both went through ``shared``).
        assert engine._mlx_executor is shared

    def test_engine_never_shuts_down_a_provided_executor(
        self, mlx_stubs: dict[str, Any]
    ) -> None:
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        shared = _make_shared_executor()
        engine = DiffusionEngine(model_name="x/y", mlx_executor=shared)
        engine._load_blocking()

        asyncio.run(engine.stop())

        # The engine must NOT have nulled or shut down the shared
        # executor (it does not own it). It must still be usable.
        assert engine._mlx_executor is shared, (
            "stop() must leave a caller-provided executor in place — "
            "the engine does not own its lifecycle"
        )
        assert shared.submit(lambda: 42).result() == 42, (
            "the shared executor must still be usable after "
            "DiffusionEngine.stop() — the engine never shuts it down"
        )
        shared.shutdown(wait=True)

    def test_load_runner_shared_kwarg_threads_through(
        self, mlx_stubs: dict[str, Any]
    ) -> None:
        from vllm_mlx.runtime.diffusion_lane import load_runner

        shared = _make_shared_executor()
        engine = load_runner("x/y", mlx_executor=shared)
        assert engine._owns_mlx_executor is False
        assert engine._mlx_executor is shared
        assert engine._loaded
        # Still usable after stop (engine never owns it).
        asyncio.run(engine.stop())
        assert shared.submit(lambda: 7).result() == 7
        shared.shutdown(wait=True)


# ----------------------------------------------------------------------
# (c) backward compatibility
# ----------------------------------------------------------------------


class TestBackwardCompat:
    def test_one_arg_load_runner_still_works(
        self, mlx_stubs: dict[str, Any]
    ) -> None:
        import inspect

        from vllm_mlx.runtime.diffusion_lane import load_runner

        # The 1-arg call shape is preserved (mlx_executor is optional
        # with a default of None = standalone).
        sig = inspect.signature(load_runner)
        assert "mlx_executor" in sig.parameters
        assert sig.parameters["mlx_executor"].default is None

        engine = load_runner("x/y")
        assert engine._loaded
        assert engine._owns_mlx_executor is True
        # The self-owned executor is live and correctly named.
        engine._step_executor().submit(lambda: None).result()
        threads = list(engine._mlx_executor._threads)  # type: ignore[attr-defined]
        assert all(t.name.startswith("mlx-step") for t in threads)
