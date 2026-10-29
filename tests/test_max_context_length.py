# SPDX-License-Identifier: Apache-2.0
"""Issue #33: ``--max-context-length`` — the server-side context cap.

Coverage:

1. ``vllm_mlx.cli`` argparse — the flag is registered on the serve
   command with the ``0`` (no cap) default.
2. CLI -> ``SchedulerConfig`` plumbing — ``serve_command`` passes
   ``max_context_length`` to its ``SchedulerConfig(...)`` construction
   (AST-level, so it runs on any machine without mlx, matching the
   ``test_cli_config_fidelity.py`` layer-1 pattern).
3. ``SchedulerConfig.max_context_length`` — default ``0`` so
   pre-#33 callers keep the model-max behaviour.
4. ``get_server_context_cap`` — resolves the cap through every
   engine-config shape and returns ``None`` when the flag is absent.
5. Admission enforcement — a request whose ``prompt + max_tokens``
   exceeds the operator cap gets HTTP 400
   ``context_length_exceeded``; under the cap it passes; with the
   flag absent (``0``) the model max stays the binding limit
   (pre-#33 behaviour preserved).
6. Real ``BatchedEngine`` — the cap resolves against the LIVE serve
   shape (the engine ``get_engine()`` hands to the gated routes), not
   just the three stub shapes. This is the t981 critical guard: the
   engine stores its config only as the private ``_scheduler_config``,
   so a stub-only suite passed while production was dead.

Runs on any machine without mlx: the real ``mlx`` / ``mlx_lm``
packages are stubbed into ``sys.modules`` for the duration of this
module's tests (saved and restored, no ``conftest`` pollution) so
``vllm_mlx.scheduler`` / ``vllm_mlx.engine`` import on a no-mlx host.
On an Apple-Silicon host the stubs transparently shadow the real
packages while this module's tests run, then the real modules win
again on teardown.
"""

from __future__ import annotations

import ast
import importlib.machinery
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# mlx stubs (no-mlx host; save/restore ``sys.modules``, no conftest)
# ----------------------------------------------------------------------

_MLX_STUB_NAMES = (
    "mlx",
    "mlx.core",
    "mlx_lm",
    "mlx_lm.generate",
    "mlx_lm.sample_utils",
    "mlx_lm.tokenizer_utils",
    "mlx_lm.models",
    "mlx_lm.models.cache",
)


def _install_mlx_stubs() -> None:
    """Install the minimal ``mlx`` / ``mlx_lm`` surface the module-level
    import chain of ``vllm_mlx.scheduler`` / ``vllm_mlx.engine`` binds:
    ``import mlx.core as mx`` (the ``_mlx_compat`` shim is a no-op
    because ``mx.new_thread_local_stream`` is absent) and the three
    ``mlx_lm`` symbols ``scheduler.py`` imports. Nothing else is
    stubbed; the tests never execute model code."""
    for name in _MLX_STUB_NAMES:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        # A real ``__spec__`` so ``importlib.util.find_spec`` (e.g.
        # transformers' ``is_mlx_available`` probe) does not raise
        # ``ValueError: mlx.__spec__ is None`` for the stub.
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = module
        parent_name, _, child_name = name.rpartition(".")
        if parent_name and parent_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, module)

    sys.modules["mlx_lm.generate"].BatchGenerator = type("BatchGenerator", (), {})  # type: ignore[attr-defined]
    sys.modules["mlx_lm.sample_utils"].make_logits_processors = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: []
    )
    sys.modules["mlx_lm.sample_utils"].make_sampler = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: object()
    )
    sys.modules["mlx_lm.tokenizer_utils"].NaiveStreamingDetokenizer = (  # type: ignore[attr-defined]
        type("NaiveStreamingDetokenizer", (), {})
    )
    # ``vllm_mlx.utils.mamba_cache`` (imported by the scheduler) binds
    # ``MambaCache`` from here at module level.
    sys.modules["mlx_lm.models.cache"].MambaCache = (  # type: ignore[attr-defined]
        type("MambaCache", (), {})
    )


def _restore_mlx_stubs(saved: dict[str, object]) -> None:
    for name in _MLX_STUB_NAMES:
        prev = saved.get(name)
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev  # type: ignore[assignment]


@pytest.fixture(scope="module", autouse=True)
def mlx_stubs() -> None:
    """Stub ``mlx`` / ``mlx_lm`` into ``sys.modules`` for this module's
    tests and restore the previous state on teardown, so the
    ``vllm_mlx.scheduler`` / ``vllm_mlx.engine`` imports inside the
    tests resolve on a no-mlx host. Real mlx wins again on hosts that
    have it (the stubs only shadow it while these tests run)."""
    saved = {name: sys.modules.get(name) for name in _MLX_STUB_NAMES}
    _install_mlx_stubs()
    yield
    _restore_mlx_stubs(saved)


# ---------------------------------------------------------------------------
# 1. CLI argparse
# ---------------------------------------------------------------------------


def _serve_help() -> str:
    """Run ``python -m vllm_mlx.cli serve --help`` and return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_serve_help_advertises_max_context_length_flag_with_default_zero():
    """The flag and its no-cap default must appear in --help.

    Assert against the argparse-rendered ``--max-context-length``
    option string plus the ``default: 0`` marker in the help text so
    a flag renamed or re-defaulted cannot pass by accident.
    """
    text = _serve_help()
    assert "--max-context-length" in text
    assert "default: 0" in text


# ---------------------------------------------------------------------------
# 2. CLI -> SchedulerConfig plumbing (AST, mirrors
#    test_cli_config_fidelity.py layer 1)
# ---------------------------------------------------------------------------


def _serve_command_scheduler_config_kwargs() -> set[str]:
    tree = ast.parse((REPO_ROOT / "vllm_mlx" / "cli.py").read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "serve_command"):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "SchedulerConfig"
            ):
                return {kw.arg for kw in call.keywords if kw.arg}
    raise AssertionError(
        "serve_command or its SchedulerConfig(...) call not found in cli.py"
    )


def test_serve_command_plumbs_max_context_length_into_scheduler_config():
    """#33 regression guard: ``serve_command`` reads
    ``args.max_context_length`` and MUST pass it to
    ``SchedulerConfig``.

    Without this kwarg the flag would be accepted by argparse and
    silently dropped — the same bug class as #400
    (``--prefill-step-size``). The audit script
    (scripts/audit_cli_config_fidelity.py) covers this at the CI
    level; this test pins it at the unit level.
    """
    assert "max_context_length" in _serve_command_scheduler_config_kwargs()


def test_scheduler_config_has_max_context_length_field_defaulting_to_zero():
    """Default ``0`` = no cap, so callers who never set the field
    (and every pre-#33 deployment) keep the model-max behaviour."""
    from vllm_mlx.scheduler import SchedulerConfig

    config = SchedulerConfig()
    assert config.max_context_length == 0
    # Round-trips when set.
    capped = SchedulerConfig(max_context_length=196608)
    assert capped.max_context_length == 196608


# ---------------------------------------------------------------------------
# 3. get_server_context_cap resolution
# ---------------------------------------------------------------------------


class _StubSchedulerConfig:
    def __init__(self, max_context_length: int = 0):
        self.max_context_length = max_context_length


class _StubEngine:
    """Engine with the model's max context and an optional operator
    cap, exposed through one of the three config shapes
    ``get_server_context_cap`` probes."""

    def __init__(
        self,
        model_max: int | None = None,
        server_cap: int | None = None,
        config_shape: str = "scheduler",
    ):
        class _Model:
            pass

        self._model = _Model()
        if model_max is not None:
            self._model.args = type("Args", (), {"max_position_embeddings": model_max})
        if server_cap is None:
            return
        if config_shape == "scheduler":
            self.scheduler = type(
                "Sched", (), {"config": _StubSchedulerConfig(server_cap)}
            )()
        elif config_shape == "direct":
            self.scheduler_config = _StubSchedulerConfig(server_cap)
        elif config_shape == "holder":

            class _Holder:
                pass

            self.config = _Holder()
            self.config.scheduler_config = _StubSchedulerConfig(server_cap)


def test_server_cap_resolves_via_scheduler_config():
    from vllm_mlx.service.helpers import get_server_context_cap

    engine = _StubEngine(
        model_max=500_000, server_cap=196_608, config_shape="scheduler"
    )
    assert get_server_context_cap(engine) == 196_608


def test_server_cap_resolves_via_direct_scheduler_config():
    from vllm_mlx.service.helpers import get_server_context_cap

    engine = _StubEngine(model_max=500_000, server_cap=196_608, config_shape="direct")
    assert get_server_context_cap(engine) == 196_608


def test_server_cap_resolves_via_engine_config_holder():
    from vllm_mlx.service.helpers import get_server_context_cap

    engine = _StubEngine(model_max=500_000, server_cap=196_608, config_shape="holder")
    assert get_server_context_cap(engine) == 196_608


def test_server_cap_is_none_when_flag_absent():
    """``0`` (flag absent) = no cap: the helper returns ``None`` and
    the model max stays the binding limit (pre-#33 behaviour)."""
    from vllm_mlx.service.helpers import get_server_context_cap

    engine = _StubEngine(model_max=500_000, server_cap=0)
    assert get_server_context_cap(engine) is None
    # No scheduler config at all (test stubs, MLLM engines) too.
    bare = _StubEngine(model_max=500_000)
    assert get_server_context_cap(bare) is None


# ---------------------------------------------------------------------------
# 4. Admission enforcement
# ---------------------------------------------------------------------------


def test_enforce_under_server_cap_is_silent():
    """prompt + max_tokens below the operator cap passes, including
    the exact boundary (199_000 against a 200_000 cap)."""
    from vllm_mlx.service.helpers import enforce_context_length

    engine = _StubEngine(model_max=500_000, server_cap=200_000)
    # 190_000 prompt + 9_000 completion = 199_000 <= 200_000.
    enforce_context_length(engine, 190_000, max_tokens=9_000)
    # Exact boundary: 180_000 + 20_000 = 200_000 <= cap passes.
    enforce_context_length(engine, 180_000, max_tokens=20_000)


def test_enforce_over_server_cap_raises_400():
    """prompt + max_tokens above the operator cap (but BELOW the model
    max) must be rejected at admission with the structured 400."""
    from fastapi import HTTPException

    from vllm_mlx.service.helpers import enforce_context_length

    engine = _StubEngine(model_max=500_000, server_cap=200_000)
    # 150_000 prompt + 60_000 completion = 210_000 > 200_000 cap,
    # still < 500_000 model max — pre-#33 this request would pass.
    with pytest.raises(HTTPException) as excinfo:
        enforce_context_length(engine, 150_000, max_tokens=60_000)
    assert excinfo.value.status_code == 400
    detail = excinfo.value.detail
    assert detail["error"]["code"] == "context_length_exceeded"
    # The operator cap is the binding limit, so the message must say
    # so — the client should not mistake this for a model limit.
    assert (
        "This server's maximum context length is 200000 tokens"
        in (detail["error"]["message"])
    )


def test_enforce_server_cap_equal_to_or_above_model_max_uses_model_max():
    """A cap at or above the model max must not tighten anything:
    the model max stays the binding limit and the message keeps the
    model-limit wording."""
    from fastapi import HTTPException

    from vllm_mlx.service.helpers import enforce_context_length

    # Cap == model max.
    engine = _StubEngine(model_max=50_000, server_cap=50_000)
    with pytest.raises(HTTPException) as excinfo:
        enforce_context_length(engine, 40_000, max_tokens=20_000)
    detail = excinfo.value.detail
    assert detail["error"]["code"] == "context_length_exceeded"
    assert (
        "This model's maximum context length is 50000 tokens"
        in (detail["error"]["message"])
    )
    # Cap above model max.
    engine = _StubEngine(model_max=50_000, server_cap=200_000)
    with pytest.raises(HTTPException) as excinfo:
        enforce_context_length(engine, 40_000, max_tokens=20_000)
    assert (
        "This model's maximum context length is 50000 tokens"
        in (excinfo.value.detail["error"]["message"])
    )


def test_enforce_no_flag_preserves_pre_33_behaviour():
    """Flag absent (``0``): the model max is the only gate. The
    210_000 request rejected by a 200_000 cap passes against a
    500_000 model max — i.e. current behaviour is preserved."""
    from fastapi import HTTPException

    from vllm_mlx.service.helpers import enforce_context_length

    engine = _StubEngine(model_max=500_000, server_cap=0)
    enforce_context_length(engine, 150_000, max_tokens=60_000)
    # ... and it still 400s past the model max.
    with pytest.raises(HTTPException):
        enforce_context_length(engine, 400_000, max_tokens=150_000)


def _messages_gate_engine(server_cap: int):
    """Deterministic 4-chars-per-token tokenizer stub mirroring
    ``test_context_overflow_400.py``; the engine exposes the operator
    cap through ``scheduler.config``."""
    model_max = 500_000

    class _Tokenizer:
        model_max_length = model_max
        bos_token = None

        def encode(self, text, add_special_tokens=True):  # noqa: ARG002
            return [0] * max(1, len(text) // 4)

        def apply_chat_template(
            self,
            messages,
            tools=None,
            add_generation_prompt=True,
            tokenize=False,
            **_kwargs,
        ):
            return "\n".join(m.get("content", "") for m in messages)

    class _Engine:
        is_mllm = False

        def __init__(self):
            self._tokenizer = _Tokenizer()
            self.scheduler = type(
                "Sched", (), {"config": _StubSchedulerConfig(server_cap)}
            )()

            class _Model:
                args = type("Args", (), {"max_position_embeddings": model_max})

            self._model = _Model()

        @property
        def tokenizer(self):
            return self._tokenizer

        def build_prompt(self, messages, tools=None, enable_thinking=None):
            return self._tokenizer.apply_chat_template(
                messages, tools=tools, enable_thinking=enable_thinking
            )

    return _Engine()


def test_messages_route_gate_surfaces_cap_400_and_passes_under_cap():
    """End-to-end-ish: the messages-route gate (the helper the chat /
    anthropic / responses routes all call) 400s when prompt +
    max_tokens crosses the operator cap and passes under it."""
    from fastapi import HTTPException

    from vllm_mlx.service.helpers import enforce_context_length_for_messages

    engine = _messages_gate_engine(200_000)
    # 80_000 chars / 4 = 20_000 prompt tokens; + 190_000 completion =
    # 210_000 > 200_000 cap (model max 500_000 is far above).
    with pytest.raises(HTTPException) as excinfo:
        enforce_context_length_for_messages(
            engine, [{"content": "x" * 80_000}], max_tokens=190_000
        )
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["error"]["code"] == "context_length_exceeded"
    # Under the cap the same prompt passes.
    assert (
        enforce_context_length_for_messages(
            engine, [{"content": "x" * 80_000}], max_tokens=10_000
        )
        is not None
    )
    # With the flag absent the same request passes (pre-#33).
    uncapped = _messages_gate_engine(0)
    assert (
        enforce_context_length_for_messages(
            uncapped, [{"content": "x" * 80_000}], max_tokens=190_000
        )
        is not None
    )


def test_repair_gate_respects_server_cap():
    """The strict-mode repair-retry gate must judge the repair prompt
    against the effective limit (model max MIN cap), not the bare
    model max — a request that fits the model max but not the cap
    still fails the fit check."""
    from vllm_mlx.service.helpers import repair_messages_fit_context

    engine = _messages_gate_engine(200_000)
    # 80_000 chars / 4 = 20_000 prompt tokens; + 190_000 completion =
    # 210_000 > 200_000 cap, < 500_000 model max.
    assert (
        repair_messages_fit_context(
            engine, [{"content": "x" * 80_000}], max_tokens=190_000
        )
        is False
    )
    # Under the cap it fits.
    assert (
        repair_messages_fit_context(
            engine, [{"content": "x" * 80_000}], max_tokens=10_000
        )
        is True
    )


# ---------------------------------------------------------------------------
# 6. Real BatchedEngine — the live serve shape (t981 critical guard)
#
# The 13 tests above use stub engine shapes that happen to match the
# config shapes ``get_server_context_cap`` probes. The production
# ``BatchedEngine`` (what ``get_engine()`` hands to the chat / anthropic
# / responses / completions routes) stores its config ONLY as the
# private ``_scheduler_config`` — none of the original three probes
# matched it, so the cap resolved ``None`` and the 400 gate never fired
# in a real ``qmlx serve``. These tests construct the REAL engine (cheap
# on a no-mlx host: ``BatchedEngine.__init__`` only sets attributes and
# runs the local ``is_mllm_model`` probe — no model load, no mlx import,
# no network) and pin the cap against that shape. A model name that does
# not resolve to a local config.json directory keeps ``is_mllm_model``
# on the substring path, so construction stays fast and deterministic.
# ---------------------------------------------------------------------------


_REAL_ENGINE_MODEL_NAME = "qmlx-test-model"


def _real_batched_engine(max_context_length: int | None):
    """Construct a REAL ``BatchedEngine`` (not a stub) with an optional
    ``SchedulerConfig.max_context_length`` cap. ``None`` = the flag
    absent (a bare engine, ``_scheduler_config`` is ``None``)."""
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.scheduler import SchedulerConfig

    if max_context_length is None:
        return BatchedEngine(_REAL_ENGINE_MODEL_NAME)
    return BatchedEngine(
        _REAL_ENGINE_MODEL_NAME,
        scheduler_config=SchedulerConfig(max_context_length=max_context_length),
    )


def test_real_batched_engine_exposes_scheduler_config_property():
    """The public ``scheduler_config`` seam returns the config the engine
    actually stores (``_scheduler_config``), so a cap set at construction
    is reachable. ``None`` when the engine was built without one."""
    capped = _real_batched_engine(8000)
    assert type(capped.scheduler_config).__name__ == "SchedulerConfig"
    assert capped.scheduler_config.max_context_length == 8000
    assert capped.scheduler_config is capped._scheduler_config
    assert _real_batched_engine(None).scheduler_config is None


def test_real_batched_engine_cap_resolves_and_gate_fires():
    """t981 critical guard: against the REAL ``BatchedEngine``, the cap
    resolves to the configured value and the 400 gate fires for a
    prompt + max_tokens over it.

    ``get_model_max_context`` falls through to the 4_194_304-token
    fallback for an unloaded engine, so a modest cap (8000) is the
    binding limit and the message keeps the operator-cap wording.
    """
    from fastapi import HTTPException

    from vllm_mlx.service.helpers import (
        enforce_context_length,
        get_server_context_cap,
    )

    engine = _real_batched_engine(8000)
    assert get_server_context_cap(engine) == 8000

    # 5000 prompt + 5000 completion = 10000 > 8000 cap (well below the
    # 4_194_304 fallback model max) — the gate must fire.
    with pytest.raises(HTTPException) as excinfo:
        enforce_context_length(engine, 5000, max_tokens=5000)
    assert excinfo.value.status_code == 400
    detail = excinfo.value.detail
    assert detail["error"]["code"] == "context_length_exceeded"
    # The operator cap (not the fallback model max) is the binding limit.
    assert (
        "This server's maximum context length is 8000 tokens"
        in (detail["error"]["message"])
    )

    # Under the cap the same-shape request passes: 5000 + 2000 = 7000
    # <= 8000.
    enforce_context_length(engine, 5000, max_tokens=2000)

    # Exact boundary: 5000 + 3000 = 8000 <= 8000 passes.
    enforce_context_length(engine, 5000, max_tokens=3000)


def test_real_batched_engine_no_cap_preserves_pre_33():
    """A real engine built without the flag resolves the cap to ``None``
    (``_scheduler_config`` is ``None``), so the model max stays the
    binding limit — pre-#33 behaviour, even on the live shape."""
    from vllm_mlx.service.helpers import get_server_context_cap

    assert get_server_context_cap(_real_batched_engine(None)) is None


def test_real_batched_engine_via_private_fallback():
    """Robustness: even an engine object that predates the public
    ``scheduler_config`` property (config reachable only via the private
    ``_scheduler_config``) still resolves the cap, matching the
    metrics-route probe. A bare object with only ``_scheduler_config``
    set exercises that fallback path."""
    from vllm_mlx.scheduler import SchedulerConfig
    from vllm_mlx.service.helpers import get_server_context_cap

    class _BareEngine:
        # No public ``scheduler_config`` attribute — only the private
        # one the live BatchedEngine stores the config under.
        _scheduler_config = SchedulerConfig(max_context_length=196_608)

    assert get_server_context_cap(_BareEngine()) == 196_608
