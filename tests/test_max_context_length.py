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
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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
