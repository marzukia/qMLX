# SPDX-License-Identifier: Apache-2.0
"""fixed_k in the speculative config + disable_auto_k semantics (#112).

Decision (option 2 of the issue): expose ``fixed_k`` in the
speculative config and KEEP ``disable_auto_k`` as the pre-0.9.13
chain-of-1 A/B bench mode — it pins K=1, NOT the configured depth
(the configured depth is silently dropped; pinning it is
``fixed_k``).

Pinned here:

1. ``fixed_k`` parses in the mtp speculative config (0 = park every
   round; the generator validates the upper bound against
   ``num_speculative_tokens`` at construction — #53).
2. ``SchedulerConfig.mtp_fixed_k`` -> ``_install_mtp_vendored`` ->
   the step-call kwarg; the config pin takes precedence over the
   #111 ``QMLX_MTP_FIXED_K`` env gate (which becomes the fallback).
3. Boot validation covers the config pin: ``0 <= fixed_k <= max_k``
   (0 = park every round) with an actionable error, not mid-stream.
4. The k_len ACTUALLY RUN under ``disable_auto_k`` +
   ``num_speculative_tokens=N`` (N=2): one mtp_forward per round —
   chain-of-1, not chain-of-2 (this issue is a step-counter catch,
   not a test catch).

No-MLX: numpy-backed stubs for the scheduler wiring; section 4 runs
only where real MLX is available (``pytest.importorskip``).
"""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


# ----------------------------------------------------------------------
# 1. Speculative-config parsing (no MLX needed)
# ---------------------------------------------------------------------------
def test_config_mtp_fixed_k_parses():
    from vllm_mlx.spec_decode.config import parse_speculative_config

    cfg = parse_speculative_config(
        '{"method":"mtp","num_speculative_tokens":2,"fixed_k":2}'
    )
    assert cfg.fixed_k == 2


def test_config_mtp_fixed_k_zero_is_valid():
    """fixed_k=0 (park every round) is a valid pin — the generator
    validates the upper bound against max_k at construction."""
    from vllm_mlx.spec_decode.config import parse_speculative_config

    cfg = parse_speculative_config('{"method":"mtp","fixed_k":0}')
    assert cfg.fixed_k == 0


def test_config_mtp_fixed_k_absent_is_none():
    from vllm_mlx.spec_decode.config import parse_speculative_config

    cfg = parse_speculative_config('{"method":"mtp"}')
    assert cfg.fixed_k is None


@pytest.mark.parametrize(
    "json_value",
    ["-1", "2.5", '"2"', "true"],
)
def test_config_mtp_fixed_k_bad_values_rejected(json_value):
    from vllm_mlx.spec_decode.config import (
        SpeculativeConfigError,
        parse_speculative_config,
    )

    with pytest.raises(SpeculativeConfigError, match=r"fixed_k"):
        parse_speculative_config(f'{{"method":"mtp","fixed_k":{json_value}}}')


def test_config_fixed_k_rejected_for_non_mtp_method():
    from vllm_mlx.spec_decode.config import (
        SpeculativeConfigError,
        parse_speculative_config,
    )

    with pytest.raises(SpeculativeConfigError, match=r"fixed_k"):
        parse_speculative_config('{"method":"suffix","fixed_k":2}')


# ----------------------------------------------------------------------
# 2. SchedulerConfig field
# ---------------------------------------------------------------------------
def test_scheduler_config_mtp_fixed_k_field():
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(mtp_fixed_k=2)
    assert cfg.mtp_fixed_k == 2
    assert SchedulerConfig().mtp_fixed_k is None


# ----------------------------------------------------------------------
# 3. Scheduler wiring (numpy-backed stubs, same pattern as #111)
# ---------------------------------------------------------------------------


class _AnyBase:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _AnyBase()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _AnyBase()


@pytest.fixture()
def _mlx_stubs(monkeypatch):
    saved = {k: v for k, v in sys.modules.items() if k == "mlx" or k.startswith("mlx.")}
    saved_ours = {
        k: v
        for k, v in sys.modules.items()
        if k
        in (
            "vllm_mlx",
            "vllm_mlx.scheduler",
            "vllm_mlx.spec_decode",
            "vllm_mlx.spec_decode.mtp",
            "vllm_mlx.spec_decode.mtp.generator",
        )
    }
    for k in list(saved_ours):
        del sys.modules[k]

    def stub(name):
        m = _StubModule(name)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, name, m)
        return m

    mx = stub("mlx")
    mx.core = stub("mlx.core")
    mx.nn = stub("mlx.nn")
    mx.core.uint32 = "uint32"
    mx.core.array = lambda *a, **k: np.asarray(a[0] if a else [])
    mx.core.inf = float("inf")
    mlm = stub("mlx_lm")
    for sub in (
        "generate",
        "sample_utils",
        "tokenizer_utils",
        "utils",
        "models",
        "quantize",
        "convert",
        "models.base",
        "models.cache",
        "models.gated_delta",
        "models.switch_layers",
    ):
        m = stub(f"mlx_lm.{sub}")
        setattr(mlm, sub.split(".")[-1], m)
    yield
    for k in [k for k in sys.modules if k == "mlx" or k.startswith("mlx.")]:
        del sys.modules[k]
    sys.modules.update(saved)
    for k in (
        "vllm_mlx",
        "vllm_mlx.scheduler",
        "vllm_mlx.spec_decode",
        "vllm_mlx.spec_decode.mtp",
        "vllm_mlx.spec_decode.mtp.generator",
    ):
        if k in saved_ours:
            sys.modules[k] = saved_ours[k]
        else:
            sys.modules.pop(k, None)


def _install_step(monkeypatch, caplog, env, max_k=3, fixed_k=None):
    from vllm_mlx import scheduler

    captured: dict = {}

    def _capture(*a, **kw):
        captured.update(kw)

        def _gen():
            yield 1, np.array(0.0), False

        return _gen()

    import vllm_mlx.spec_decode.mtp.generator as gen_mod

    monkeypatch.setattr(gen_mod, "mtp_generate_step", _capture)

    class _TokArr:
        def __init__(self, vals) -> None:
            self._v = np.asarray(vals)

        def __getitem__(self, i):
            return self._v[i]

        def astype(self, *a, **k):
            return self

    class _GB:
        def __init__(self) -> None:
            self.uids = [1]
            self.logits_processors = None
            self._next_tokens = _TokArr([7])
            self._next_logprobs = [np.array(0.0)]
            self.max_tokens = [80]
            self.tokens = [[]]
            self.prompt_cache = None

            def _step(self):
                return [0], [np.array(0.0)]

            self._step = _step.__get__(self)

    class _BG:
        _generation_batch = _GB()
        stop_tokens = None

    batch_gen = _BG()
    model = SimpleNamespace(mtp_forward=None, make_mtp_cache=None, mtp=None)

    for key in ("QMLX_MTP_FIXED_K", "QMLX_XTC_PROB", "QMLX_XTC_THR"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    with caplog.at_level(logging.INFO):
        assert (
            scheduler._install_mtp_vendored(
                batch_gen,
                model,
                max_k=max_k,
                fixed_k=fixed_k,
            )
            is True
        )
        batch_gen._generation_batch._step()
    return captured, caplog


def test_config_pin_reaches_step_call(_mlx_stubs, monkeypatch, caplog):
    captured, caplog = _install_step(monkeypatch, caplog, env={}, fixed_k=2)
    assert captured["fixed_k"] == 2
    assert "fixed depth pinned" in caplog.text
    assert "speculative config" in caplog.text


def test_config_pin_takes_precedence_over_env(_mlx_stubs, monkeypatch, caplog):
    """#112 config pin > #111 env gate (the env becomes the fallback
    when no pin is configured)."""
    captured, _ = _install_step(
        monkeypatch,
        caplog,
        env={"QMLX_MTP_FIXED_K": "3"},
        fixed_k=2,
    )
    assert captured["fixed_k"] == 2


def test_env_gate_still_works_as_fallback(_mlx_stubs, monkeypatch, caplog):
    captured, caplog = _install_step(monkeypatch, caplog, env={"QMLX_MTP_FIXED_K": "3"})
    assert captured["fixed_k"] == 3
    assert "QMLX_MTP_FIXED_K" in caplog.text


def test_no_pin_no_kwarg(_mlx_stubs, monkeypatch, caplog):
    captured, _ = _install_step(monkeypatch, caplog, env={})
    assert "fixed_k" not in captured


def test_config_pin_boot_validation(_mlx_stubs, monkeypatch, caplog):
    import pytest as _pt

    with _pt.raises(ValueError, match=r"mtp_fixed_k=4 must lie in \[0, max_k=3\]"):
        _install_step(monkeypatch, caplog, env={}, fixed_k=4)


def test_env_gate_boot_validation_still_names_env(_mlx_stubs, monkeypatch, caplog):
    import pytest as _pt

    with _pt.raises(ValueError, match=r"QMLX_MTP_FIXED_K=4 must lie in \[0, max_k=3\]"):
        _install_step(monkeypatch, caplog, env={"QMLX_MTP_FIXED_K": "4"})


def test_config_pin_zero_parks_every_round_at_boot(_mlx_stubs, monkeypatch, caplog):
    """fixed_k=0 (park every round) passes boot validation and reaches
    the step call — the generator's [0, max_k] contract (per #53) is
    the upper-bound gate; the runtime park branch is K=0."""
    captured, caplog = _install_step(monkeypatch, caplog, env={}, fixed_k=0)
    assert captured["fixed_k"] == 0
    assert "fixed depth pinned: mtp_fixed_k=0" in caplog.text


def test_env_gate_zero_passes_boot_validation(_mlx_stubs, monkeypatch, caplog):
    captured, caplog = _install_step(monkeypatch, caplog, env={"QMLX_MTP_FIXED_K": "0"})
    assert captured["fixed_k"] == 0
    assert "fixed depth pinned: QMLX_MTP_FIXED_K=0" in caplog.text


def test_config_pin_negative_still_rejected_at_boot(_mlx_stubs, monkeypatch, caplog):
    import pytest as _pt

    with _pt.raises(ValueError, match=r"mtp_fixed_k=-1 must lie in \[0, max_k=3\]"):
        _install_step(monkeypatch, caplog, env={}, fixed_k=-1)


# ----------------------------------------------------------------------
# 4. The k_len actually run under disable_auto_k + num_speculative_tokens
#    (real MLX only — this issue is a step-counter catch, not a test
#    catch)
# ---------------------------------------------------------------------------
class _CountingMockModel:
    """_MockedQwen35Model (test_mtp_spec_decode) + an mtp_forward call
    counter — one mtp_forward per chain position, so the call count
    IS the k_len actually run per round."""

    def __init__(
        self,
        backbone_outputs: list[int],
        mtp_outputs: list[int],
        vocab: int = 32,
        hidden_size: int = 8,
    ):
        self._backbone = list(backbone_outputs)
        self._mtp = list(mtp_outputs)
        self._backbone_cursor = 0
        self._mtp_cursor = 0
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.layers = []
        self.mtp_forward_calls = 0

    def _logits_for_positions(self, target_ids: list[int], batch: int):
        import mlx.core as mx

        out_rows = []
        for tid in target_ids:
            row = mx.zeros((batch, self.vocab))
            row = row + mx.where(
                mx.arange(self.vocab)[None, :] == tid,
                mx.array(50.0),
                mx.array(0.0),
            )
            out_rows.append(row)
        return mx.stack(out_rows, axis=1)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        import mlx.core as mx

        B, S = inputs.shape
        targets = []
        for _ in range(S):
            if self._backbone_cursor < len(self._backbone):
                targets.append(self._backbone[self._backbone_cursor])
                self._backbone_cursor += 1
            else:
                targets.append(0)
        logits = self._logits_for_positions(targets, B)
        hidden = mx.zeros((B, S, self.hidden_size))
        if return_hidden:
            return logits, hidden
        return logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        self.mtp_forward_calls += 1
        S = next_token_ids.shape[1]
        targets = []
        for _ in range(S):
            if self._mtp_cursor < len(self._mtp):
                targets.append(self._mtp[self._mtp_cursor])
                self._mtp_cursor += 1
            else:
                targets.append(0)
        return self._logits_for_positions(targets, next_token_ids.shape[0])

    def make_mtp_cache(self):
        return []


def test_disable_auto_k_pins_chain_of_1_not_configured_depth():
    """#112: ``num_speculative_tokens=2`` (-> max_k=2) +
    ``disable_auto_k=True`` runs K=1 EVERY round (one mtp_forward per
    round) — the configured depth is silently dropped. This is the
    step-counter catch the issue describes (the invalid A/B's
    ``attempts_total == rounds``)."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import mlx_lm.generate  # noqa: F401 — ensure module exists in sys.modules

    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    # Re-bind ``generation_stream`` to this thread's default stream —
    # a preceding test may have left it at a worker thread's stream
    # (same fix as test_mtp_spec_decode's fixture).
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )

    # 1 round: primary 7; draft 11; verify 11 (accept) + bonus 13.
    model = _CountingMockModel(backbone_outputs=[7, 11, 13], mtp_outputs=[11])
    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    for tok, _lp, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=3,
        max_k=2,  # num_speculative_tokens=2
        disable_auto_k=True,
    ):
        emitted.append((tok, from_draft))

    assert emitted == [(7, False), (11, True), (13, False)], (
        f"chain-of-1 stream wrong: {emitted}"
    )
    assert model.mtp_forward_calls == 1, (
        f"disable_auto_k + max_k=2 ran {model.mtp_forward_calls} mtp_forward "
        "calls — the configured depth (K=2) leaked into the bench "
        "mode; it must be chain-of-1 (1 call per round)."
    )


def test_fixed_k_two_runs_chain_of_2():
    """Positive control: ``fixed_k=2`` + ``max_k=2`` runs K=2 (two
    mtp_forward chain positions in the single round) — proves the
    counting actually distinguishes the depth."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import mlx_lm.generate  # noqa: F401 — ensure module exists in sys.modules

    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )

    # 1 round: primary 7; chain 11, 12; verify [11, 12] (both accept)
    # + bonus 13.
    model = _CountingMockModel(backbone_outputs=[7, 11, 12, 13], mtp_outputs=[11, 12])
    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    for tok, _lp, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=4,
        max_k=2,
        fixed_k=2,
    ):
        emitted.append((tok, from_draft))

    assert emitted == [(7, False), (11, True), (12, True), (13, False)], (
        f"chain-of-2 stream wrong: {emitted}"
    )
    assert model.mtp_forward_calls == 2, (
        f"fixed_k=2 ran {model.mtp_forward_calls} mtp_forward calls — "
        "expected 2 (one per chain position)."
    )
