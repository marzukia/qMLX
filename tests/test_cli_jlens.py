# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx jlens`` (Jacobian-lens interpretability command).

These are fast, weight-free tests. The J-lens engine imports mlx lazily, so the
module imports and its pure helpers (architecture location, rendering, the
workspace heuristic, and the command's error/JSON paths) are all exercised
without loading a model. The heavy numerical path (JVP transport) is validated
end-to-end manually against cached models; here we lock the plumbing that a
refactor could silently break: architecture detection, unsupported-arch
handling, and the rendered/JSON output shape.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from unittest.mock import patch

import pytest

from vllm_mlx import jlens


# --- pure helpers -----------------------------------------------------------
def test_is_content_filters_padding_and_punctuation() -> None:
    assert jlens._is_content("France")
    assert jlens._is_content("Paris")
    assert not jlens._is_content("____")  # logit-lens padding artifact
    assert not jlens._is_content(",")
    assert not jlens._is_content("")
    assert not jlens._is_content("a")  # single char


def test_get_inner_locates_standard_decoder() -> None:
    """A Qwen3/Llama-style ``model.model`` holds the residual stream."""

    class Inner:
        layers = [object(), object()]
        embed_tokens = object()
        norm = object()

    class Model:
        model = Inner()

    inner = jlens._get_inner(Model())
    assert len(inner.layers) == 2
    assert hasattr(inner, "embed_tokens")


def test_get_inner_locates_nested_language_model() -> None:
    """Qwen3.5-VL-style layout nests the text model under language_model."""

    class Inner:
        layers = [object()]
        embed_tokens = object()
        norm = object()

    class LM:
        model = Inner()

    class Model:
        language_model = LM()

    assert jlens._get_inner(Model()).layers == Inner.layers


def test_get_inner_raises_on_unsupported_architecture() -> None:
    class Weird:
        something_else = 1

    with pytest.raises(jlens.UnsupportedArchitectureError):
        jlens._get_inner(Weird())


def test_get_unembed_prefers_lm_head_else_tied_embedding() -> None:
    # Real lm_head is a callable nn.Module instance, not a plain function
    # (a function class-attr would bind into a method on access).
    class _Head:
        def __call__(self, x):
            return x

    sentinel = _Head()

    class WithHead:
        lm_head = sentinel

    class Inner:
        class embed_tokens:  # noqa: N801
            @staticmethod
            def as_linear(x):
                return x

    assert jlens._get_unembed(WithHead(), Inner()) is sentinel

    class NoHead:
        pass

    assert jlens._get_unembed(NoHead(), Inner()) is Inner.embed_tokens.as_linear


# --- synthetic analysis result ---------------------------------------------
def _fake_result() -> dict:
    layers = list(range(0, 28, 3)) + [27]
    jl = {li: ["a", "the", ","] for li in layers}
    jl[12] = ["France", "Germany", "Spain"]
    jl[21] = ["France", "Rome", "Paris"]
    jl[24] = ["Paris", "France", "city"]
    jl[27] = ["Paris", "France", "巴黎"]
    ll = {li: ["____", "a", ","] for li in layers}
    ll[24] = ["Paris", "____", "France"]
    ranks = {li: 500 for li in layers}
    ranks[21] = 2
    ranks[24] = 0
    ranks[27] = 0
    return {
        "prompt": "The capital of France is",
        "completion": "Paris, the",
        "n_layers": 28,
        "layers": layers,
        "answer": "Paris",
        "answer_rank_by_layer": ranks,
        "jlens_first_layer": 21,
        "logit_lens_first_layer": 24,
        "jlens_by_layer": jl,
        "logit_lens_by_layer": ll,
        "transport": "jvp",
    }


def test_workspace_signal_in_range_and_rewards_lead() -> None:
    score, density = jlens._workspace_signal(_fake_result())
    assert 0.0 <= score <= 1.0
    assert 0.0 <= density <= 1.0


def test_render_text_has_all_sections() -> None:
    out = jlens.render_text(_fake_result(), "Qwen3-1.7B-4bit")
    assert "rapid-mlx jlens" in out
    assert "model continues → 'Paris, the'" in out
    assert "internal trajectory" in out
    assert "answer 'Paris'" in out
    assert "crystallizes at  L21/28" in out
    # J-lens locks 'Paris' at L21, logit lens at L24 → +3 layer lead
    assert "+3 layers" in out
    assert "workspace signal" in out


def test_render_text_survives_missing_completion() -> None:
    r = _fake_result()
    r["completion"] = ""
    out = jlens.render_text(r, "m")
    assert "model continues" not in out
    assert "answer 'Paris'" in out


def test_run_layer_handles_both_cache_signatures() -> None:
    """Some decoder blocks are layer(x, mask); others require an explicit
    cache arg (mlx_lm StableLM). _run_layer probes and adapts to either."""

    class CacheOptional:
        def __call__(self, x, mask=None, cache=None):
            return x + 1

    class CacheRequired:
        def __call__(self, x, mask, cache):  # no default → needs the arg
            return x + 10

    a = jlens.JLensAnalyzer.__new__(jlens.JLensAnalyzer)
    a._layer_mode = None
    assert a._run_layer(CacheOptional(), 0, "m") == 1
    assert a._layer_mode == "no_cache"

    b = jlens.JLensAnalyzer.__new__(jlens.JLensAnalyzer)
    b._layer_mode = None
    assert b._run_layer(CacheRequired(), 0, "m") == 10
    assert b._layer_mode == "cache"


def test_render_verbose_has_per_layer_table_and_rank_trajectory() -> None:
    out = jlens.render_verbose(_fake_result(), "Qwen3-1.7B-4bit")
    # includes the concise summary
    assert "internal trajectory" in out
    # plus the fuller per-layer readouts and the rank trajectory
    assert "per-layer readouts" in out
    assert "rank trajectory of answer 'Paris'" in out
    assert "L24:0" in out  # answer is top-1 at layer 24


# --- command wiring (mocked model, no weights) ------------------------------
def _make_args(**kw) -> argparse.Namespace:
    base = dict(
        prompt="The capital of France is", model="mlx-community/X", step=2, json=False
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_jlens_command_json_output() -> None:
    class FakeAnalyzer:
        def __init__(self, *a, **k):
            pass

        def analyze(self, prompt, step=2):
            r = _fake_result()
            r["prompt"] = prompt
            return r

    buf = io.StringIO()
    with (
        patch.object(jlens, "_load_model", return_value=(object(), object())),
        patch.object(jlens, "JLensAnalyzer", FakeAnalyzer),
        patch.object(sys, "stdout", buf),
    ):
        jlens.jlens_command(_make_args(json=True))
    payload = json.loads(buf.getvalue())
    assert payload["answer"] == "Paris"
    assert payload["model"] == "mlx-community/X"
    assert "workspace_signal" in payload


def test_jlens_command_unsupported_architecture_exits_2() -> None:
    def boom(_path):
        raise jlens.UnsupportedArchitectureError("linear attention")

    buf = io.StringIO()
    with (
        patch.object(jlens, "_load_model", boom),
        patch.object(sys, "stdout", buf),
        pytest.raises(SystemExit) as exc,
    ):
        jlens.jlens_command(_make_args())
    assert exc.value.code == 2
    assert "does not support this model architecture" in buf.getvalue()


def test_jlens_command_rejects_nonpositive_step_before_loading() -> None:
    from unittest.mock import MagicMock

    loader = MagicMock()
    buf = io.StringIO()
    with (
        patch.object(jlens, "_load_model", loader),
        patch.object(sys, "stdout", buf),
        pytest.raises(SystemExit) as exc,
    ):
        jlens.jlens_command(_make_args(step=0))
    assert exc.value.code == 2
    assert "--step must be a positive integer" in buf.getvalue()
    loader.assert_not_called()  # must fail before loading weights
