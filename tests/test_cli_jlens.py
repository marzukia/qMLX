# SPDX-License-Identifier: Apache-2.0
"""Tests for ``qmlx jlens`` (Jacobian-lens interpretability command).

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
    assert "qmlx jlens" in out
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


# --- R2 mirror routing (bug fix: jlens was bypassing the mirror) ------------
def test_prefetch_via_mirror_skips_local_paths(tmp_path) -> None:
    """Existing local dirs / arbitrary strings must not hit the mirror."""
    from unittest.mock import MagicMock

    fake = MagicMock(return_value=True)
    with patch("vllm_mlx._mirror.download_with_mirror_fallback", fake):
        # Existing local directory — bypass mirror entirely.
        assert jlens._prefetch_via_mirror(str(tmp_path)) is False
        # No slash → not an HF-style ``owner/name`` id.
        assert jlens._prefetch_via_mirror("not-a-repo-id") is False
        # Empty path — degenerate.
        assert jlens._prefetch_via_mirror("") is False
    fake.assert_not_called()


def test_looks_like_hf_repo_id_rejects_non_hf_shapes() -> None:
    """The tightened heuristic — HF ids are exactly ``owner/name``. Codex
    nit on PR #1045: without this, ``./missing/model``, ``/tmp/model``,
    ``foo/bar/baz`` and URLs would each incur a wasted
    ``download_with_mirror_fallback`` → ``model_info`` HF round-trip
    before ``mlx_lm.load`` had a chance to reject them locally."""
    # Positive cases — canonical HF ids.
    assert jlens._looks_like_hf_repo_id("mlx-community/Qwen3-1.7B-4bit")
    assert jlens._looks_like_hf_repo_id("meta-llama/Llama-3.1-8B")
    # Negative cases — everything else.
    assert not jlens._looks_like_hf_repo_id("")
    assert not jlens._looks_like_hf_repo_id("qwen3-1.7b")  # no slash
    assert not jlens._looks_like_hf_repo_id("foo/bar/baz")  # too many parts
    assert not jlens._looks_like_hf_repo_id("/tmp/model")  # absolute path
    assert not jlens._looks_like_hf_repo_id("./missing/model")  # relative path
    assert not jlens._looks_like_hf_repo_id("~/models/foo")  # home-relative
    assert not jlens._looks_like_hf_repo_id("https://hf.co/mlx-community/X")
    assert not jlens._looks_like_hf_repo_id("/only-name")  # empty owner
    assert not jlens._looks_like_hf_repo_id("only-owner/")  # empty name


def test_prefetch_via_mirror_prefers_mirror_for_hf_repos() -> None:
    """A ``mlx-community/Foo`` id must be handed to the mirror before HF."""
    from unittest.mock import MagicMock

    fake = MagicMock(return_value=True)
    with patch("vllm_mlx._mirror.download_with_mirror_fallback", fake):
        assert jlens._prefetch_via_mirror("mlx-community/Qwen3-1.7B-4bit") is True
    fake.assert_called_once_with("mlx-community/Qwen3-1.7B-4bit")


def test_prefetch_via_mirror_falls_back_when_mirror_misses() -> None:
    """A mirror miss (return False) must not fail the call — mlx_lm.load
    will complete the pull via HF. Mirrors observed returning False in
    production: catalog says ``not yet mirrored``, per-file 404, or
    catalog outage on a custom mirror."""
    from unittest.mock import MagicMock

    fake = MagicMock(return_value=False)
    with patch("vllm_mlx._mirror.download_with_mirror_fallback", fake):
        assert jlens._prefetch_via_mirror("mlx-community/UnmirroredModel") is False
    fake.assert_called_once_with("mlx-community/UnmirroredModel")


def test_prefetch_via_mirror_swallows_unexpected_exceptions() -> None:
    """A raised mirror error must NOT bubble up — the mirror is a UX
    optimization, not a correctness dependency. jlens still runs; the
    ensuing ``mlx_lm.load`` handles the pull via HF."""
    from unittest.mock import MagicMock

    fake = MagicMock(side_effect=RuntimeError("boom"))
    with patch("vllm_mlx._mirror.download_with_mirror_fallback", fake):
        # Must return False and not raise.
        assert jlens._prefetch_via_mirror("mlx-community/Qwen3-1.7B-4bit") is False


def _install_fake_mlx_lm(fake_load) -> object:
    """Patch ``sys.modules`` so ``from mlx_lm import load`` binds our fake.

    Returns a ModuleType we've registered as ``mlx_lm`` with a single
    ``load`` attribute pointing at ``fake_load``. Tests use this to
    intercept ``jlens._load_model``'s deferred import without touching
    the real mlx_lm — the real one loads native MLX runtime state that
    doesn't survive being clobbered by mid-test module replacement.
    """
    import types

    fake_mod = types.ModuleType("mlx_lm")
    fake_mod.load = fake_load
    return fake_mod


def test_load_model_prefetches_via_mirror_before_hf() -> None:
    """The concrete fix: ``_load_model`` must give the mirror first crack
    at the download before ``mlx_lm.load`` runs its own
    ``snapshot_download``. Otherwise a fresh install jlens invocation
    fetches directly from HF (the regression reported on 0.10.2)."""
    from unittest.mock import MagicMock, call

    order: list[str] = []

    def _mirror_side_effect(name):
        order.append(f"mirror:{name}")
        return True  # simulate the mirror hydrating the HF cache

    def _load_side_effect(name):
        order.append(f"mlx_lm.load:{name}")
        return (object(), object())

    fake_mirror = MagicMock(side_effect=_mirror_side_effect)
    fake_load = MagicMock(side_effect=_load_side_effect)
    fake_compat = MagicMock()

    with (
        patch("vllm_mlx._mirror.download_with_mirror_fallback", fake_mirror),
        patch("vllm_mlx._mlx_compat.install", fake_compat),
        patch.dict(sys.modules, {"mlx_lm": _install_fake_mlx_lm(fake_load)}),
    ):
        jlens._load_model("mlx-community/Qwen3-1.7B-4bit")

    # 1) Mirror was consulted, 2) before mlx_lm.load ran, 3) with the
    # SAME model id — no accidental alias re-resolution or mangling.
    assert order == [
        "mirror:mlx-community/Qwen3-1.7B-4bit",
        "mlx_lm.load:mlx-community/Qwen3-1.7B-4bit",
    ]
    fake_mirror.assert_has_calls([call("mlx-community/Qwen3-1.7B-4bit")])
    fake_load.assert_called_once_with("mlx-community/Qwen3-1.7B-4bit")


def test_load_model_still_runs_when_mirror_returns_404() -> None:
    """When the mirror can't serve the model (catalog says ``not mirrored``
    or per-file 404), ``_load_model`` must proceed to ``mlx_lm.load``,
    which will complete the pull via HuggingFace. This is the graceful-
    degradation contract."""
    from unittest.mock import MagicMock

    fake_mirror = MagicMock(return_value=False)  # simulate mirror miss
    fake_load = MagicMock(return_value=(object(), object()))
    fake_compat = MagicMock()

    with (
        patch("vllm_mlx._mirror.download_with_mirror_fallback", fake_mirror),
        patch("vllm_mlx._mlx_compat.install", fake_compat),
        patch.dict(sys.modules, {"mlx_lm": _install_fake_mlx_lm(fake_load)}),
    ):
        jlens._load_model("mlx-community/UnmirroredModel")

    fake_mirror.assert_called_once_with("mlx-community/UnmirroredModel")
    # mlx_lm.load STILL runs — it will drive its own snapshot_download.
    fake_load.assert_called_once_with("mlx-community/UnmirroredModel")


def test_load_model_skips_mirror_for_local_paths(tmp_path) -> None:
    """A user pointing jlens at a local directory (``qmlx jlens -m
    ~/models/foo ...``) must not incur a mirror round-trip."""
    from unittest.mock import MagicMock

    fake_mirror = MagicMock(return_value=True)
    fake_load = MagicMock(return_value=(object(), object()))
    fake_compat = MagicMock()

    with (
        patch("vllm_mlx._mirror.download_with_mirror_fallback", fake_mirror),
        patch("vllm_mlx._mlx_compat.install", fake_compat),
        patch.dict(sys.modules, {"mlx_lm": _install_fake_mlx_lm(fake_load)}),
    ):
        jlens._load_model(str(tmp_path))

    fake_mirror.assert_not_called()
    fake_load.assert_called_once_with(str(tmp_path))
