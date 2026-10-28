# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #477 — MTP injection on VLM/hybrid models.

Two surfaces are pinned here:

1. ``patches.qwen3_next_mtp._looks_like_vlm_wrapper`` — VLM checkpoints
   nest the LLM config under ``text_config`` and expose the inner LLM as
   ``model.language_model``. The outer ``model.args`` lacks LLM fields.
   ``inject_mtp_support`` must bail out cleanly in this shape rather than
   patch the outer class and crash on the next forward (codex round-1
   P1 on #477 — wrapper methods reference ``self.model.embed_tokens`` /
   ``self.lm_head`` that don't exist on the outer VLM).

The old ``scheduler._install_mtp`` BatchGenerator monkey-patch has been
removed. Runtime MTP now goes through the common speculative-config path
and ``_install_mtp_vendored``.
"""

from __future__ import annotations

from types import SimpleNamespace

# ----------------------------------------------------------------------
# _looks_like_vlm_wrapper — gate for VLM detection in inject_mtp_support
# ----------------------------------------------------------------------


def _llm_args_ns(**overrides):
    """SimpleNamespace shaped like a populated ``ModelArgs``."""
    defaults = {
        "hidden_size": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "full_attention_interval": 4,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "tie_word_embeddings": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_looks_like_vlm_wrapper_false_for_text_only_model():
    """Text-only path: ``model.args.hidden_size`` exists → not a VLM,
    inject_mtp_support proceeds normally."""
    from vllm_mlx.patches.qwen3_next_mtp import _looks_like_vlm_wrapper

    model = SimpleNamespace(args=_llm_args_ns(hidden_size=4096))
    assert _looks_like_vlm_wrapper(model) is False


def test_looks_like_vlm_wrapper_true_for_vlm_with_language_model():
    """VLM checkpoint: outer args lacks hidden_size AND
    model.language_model is present → bail out so we don't deferred-crash
    on the next forward. Pins codex round-1 P1 on issue #477."""
    from vllm_mlx.patches.qwen3_next_mtp import _looks_like_vlm_wrapper

    vlm_outer = SimpleNamespace(
        text_config={"hidden_size": 3584},
        vision_config={"hidden_size": 1280},
    )
    model = SimpleNamespace(
        args=vlm_outer,
        language_model=SimpleNamespace(args=_llm_args_ns(hidden_size=3584)),
    )
    assert _looks_like_vlm_wrapper(model) is True


def test_looks_like_vlm_wrapper_false_when_language_model_is_none():
    """Defensive — ``language_model`` attr exists but is None (e.g.
    text-only branch of a multimodal class). Not a usable VLM; let the
    "no fallback available" warning path fire instead of pretending it's
    a VLM wrapper."""
    from vllm_mlx.patches.qwen3_next_mtp import _looks_like_vlm_wrapper

    model = SimpleNamespace(args=SimpleNamespace(), language_model=None)
    assert _looks_like_vlm_wrapper(model) is False


def test_looks_like_vlm_wrapper_false_when_args_already_has_hidden_size():
    """Even if ``language_model`` is somehow attached to a text-only
    model, the populated ``model.args.hidden_size`` short-circuits the
    check (text-only path always wins)."""
    from vllm_mlx.patches.qwen3_next_mtp import _looks_like_vlm_wrapper

    model = SimpleNamespace(
        args=_llm_args_ns(hidden_size=2048),
        language_model=SimpleNamespace(args=_llm_args_ns(hidden_size=1024)),
    )
    assert _looks_like_vlm_wrapper(model) is False


# ----------------------------------------------------------------------
# _resolve_model_weights_dir / _load_mtp_weights_from_repo — local
# checkpoint support (dogfood 2026-10-28: a self-contained MTP
# checkpoint served from a local directory must not be rejected as
# "without mtp_sidecar or model_repo" — its own safetensors carry the
# mtp.* keys)
# ----------------------------------------------------------------------


def test_resolve_model_weights_dir_local_directory(tmp_path):
    """A local checkpoint directory is used in place (no download)."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _resolve_model_weights_dir,
    )

    assert _resolve_model_weights_dir(str(tmp_path)) == str(tmp_path)


def test_resolve_model_weights_dir_local_safetensors_file(tmp_path):
    """A direct *.safetensors file resolves to its parent directory."""
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _resolve_model_weights_dir,
    )

    f = tmp_path / "model.safetensors"
    f.write_bytes(b"")
    assert _resolve_model_weights_dir(str(f)) == str(tmp_path)


def test_resolve_model_weights_dir_hf_repo_download(monkeypatch, tmp_path):
    """Non-path references still go through snapshot_download."""
    import huggingface_hub

    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _resolve_model_weights_dir,
    )

    calls = {}

    def _fake_snapshot_download(repo_id, allow_patterns=None):
        calls["repo_id"] = repo_id
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)
    assert _resolve_model_weights_dir("some/repo") == str(tmp_path)
    assert calls["repo_id"] == "some/repo"


def test_resolve_model_weights_dir_unresolvable_repo(monkeypatch):
    """A repo id that fails to download returns None (soft failure)."""
    import huggingface_hub

    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _resolve_model_weights_dir,
    )

    def _boom(repo_id, allow_patterns=None):
        raise RuntimeError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
    assert _resolve_model_weights_dir("some/repo") is None


def test_load_mtp_weights_from_repo_local_dir_embedded_mtp(tmp_path):
    """Self-contained MTP checkpoint: mtp.* keys in the local
    directory's safetensors are extracted (prefixes stripped) with no
    sidecar and no HF round-trip — the exact dogfood case that used
    to hard-reject at boot."""
    import mlx.core as mx

    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _load_mtp_weights_from_repo,
    )

    weights = {
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 8)),
        "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((4, 8)),
        "mtp.layers.0.self_attn.o_proj.weight": mx.zeros((8, 4)),
        "mtp.fc.weight": mx.zeros((8, 16)),
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    loaded = _load_mtp_weights_from_repo(str(tmp_path))
    assert loaded is not None
    assert "layers.0.self_attn.q_proj.weight" in loaded
    assert "layers.0.self_attn.o_proj.weight" in loaded
    assert "fc.weight" in loaded
    # backbone (non-mtp) keys must not leak into the MTP weight set
    assert not any(k.startswith("model.") for k in loaded)


def test_load_mtp_weights_from_repo_local_dir_without_mtp_keys(tmp_path):
    """A local checkpoint without mtp.* keys returns None — the
    injector then reports the accurate 'no MTP weight tensors found'
    refusal instead of a download error."""
    import mlx.core as mx

    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        _load_mtp_weights_from_repo,
    )

    weights = {
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 8)),
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    assert _load_mtp_weights_from_repo(str(tmp_path)) is None
