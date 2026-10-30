# SPDX-License-Identifier: Apache-2.0
"""Tests for the offline-capable install path — ``qmlx pull <local-dir>``.

Covers issue #52 step 1: a local model directory is validated and
registered (symlink by default, ``--copy`` for a real copy), a broken
dir fails with exit 1, and the whole path never touches HuggingFace
(the hub functions explode if called).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_mlx import cli
from vllm_mlx.local_model import (
    list_local_models,
    resolve_local_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model_dir(root: Path, *, name: str = "fixture-model") -> Path:
    p = root / name
    p.mkdir(parents=True)
    (p / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    )
    (p / "model.safetensors").write_bytes(b"\x00" * 1024)
    (p / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    return p


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return _make_model_dir(tmp_path)


@pytest.fixture
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "registry"
    root.mkdir()
    monkeypatch.setenv("QMLX_MODELS_DIR", str(root))
    return root


@pytest.fixture
def hf_explode(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*_args, **_kwargs):
        raise RuntimeError("HF hub touched — expected zero network access")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _explode)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _explode)
    monkeypatch.setattr("huggingface_hub.model_info", _explode)


def _args(model: str, **kwargs):
    import argparse

    a = argparse.Namespace(model=model)
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


# ---------------------------------------------------------------------------
# pull <local-dir> — registration
# ---------------------------------------------------------------------------


def test_pull_local_dir_registers_symlink(
    model_dir: Path, models_root: Path, capsys: pytest.CaptureFixture
):
    cli.pull_command(_args(str(model_dir), name=None, copy=False))
    out = capsys.readouterr().out
    assert "Linked" in out
    dest = resolve_local_model("fixture-model")
    assert dest is not None
    assert dest.is_symlink()
    assert dest.resolve() == model_dir.resolve()
    assert "qmlx serve fixture-model" in out


def test_pull_local_dir_custom_name(
    model_dir: Path, models_root: Path, capsys: pytest.CaptureFixture
):
    cli.pull_command(_args(str(model_dir), name="my-qwen", copy=False))
    dest = resolve_local_model("my-qwen")
    assert dest is not None
    assert dest.resolve() == model_dir.resolve()


def test_pull_local_dir_copy(
    model_dir: Path, models_root: Path, capsys: pytest.CaptureFixture
):
    cli.pull_command(_args(str(model_dir), name=None, copy=True))
    dest = resolve_local_model("fixture-model")
    assert dest is not None
    assert not dest.is_symlink()
    assert (dest / "config.json").is_file()
    # The copy is independent of the source.
    (model_dir / "config.json").unlink()
    assert (dest / "config.json").is_file()


def test_pull_local_dir_idempotent(
    model_dir: Path, models_root: Path, capsys: pytest.CaptureFixture
):
    cli.pull_command(_args(str(model_dir), name=None, copy=False))
    cli.pull_command(_args(str(model_dir), name=None, copy=False))
    assert len(list_local_models()) == 1


def test_pull_local_dir_never_touches_hf(
    model_dir: Path, models_root: Path, hf_explode: None
):
    cli.pull_command(_args(str(model_dir), name=None, copy=False))


# ---------------------------------------------------------------------------
# pull <local-dir> — failure handling
# ---------------------------------------------------------------------------


def test_pull_broken_dir_exits_1(
    tmp_path: Path, models_root: Path, capsys: pytest.CaptureFixture
):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        cli.pull_command(_args(str(bad), name=None, copy=False))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "not a loadable model directory" in out
    # Nothing was registered.
    assert list_local_models() == {}


def test_pull_nonexistent_dir_exits_1(
    tmp_path: Path, models_root: Path, capsys: pytest.CaptureFixture
):
    with pytest.raises(SystemExit) as exc:
        cli.pull_command(_args(str(tmp_path / "nope"), name=None, copy=False))
    assert exc.value.code == 1


def test_pull_invalid_name_exits_1(
    model_dir: Path, models_root: Path, capsys: pytest.CaptureFixture
):
    with pytest.raises(SystemExit) as exc:
        cli.pull_command(_args(str(model_dir), name="a/b", copy=False))
    assert exc.value.code == 1
    assert list_local_models() == {}


# ---------------------------------------------------------------------------
# main() dispatch — pull routes local dirs to the local branch
# ---------------------------------------------------------------------------


def test_main_pull_local_dir(
    model_dir: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
):
    with patch.object(cli, "pull_command") as pull_mock:
        monkeypatch.setattr(sys, "argv", ["qmlx", "pull", str(model_dir)])
        cli.main()
    pull_mock.assert_called_once()
    args = pull_mock.call_args.args[0]
    assert args.model == str(model_dir)


def test_main_pull_hf_repo_still_routes_to_pull(
    models_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """A bare HF repo name (with a slash, not a path) still goes to
    the normal pull path — the local branch must not swallow it."""
    with patch.object(cli, "pull_command") as pull_mock:
        monkeypatch.setattr(sys, "argv", ["qmlx", "pull", "some/repo"])
        cli.main()
    pull_mock.assert_called_once()
    args = pull_mock.call_args.args[0]
    assert args.model == "some/repo"
