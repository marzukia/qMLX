# SPDX-License-Identifier: Apache-2.0
"""Tests for local model directories — offline-capable install + serve.

Covers issue #52 step 1 (``qmlx pull <local-dir>`` registration) and
step 3 (``qmlx serve <local-path>`` with zero HuggingFace access):

* ``validate_local_model_dir`` — fixture model dirs, each missing piece
  reported.
* ``register_local_model`` / ``resolve_local_model`` — the name
  registry (symlink default, ``--copy`` variant, idempotent re-
  register, path-escape rejection).
* ``preflight_local_model`` — the serve-path gate (clean report +
  exit 1 on a broken dir).
* ``set_hf_offline`` — the zero-network guarantee: after the call, an
  uncached ``snapshot_download`` must raise (offline mode pinned in
  code, not in an env var the user sets).
* The ``main()`` dispatch — a registered local name resolves to its
  path; an unknown bare name still fails fast.

The "no network" tests make HuggingFace *explode if touched*: the
network-facing hub functions are patched to raise ``RuntimeError`` so
any accidental call fails the test loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_mlx import cli
from vllm_mlx.local_model import (
    LocalModelValidationError,
    is_local_model_path,
    list_local_models,
    models_dir,
    preflight_local_model,
    register_local_model,
    remove_local_model,
    resolve_local_model,
    set_hf_offline,
    validate_local_model_dir,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model_dir(root: Path, *, name: str = "fixture-model") -> Path:
    """A minimal loadable mlx-lm snapshot: config + one weight shard +
    a tokenizer file."""
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
    """Point the name registry at a temp dir (isolate from ~/.cache)."""
    root = tmp_path / "registry"
    root.mkdir()
    monkeypatch.setenv("QMLX_MODELS_DIR", str(root))
    return root


@pytest.fixture
def hf_explode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every network-facing HuggingFace hub call explode.

    ``try_to_load_from_cache`` is deliberately NOT patched — it is a
    local cache probe (no network) and the code under test may use it.
    """

    def _explode(*_args, **_kwargs):
        raise RuntimeError("HF hub touched — expected zero network access")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _explode)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _explode)
    monkeypatch.setattr("huggingface_hub.model_info", _explode)


# ---------------------------------------------------------------------------
# validate_local_model_dir
# ---------------------------------------------------------------------------


def test_validate_ok(model_dir: Path):
    report = validate_local_model_dir(model_dir)
    assert report.ok
    assert report.is_dir
    assert report.readable
    assert report.config_ok
    assert report.model_type == "qwen3"
    assert report.weight_files == ["model.safetensors"]
    assert report.tokenizer_file == "tokenizer.json"
    assert report.size_bytes > 0


def test_validate_missing(tmp_path: Path):
    report = validate_local_model_dir(tmp_path / "nope")
    assert not report.ok
    assert any("not found" in p for p in report.problems)


def test_validate_not_a_dir(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    report = validate_local_model_dir(f)
    assert not report.ok
    assert any("not a directory" in p for p in report.problems)


def test_validate_missing_config(tmp_path: Path):
    p = _make_model_dir(tmp_path)
    (p / "config.json").unlink()
    report = validate_local_model_dir(p)
    assert not report.ok
    assert any("config.json" in x for x in report.problems)


def test_validate_unparseable_config(tmp_path: Path):
    p = _make_model_dir(tmp_path)
    (p / "config.json").write_text("{not json")
    report = validate_local_model_dir(p)
    assert not report.ok
    assert any("unreadable" in x for x in report.problems)


def test_validate_missing_weights(tmp_path: Path):
    p = _make_model_dir(tmp_path)
    (p / "model.safetensors").unlink()
    report = validate_local_model_dir(p)
    assert not report.ok
    assert any("weight" in x.lower() for x in report.problems)


def test_validate_missing_tokenizer(tmp_path: Path):
    p = _make_model_dir(tmp_path)
    (p / "tokenizer.json").unlink()
    report = validate_local_model_dir(p)
    assert not report.ok
    assert any("tokenizer" in x.lower() for x in report.problems)


def test_validate_accepts_tokenizer_model_layout(tmp_path: Path):
    p = _make_model_dir(tmp_path)
    (p / "tokenizer.json").unlink()
    (p / "tokenizer.model").write_bytes(b"\x00" * 8)
    report = validate_local_model_dir(p)
    assert report.ok
    assert report.tokenizer_file == "tokenizer.model"


def test_validate_multiple_shards(tmp_path: Path):
    p = _make_model_dir(tmp_path)
    (p / "model-00001.safetensors").write_bytes(b"\x00" * 512)
    report = validate_local_model_dir(p)
    assert report.ok
    assert len(report.weight_files) == 2


# ---------------------------------------------------------------------------
# Name registry
# ---------------------------------------------------------------------------


def test_register_and_resolve(model_dir: Path, models_root: Path):
    dest = register_local_model(model_dir)
    assert dest.is_symlink()
    assert dest.resolve() == model_dir.resolve()
    assert resolve_local_model("fixture-model") == dest
    assert list_local_models() == {"fixture-model": dest}


def test_register_custom_name_and_copy(model_dir: Path, models_root: Path):
    dest = register_local_model(model_dir, name="my-qwen", copy=True)
    assert not dest.is_symlink()
    assert (dest / "config.json").is_file()
    assert resolve_local_model("my-qwen") == dest


def test_register_replaces_existing(model_dir: Path, models_root: Path):
    register_local_model(model_dir)
    other = _make_model_dir(model_dir.parent, name="other-model")
    dest = register_local_model(other, name="fixture-model")
    assert dest.resolve() == other.resolve()
    assert resolve_local_model("fixture-model") == dest


def test_register_invalid_dir_raises(model_dir: Path, models_root: Path):
    bad = model_dir.parent / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("{}")
    with pytest.raises(LocalModelValidationError):
        register_local_model(bad)


def test_remove(model_dir: Path, models_root: Path):
    register_local_model(model_dir)
    assert remove_local_model("fixture-model")
    assert resolve_local_model("fixture-model") is None
    assert remove_local_model("fixture-model") is False


def test_resolve_rejects_path_escaped_names(models_root: Path):
    for name in ("../elsewhere", "a/b", ".", "..", "-x", ""):
        assert resolve_local_model(name) is None


def test_resolve_nonexistent_name_returns_none(models_root: Path):
    assert resolve_local_model("never-registered") is None


def test_models_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QMLX_MODELS_DIR", str(tmp_path / "custom"))
    assert models_dir() == tmp_path / "custom"


# ---------------------------------------------------------------------------
# Serve-path gate (preflight + offline pin)
# ---------------------------------------------------------------------------


def test_preflight_ok_prints_confirmation(
    model_dir: Path, capsys: pytest.CaptureFixture
):
    preflight_local_model(str(model_dir))
    out = capsys.readouterr().out
    assert "offline" in out
    assert "no HuggingFace hub access" in out


def test_preflight_bad_dir_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture):
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(SystemExit) as exc:
        preflight_local_model(str(bad))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "not a loadable model directory" in out
    # Every missing piece is named.
    assert "config.json" in out
    assert "weight" in out.lower()
    assert "tokenizer" in out.lower()


def test_preflight_never_touches_hf(model_dir: Path, hf_explode: None):
    # The validation is pure filesystem — the patched hub functions
    # would explode if any of them were called.
    preflight_local_model(str(model_dir))


def test_set_hf_offline_pins_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    set_hf_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    # Idempotent.
    set_hf_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_offline_pin_makes_hf_explode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The zero-network guarantee, proven: with the pin in place, an
    uncached ``snapshot_download`` raises instead of hitting the hub.

    The exception type varies across huggingface_hub versions (0.28
    raises ``LocalEntryNotFoundError`` with an "outgoing traffic has
    been disabled" message; newer ones raise ``OfflineModeIsEnabled``
    directly) — the contract is "it raises, offline", so accept the
    family.
    """
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    set_hf_offline()
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
    )

    with pytest.raises(
        (OfflineModeIsEnabled, LocalEntryNotFoundError, RepositoryNotFoundError)
    ) as exc:
        snapshot_download("some/uncached")
    # The 0.28 message names the disabled traffic explicitly — pin the
    # contract on it when that is the shape we got.
    if isinstance(exc.value, LocalEntryNotFoundError):
        assert "outgoing traffic has been disabled" in str(exc.value)


def test_is_local_model_path():
    import vllm_mlx

    assert is_local_model_path(os.path.dirname(vllm_mlx.__file__))
    assert not is_local_model_path("/definitely/not/a/real/dir")


# ---------------------------------------------------------------------------
# main() dispatch — registered names resolve, unknown names fail fast
# ---------------------------------------------------------------------------


def _run_main(argv: list[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", argv)
    with patch.object(cli, "serve_command") as serve_mock:
        cli.main()
    return serve_mock


def test_main_resolves_registered_local_name(
    model_dir: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
):
    register_local_model(model_dir)
    serve_mock = _run_main(["qmlx", "serve", "fixture-model"], monkeypatch)
    serve_mock.assert_called_once()
    args = serve_mock.call_args.args[0]
    # Resolves to the registry entry (a symlink that follows to the
    # model dir) — serving through the symlink is the contract.
    assert args.model == str(models_root / "fixture-model")
    assert Path(args.model).resolve() == model_dir.resolve()


def test_main_broken_registry_link_exits_1(
    model_dir: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
):
    dest = register_local_model(model_dir)
    dest.unlink()  # break the symlink (target moved / vanished)
    with pytest.raises(SystemExit) as exc:
        _run_main(["qmlx", "serve", "fixture-model"], monkeypatch)
    assert exc.value.code == 1


def test_main_passes_direct_local_path_through(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    serve_mock = _run_main(["qmlx", "serve", str(model_dir)], monkeypatch)
    serve_mock.assert_called_once()
    args = serve_mock.call_args.args[0]
    assert args.model == str(model_dir)


def test_main_unknown_bare_name_exits_1(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(SystemExit) as exc:
        _run_main(["qmlx", "serve", "definitely-not-a-model-xyz"], monkeypatch)
    assert exc.value.code == 1


def test_main_alias_still_resolves(monkeypatch: pytest.MonkeyPatch):
    serve_mock = _run_main(["qmlx", "serve", "qwen3.5-4b-4bit"], monkeypatch)
    serve_mock.assert_called_once()
    args = serve_mock.call_args.args[0]
    assert "/" in args.model  # resolved to an HF path


def test_serve_local_path_sets_offline_pin(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """End-to-end resolution: serve_command with a local path validates
    the dir, pins the process offline, and never touches the hub.

    Runs the real serve_command up to ``_ensure_model_downloaded``
    (the first call after the local-path block) and stops there via a
    sentinel — the heavy server import never runs in the test.
    """
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)

    class _StopAfterLocalPreflightError(Exception):
        pass

    def _stop(*_a, **_k):
        raise _StopAfterLocalPreflightError()

    args = argparse.Namespace()
    args.model = str(model_dir)
    args.max_tokens = None
    args.watchdog_ppid = None

    with (
        patch.object(cli, "_ensure_model_downloaded", side_effect=_stop),
        patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available", return_value=False
        ),
        pytest.raises(_StopAfterLocalPreflightError),
    ):
        cli.serve_command(args)

    out = capsys.readouterr().out
    assert "offline" in out
    assert "no HuggingFace hub access" in out
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_serve_local_path_bad_dir_exits_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """A broken local dir fails at the preflight (exit 1) — never
    reaches the model load, never touches the hub."""
    bad = tmp_path / "bad"
    bad.mkdir()

    args = argparse.Namespace()
    args.model = str(bad)
    args.max_tokens = None
    args.watchdog_ppid = None

    with (
        patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available", return_value=False
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.serve_command(args)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "not a loadable model directory" in out
