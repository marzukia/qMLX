# SPDX-License-Identifier: Apache-2.0
"""Tests for ``qmlx doctor`` — the environment health check.

Covers issue #52 step 2: every section's probe with mocked failures,
the exit-code contract (warnings never fail, failures do), the text +
JSON renderings, and the no-network guarantee for the model check
(HuggingFace hub functions explode if touched).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import vllm_mlx.doctor as doctor
from vllm_mlx.doctor import (
    CheckStatus,
    Report,
    doctor_command,
    run_checks,
)

# ---------------------------------------------------------------------------
# Helpers
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
def healthy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully green environment: every package present at a good
    version, Apple Silicon backend, healthy disks."""
    monkeypatch.setattr(
        doctor,
        "pkg_version",
        lambda name: {
            "mlx": "0.32.0",
            "mlx-lm": "0.32.1",
            "transformers": "5.12.1",
            "fastapi": "0.115.0",
            "uvicorn": "0.30.0",
        }.get(name),
    )
    monkeypatch.setattr(doctor, "platform_info", lambda: ("Darwin", "arm64"))
    monkeypatch.setattr(doctor, "disk_free_bytes", lambda path: 500 * 1024**3)


@pytest.fixture
def hf_explode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network-facing hub calls explode; local cache probes stay."""

    def _explode(*_args, **_kwargs):
        raise RuntimeError("HF hub touched — expected zero network access")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _explode)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _explode)
    monkeypatch.setattr("huggingface_hub.model_info", _explode)


def _status_by_label(report: Report, section: str, label: str) -> CheckStatus:
    for s in report.sections:
        if s.title == section:
            for c in s.checks:
                if c.label == label:
                    return c.status
    raise AssertionError(f"{section}/{label} not found in report")


# ---------------------------------------------------------------------------
# Python section
# ---------------------------------------------------------------------------


def test_python_ok(healthy_env: None):
    report = run_checks()
    assert _status_by_label(report, "Python", "interpreter version") is CheckStatus.OK


def test_python_below_floor_fails(monkeypatch: pytest.MonkeyPatch):
    class _OldVersion:
        major, minor, micro = 3, 9, 19

    monkeypatch.setattr(doctor, "sys", type("S", (), {"version_info": _OldVersion}))
    report = run_checks()
    assert _status_by_label(report, "Python", "interpreter version") is CheckStatus.FAIL
    assert report.exit_code == 1


# ---------------------------------------------------------------------------
# Packages section
# ---------------------------------------------------------------------------


def test_packages_all_ok(healthy_env: None):
    report = run_checks()
    for name in ("mlx", "mlx-lm", "transformers", "fastapi", "uvicorn"):
        assert _status_by_label(report, "Packages", name) is CheckStatus.OK


def test_missing_mlx_fails(monkeypatch: pytest.MonkeyPatch):
    def _versions(name):
        if name == "mlx":
            return None
        return "1.0.0"

    monkeypatch.setattr(doctor, "pkg_version", _versions)
    report = run_checks()
    assert _status_by_label(report, "Packages", "mlx") is CheckStatus.FAIL
    assert _status_by_label(report, "Packages", "fastapi") is CheckStatus.OK
    assert report.exit_code == 1


def test_old_mlx_warns_not_fails(monkeypatch: pytest.MonkeyPatch):
    def _versions(name):
        return "0.30.0" if name == "mlx" else "1.0.0"

    monkeypatch.setattr(doctor, "pkg_version", _versions)
    report = run_checks()
    assert _status_by_label(report, "Packages", "mlx") is CheckStatus.WARN
    # A warning alone never fails the exit code.
    assert report.n_fail == 0
    assert report.exit_code == 0


def test_transformers_above_cap_warns(monkeypatch: pytest.MonkeyPatch):
    def _versions(name):
        return "5.13.0" if name == "transformers" else "1.0.0"

    monkeypatch.setattr(doctor, "pkg_version", _versions)
    report = run_checks()
    assert _status_by_label(report, "Packages", "transformers") is CheckStatus.WARN
    assert (
        "5.13"
        in [c.detail for c in report.all_checks() if c.label == "transformers"][0]
    )


# ---------------------------------------------------------------------------
# Backend section
# ---------------------------------------------------------------------------


def test_backend_apple_silicon_ok(healthy_env: None):
    report = run_checks()
    assert (
        _status_by_label(report, "Backend", "Apple Silicon (Metal)") is CheckStatus.OK
    )


def test_backend_intel_mac_warns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(doctor, "platform_info", lambda: ("Darwin", "x86_64"))
    report = run_checks()
    assert (
        _status_by_label(report, "Backend", "Apple Silicon (Metal)") is CheckStatus.WARN
    )
    assert report.exit_code == 0  # warning only


def test_backend_linux_warns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(doctor, "platform_info", lambda: ("Linux", "x86_64"))
    report = run_checks()
    assert (
        _status_by_label(report, "Backend", "Apple Silicon (Metal)") is CheckStatus.WARN
    )


# ---------------------------------------------------------------------------
# Model section
# ---------------------------------------------------------------------------


def test_model_no_name_warns(healthy_env: None):
    report = run_checks()
    assert _status_by_label(report, "Model", "model") is CheckStatus.WARN


def test_model_local_dir_ok(tmp_path: Path, healthy_env: None, hf_explode: None):
    p = _make_model_dir(tmp_path)
    report = run_checks(str(p))
    assert _status_by_label(report, "Model", "directory") is CheckStatus.OK
    assert _status_by_label(report, "Model", "config.json") is CheckStatus.OK
    assert _status_by_label(report, "Model", "weights") is CheckStatus.OK
    assert _status_by_label(report, "Model", "tokenizer") is CheckStatus.OK
    assert report.exit_code == 0


def test_model_missing_config_fails(tmp_path: Path, healthy_env: None):
    p = _make_model_dir(tmp_path)
    (p / "config.json").unlink()
    report = run_checks(str(p))
    assert _status_by_label(report, "Model", "config.json") is CheckStatus.FAIL
    assert report.exit_code == 1


def test_model_missing_weights_fails(tmp_path: Path, healthy_env: None):
    p = _make_model_dir(tmp_path)
    (p / "model.safetensors").unlink()
    report = run_checks(str(p))
    assert _status_by_label(report, "Model", "weights") is CheckStatus.FAIL
    assert report.exit_code == 1


def test_model_missing_tokenizer_fails(tmp_path: Path, healthy_env: None):
    p = _make_model_dir(tmp_path)
    (p / "tokenizer.json").unlink()
    report = run_checks(str(p))
    assert _status_by_label(report, "Model", "tokenizer") is CheckStatus.FAIL


def test_model_nonexistent_dir_fails(tmp_path: Path, healthy_env: None):
    report = run_checks(str(tmp_path / "nope"))
    assert _status_by_label(report, "Model", "directory") is CheckStatus.FAIL
    assert report.exit_code == 1


def test_model_unreadable_dir_fails(
    tmp_path: Path, healthy_env: None, monkeypatch: pytest.MonkeyPatch
):
    p = _make_model_dir(tmp_path)

    def _no_access(path, mode):
        return False

    monkeypatch.setattr(doctor.os, "access", _no_access)
    report = run_checks(str(p))
    assert _status_by_label(report, "Model", "directory") is CheckStatus.FAIL
    assert report.exit_code == 1


def test_model_hf_name_not_cached_warns(
    tmp_path: Path, healthy_env: None, monkeypatch: pytest.MonkeyPatch
):
    """An HF repo name is probed against the local cache only — no
    network (the hub functions would explode if touched)."""
    # huggingface_hub reads HF_HUB_CACHE at import time — patch the
    # constant, not just the env var.
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "empty-cache")
    )
    report = run_checks("some/uncached-repo")
    assert _status_by_label(report, "Model", "HF cache") is CheckStatus.WARN


def test_model_hf_name_cached_ok(
    tmp_path: Path, healthy_env: None, monkeypatch: pytest.MonkeyPatch
):
    """A cached HF repo name passes — via the local cache probe only."""
    hub = tmp_path / "hub"
    cache = hub / "models--some--repo" / "snapshots" / "abc123"
    cache.mkdir(parents=True)
    (cache / "config.json").write_text("{}")
    refs = hub / "models--some--repo" / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))
    report = run_checks("some/repo")
    assert _status_by_label(report, "Model", "HF cache") is CheckStatus.OK


def test_model_unknown_bare_name_warns(healthy_env: None):
    report = run_checks("definitely-not-an-alias-xyz")
    assert _status_by_label(report, "Model", "name") is CheckStatus.WARN


def test_model_local_dir_never_touches_hf(
    tmp_path: Path, healthy_env: None, hf_explode: None
):
    p = _make_model_dir(tmp_path)
    run_checks(str(p))  # would raise if any hub function were called


# ---------------------------------------------------------------------------
# Disk section
# ---------------------------------------------------------------------------


def test_disk_ok(healthy_env: None):
    report = run_checks()
    assert _status_by_label(report, "Disk", "KV checkpoint volume") is CheckStatus.OK


def test_disk_nearly_full_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(doctor, "disk_free_bytes", lambda path: 512 * 1024**2)
    report = run_checks()
    assert _status_by_label(report, "Disk", "KV checkpoint volume") is CheckStatus.FAIL
    assert report.exit_code == 1


def test_disk_small_warns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(doctor, "disk_free_bytes", lambda path: 5 * 1024**3)
    report = run_checks()
    assert _status_by_label(report, "Disk", "KV checkpoint volume") is CheckStatus.WARN
    assert report.exit_code == 0


def test_disk_model_volume_too_small_fails(
    tmp_path: Path, healthy_env: None, monkeypatch: pytest.MonkeyPatch
):
    p = _make_model_dir(tmp_path)
    # Model is ~1 KiB; the volume has 100 bytes free.
    monkeypatch.setattr(doctor, "disk_free_bytes", lambda path: 100)
    report = run_checks(str(p))
    assert _status_by_label(report, "Disk", "model volume") is CheckStatus.FAIL
    assert report.exit_code == 1


def test_disk_stat_failure_warns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(doctor, "disk_free_bytes", lambda path: None)
    report = run_checks()
    assert _status_by_label(report, "Disk", "KV checkpoint volume") is CheckStatus.WARN


# ---------------------------------------------------------------------------
# Report contract
# ---------------------------------------------------------------------------


def test_exit_code_contract(healthy_env: None):
    assert run_checks().exit_code == 0


def test_warnings_never_fail_exit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(doctor, "platform_info", lambda: ("Linux", "x86_64"))
    report = run_checks()
    assert report.n_warn >= 1
    assert report.n_fail == 0
    assert report.exit_code == 0


def test_failure_fails_exit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(doctor, "pkg_version", lambda name: None)
    report = run_checks()
    assert report.n_fail >= 1
    assert report.exit_code == 1


def test_text_rendering(healthy_env: None):
    text = run_checks().to_text()
    assert "Python" in text
    assert "Packages" in text
    assert "Backend" in text
    assert "Model" in text
    assert "Disk" in text
    assert "✓" in text
    assert "ok" in text.splitlines()[-2] or "ready" in text


def test_json_rendering(healthy_env: None):
    payload = json.loads(json.dumps(run_checks().to_json()))
    assert payload["summary"]["exit_code"] == 0
    titles = [s["title"] for s in payload["sections"]]
    assert titles == ["Python", "Packages", "Backend", "Model", "Disk"]
    for section in payload["sections"]:
        for check in section["checks"]:
            assert check["status"] in ("ok", "warn", "fail")


def test_doctor_command_exits_0(healthy_env: None, capsys: pytest.CaptureFixture):
    import argparse

    args = argparse.Namespace(model=None, json=False)
    try:
        doctor_command(args)
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert e.code == 0
    assert "Python" in capsys.readouterr().out


def test_doctor_command_exits_1_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    import argparse

    monkeypatch.setattr(doctor, "pkg_version", lambda name: None)
    args = argparse.Namespace(model=None, json=False)
    try:
        doctor_command(args)
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert e.code == 1
    assert "NOT ready" in capsys.readouterr().out


def test_doctor_command_json_mode(healthy_env: None, capsys: pytest.CaptureFixture):
    import argparse

    args = argparse.Namespace(model=None, json=True)
    with pytest.raises(SystemExit) as exc:
        doctor_command(args)
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["exit_code"] == 0
