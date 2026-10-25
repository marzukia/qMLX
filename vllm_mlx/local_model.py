# SPDX-License-Identifier: Apache-2.0
"""Offline-capable local model directories (issue #52 — 3-step onboarding).

qMLX's default onboarding pulls models from the HuggingFace hub. This
module is the zero-network alternative:

* :func:`validate_local_model_dir` — probe a directory and report which
  of the pieces mlx-lm needs (``config.json``, ``model*.safetensors``
  weights, a tokenizer file) are present.
* :func:`register_local_model` / :func:`resolve_local_model` — a tiny
  name registry under ``~/.cache/qmlx/models/`` (symlinks by default,
  real copies with ``copy=True``) so ``qmlx pull /path/to/model``
  produces a name the user can serve later: ``qmlx serve <name>``.
* :func:`preflight_local_model` — the serve-path gate: validate the
  directory up front and fail with a readable report instead of a deep
  loader traceback.
* :func:`set_hf_offline` — pin the process to offline mode so a
  local-path serve resolves config + weights from the local dir with
  **zero** HuggingFace hub access. The guarantee is code, not an
  env-var the user has to remember: no fallback to the hub on any
  incidental lookup.

The module is self-contained (stdlib only, no qmlx imports) so it stays
cheap to import from ``cli.main()`` on every invocation — same design
contract as ``vllm_mlx._download_gate``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# A loadable mlx-lm snapshot needs: config.json, at least one
# model*.safetensors shard, and at least one tokenizer file. The
# tokenizer file can be any of the three common layouts — mlx-community
# repos ship ``tokenizer.json`` (+ ``tokenizer_config.json``), some
# convert outputs ship ``tokenizer.model`` (Rust tokenizers).
WEIGHT_GLOB = "model*.safetensors"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")


@dataclass
class LocalModelReport:
    """Validation result for one directory.

    ``problems`` is empty iff the directory is loadable. Each entry is a
    one-line, user-readable description of a missing/broken piece.
    """

    path: Path
    is_dir: bool = False
    readable: bool = False
    config_ok: bool = False
    model_type: str | None = None
    weight_files: list[str] = field(default_factory=list)
    tokenizer_file: str | None = None
    size_bytes: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _size_of(path: Path) -> int:
    try:
        return sum(
            os.path.getsize(os.path.join(root, f))
            for root, _dirs, files in os.walk(path)
            for f in files
        )
    except OSError:
        return 0


def validate_local_model_dir(path: Path | str) -> LocalModelReport:
    """Probe ``path`` and report which loadable-model pieces are present.

    Never raises for filesystem conditions — every problem is recorded
    in ``report.problems`` so callers can render one readable block.
    """
    p = Path(path).expanduser()
    report = LocalModelReport(path=p)

    if not p.exists():
        report.problems.append(f"not found: {p}")
        return report
    report.is_dir = p.is_dir()
    if not report.is_dir:
        report.problems.append(f"not a directory: {p}")
        return report
    report.readable = os.access(p, os.R_OK | os.X_OK)
    if not report.readable:
        report.problems.append(f"not readable: {p}")
        return report

    # config.json — must exist and parse; model_type is surfaced in the
    # report so `qmlx doctor` can show it without a second read.
    config_path = p / "config.json"
    if not config_path.is_file():
        report.problems.append("missing config.json")
    else:
        try:
            with open(config_path) as fh:
                cfg = json.load(fh)
            report.config_ok = isinstance(cfg, dict)
            if report.config_ok:
                report.model_type = cfg.get("model_type")
            else:
                report.problems.append("config.json is not a JSON object")
        except (OSError, json.JSONDecodeError) as e:
            report.problems.append(f"config.json unreadable: {e}")

    # Weights — mlx-lm's loader globs model*.safetensors (mlx_lm/utils.py).
    report.weight_files = sorted(f.name for f in p.glob(WEIGHT_GLOB) if f.is_file())
    if not report.weight_files:
        report.problems.append(f"no weight files (expected at least one {WEIGHT_GLOB})")

    # Tokenizer — any of the common layouts.
    for name in TOKENIZER_FILES:
        if (p / name).is_file():
            report.tokenizer_file = name
            break
    if report.tokenizer_file is None:
        report.problems.append(
            "no tokenizer file (expected one of " + " / ".join(TOKENIZER_FILES) + ")"
        )

    report.size_bytes = _size_of(p)
    return report


# ---------------------------------------------------------------------------
# Name registry — ~/.cache/qmlx/models/
# ---------------------------------------------------------------------------


def models_dir() -> Path:
    """Registry root. ``QMLX_MODELS_DIR`` overrides (tests, shared volumes)."""
    override = os.environ.get("QMLX_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "qmlx" / "models"


def _registry_entry(name: str) -> Path:
    # Reject path-shaped names so ``qmlx serve ../elsewhere`` can never
    # escape the registry via a crafted name.
    if not name or "/" in name or name in (".", "..") or name.startswith("-"):
        raise ValueError(f"invalid model name: {name!r}")
    return models_dir() / name


def list_local_models() -> dict[str, Path]:
    """Return ``{name: path}`` for every registered local model.

    Broken symlinks are reported as-is (the path is returned even when
    the target vanished — ``qmlx doctor`` is the place that surfaces
    that, not the registry).
    """
    root = models_dir()
    if not root.is_dir():
        return {}
    return {p.name: p for p in sorted(root.iterdir()) if p.is_symlink() or p.is_dir()}


def resolve_local_model(name: str) -> Path | None:
    """Return the registered path for ``name``, or ``None``.

    Only a bare registry hit counts — an existing on-disk directory is
    *not* a registry hit (that is the direct ``qmlx serve /path`` case
    and needs no registration).
    """
    try:
        entry = _registry_entry(name)
    except ValueError:
        return None
    if entry.is_symlink() or entry.is_dir():
        return entry
    return None


def register_local_model(
    src: Path | str,
    name: str | None = None,
    copy: bool = False,
) -> Path:
    """Validate ``src`` and register it under the models dir.

    Default is a symlink (zero extra disk); ``copy=True`` makes a real
    copy (for models that will be moved or unmounted later).
    Re-registering an existing name replaces the entry (idempotent
    refresh — re-pointing a symlink at a re-quantised directory is the
    common case).
    """
    src = Path(src).expanduser().resolve()
    report = validate_local_model_dir(src)
    if not report.ok:
        raise LocalModelValidationError(src, report)
    name = name or src.name
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    dest = _registry_entry(name)
    if dest.is_symlink() or dest.is_dir():
        if dest.is_symlink():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    if copy:
        shutil.copytree(src, dest)
    else:
        dest.symlink_to(src)
    return dest


def remove_local_model(name: str) -> bool:
    """Unregister ``name``. Returns True if an entry was removed."""
    entry = _registry_entry(name)
    if entry.is_symlink():
        entry.unlink()
        return True
    if entry.is_dir():
        shutil.rmtree(entry)
        return True
    return False


class LocalModelValidationError(Exception):
    """A directory is not a loadable model snapshot."""

    def __init__(self, path: Path, report: LocalModelReport) -> None:
        self.path = path
        self.report = report
        super().__init__(self.render())

    def render(self) -> str:
        lines = [f"  Error: {self.path} is not a loadable model directory:"]
        lines.extend(f"    - {p}" for p in self.report.problems)
        lines.append(
            "  Expected: config.json, at least one model*.safetensors "
            "weight shard, and a tokenizer file (tokenizer.json / "
            "tokenizer_config.json / tokenizer.model)."
        )
        return "\n".join(lines)


def is_local_model_path(name: str) -> bool:
    """True when ``name`` is an existing local directory (the direct
    ``qmlx serve /path`` case — no registry involved)."""
    return os.path.isdir(name)


def preflight_local_model(name: str) -> None:
    """Serve-path gate for local model directories (issue #52 step 3).

    Validates the directory and either prints a one-line confirmation
    (the model is loadable) or exits 1 with a readable report. Runs
    BEFORE any model load so a missing shard surfaces as a clean error
    instead of a deep mlx-lm traceback after a long import chain.
    """
    report = validate_local_model_dir(Path(name))
    if not report.ok:
        print()
        print(LocalModelValidationError(Path(name).expanduser(), report).render())
        print()
        sys.exit(1)
    size_gb = report.size_bytes / (1024**3)
    print(
        f"  Local model: {name} ({size_gb:.1f} GB, "
        f"weights={len(report.weight_files)} shard(s), "
        f"tokenizer={report.tokenizer_file}) — offline, no HuggingFace hub access"
    )


def set_hf_offline() -> None:
    """Pin this process to offline mode for local-path serving.

    With the model resolved from a local directory, any HuggingFace hub
    access would be a pure failure mode (no network by definition) or a
    slow hang (network present but not wanted). Setting the standard
    offline flags makes every incidental hub lookup fail fast and keeps
    the zero-network guarantee in code rather than in an env var the
    user has to set:

    * ``HF_HUB_OFFLINE=1`` — huggingface_hub never touches the network.
    * ``HF_DATASETS_OFFLINE=1`` — same for the datasets client.
    * ``TRANSFORMERS_OFFLINE=1`` — transformers ``from_pretrained`` on a
      local path stays local (no hub probe for tokenizer extras).

    Idempotent; safe to call more than once.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
