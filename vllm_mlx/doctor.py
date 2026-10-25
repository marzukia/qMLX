# SPDX-License-Identifier: Apache-2.0
"""qmlx doctor — environment health check (issue #52, onboarding step 2).

``qmlx doctor`` answers one question: *is this machine ready to serve?*
It probes, in order:

* **Python** — interpreter version (the ``requires-python`` floor).
* **Packages** — mlx / mlx-lm / transformers / fastapi / uvicorn
  installed, with the pyproject version floors (a present-but-too-old
  package is a warning, a missing one is a failure).
* **Backend** — Apple Silicon / Metal (qMLX is built for Apple Silicon;
  other platforms warn rather than fail — the package still imports).
* **Model** (``--model <name-or-path>``) — the model directory is
  readable, ``config.json`` parses, weight shards and a tokenizer file
  are present. Local directories are validated on disk; HuggingFace
  repo names are probed against the *local* HF cache only (no network).
* **Disk** — free space on the KV-checkpoint volume (the disk KV
  store is the only cache tier, so a nearly-full volume is a real
  serving problem) and, for local model dirs, on the model volume.

Every check is cheap (well under a second total, one bounded cache
walk at most) and side-effect-free. The report renders as fixed
sections with ✓ / ⚠ / ✗ rows and an aggregate summary.

Exit codes:
  0 — no ✗ items (warnings never fail the exit code)
  1 — one or more ✗ items

``--json`` emits the same report as a JSON object for scripting.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


_STATUS_GLYPH = {
    CheckStatus.OK: "✓",
    CheckStatus.WARN: "⚠",
    CheckStatus.FAIL: "✗",
}


@dataclass
class Check:
    """One row in a section. ``detail`` renders under the label."""

    label: str
    status: CheckStatus
    detail: str = ""


@dataclass
class Section:
    title: str
    checks: list[Check] = field(default_factory=list)

    def add(self, label: str, status: CheckStatus, detail: str = "") -> None:
        self.checks.append(Check(label=label, status=status, detail=detail))


@dataclass
class Report:
    sections: list[Section] = field(default_factory=list)

    def add_section(self, title: str) -> Section:
        section = Section(title=title)
        self.sections.append(section)
        return section

    def all_checks(self) -> list[Check]:
        return [c for s in self.sections for c in s.checks]

    @property
    def n_ok(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.OK)

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.WARN)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.FAIL)

    @property
    def exit_code(self) -> int:
        # Warnings never fail the exit code — only ✗ items do. A CI
        # script gating on doctor wants "is anything broken", not "is
        # anything not perfect".
        return 1 if self.n_fail else 0

    def to_text(self) -> str:
        lines: list[str] = []
        for section in self.sections:
            lines.append(f"{section.title}")
            lines.append("─" * len(section.title))
            for check in section.checks:
                glyph = _STATUS_GLYPH[check.status]
                line = f"  {glyph} {check.label}"
                if check.detail:
                    line += f" — {check.detail}"
                lines.append(line)
            lines.append("")
        lines.append(
            f"  {self.n_ok} ok, {self.n_warn} warning(s), {self.n_fail} failure(s)"
        )
        if self.n_fail:
            lines.append("  Environment is NOT ready — fix the ✗ items above.")
        elif self.n_warn:
            lines.append("  Environment is ready (with warnings).")
        else:
            lines.append("  Environment is ready.")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "sections": [
                {
                    "title": s.title,
                    "checks": [
                        {
                            "label": c.label,
                            "status": c.status.value,
                            "detail": c.detail,
                        }
                        for c in s.checks
                    ],
                }
                for s in self.sections
            ],
            "summary": {
                "ok": self.n_ok,
                "warn": self.n_warn,
                "fail": self.n_fail,
                "exit_code": self.exit_code,
            },
        }


# ---------------------------------------------------------------------------
# Version floors — mirror pyproject.toml [project].dependencies. Bump both
# together; the doctor surfaces a WARN (not FAIL) for a present-but-old
# package because a slightly-old mlx still loads, it just may miss the
# fixes the floor was bumped for.
# ---------------------------------------------------------------------------

PYTHON_FLOOR = (3, 10)
PACKAGE_FLOORS: dict[str, str] = {
    "mlx": "0.31.2",
    "mlx-lm": "0.31.3",
    "transformers": "5.0.0",
}
# transformers 5.13.0 broke ``AutoTokenizer.register`` for string-class
# registrations (see pyproject.toml) — above the cap, even ``import
# mlx_lm`` can raise at import time.
TRANSFORMERS_CAP = "5.13"
# Present-and-working is enough for the web stack; no floor beyond
# "installed" (pyproject pins the real floors at install time).
WEB_PACKAGES: tuple[str, ...] = ("fastapi", "uvicorn")
# Install hint rendered when a required package is missing.
INSTALL_HINT = "pip install qmlx-serve"


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse ``0.31.2`` / ``5.12.1`` into a comparable int tuple.

    Non-numeric suffixes (``rc1``, ``.dev0``) are dropped — a pre-
    release of 5.13 sorts as 5.13 for floor/cap purposes, which is the
    conservative reading (it already has the 5.13 behaviour).
    """
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _version_at_least(version: str, floor: str) -> bool:
    v, f = _version_tuple(version), _version_tuple(floor)
    n = max(len(v), len(f))
    v += (0,) * (n - len(v))
    f += (0,) * (n - len(f))
    return v >= f


def _version_below_cap(version: str, cap: str) -> bool:
    v, c = _version_tuple(version), _version_tuple(cap)
    n = max(len(v), len(c))
    v += (0,) * (n - len(v))
    c += (0,) * (n - len(c))
    return v < c


# ---------------------------------------------------------------------------
# Probes — small indirections so tests can monkeypatch without touching
# importlib / platform / the filesystem.
# ---------------------------------------------------------------------------


def pkg_version(name: str) -> str | None:
    """Installed version of a PyPI distribution, or ``None`` if absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def platform_info() -> tuple[str, str]:
    """``(system, machine)`` — e.g. ``("Darwin", "arm64")``."""
    return platform.system(), platform.machine()


def disk_free_bytes(path: Path) -> int | None:
    """Free bytes on the volume holding ``path`` (walking up to the
    first existing ancestor, mirroring ``cli._check_disk_space``)."""
    probe = os.path.abspath(str(path))
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not probe or not os.path.exists(probe):
        probe = os.path.expanduser("~")
    try:
        stat = os.statvfs(probe)
        return stat.f_bavail * stat.f_frsize
    except (OSError, AttributeError):
        return None


def _format_bytes(n: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(n)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"  # unreachable, keeps the flow linear


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def check_python(report: Report) -> None:
    section = report.add_section("Python")
    version = sys.version_info
    rendered = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) >= PYTHON_FLOOR:
        section.add(
            "interpreter version",
            CheckStatus.OK,
            f"{rendered} (floor {'.'.join(map(str, PYTHON_FLOOR))})",
        )
    else:
        section.add(
            "interpreter version",
            CheckStatus.FAIL,
            f"{rendered} — qMLX requires >= {'.'.join(map(str, PYTHON_FLOOR))}",
        )


def check_packages(report: Report) -> None:
    section = report.add_section("Packages")
    for name in ("mlx", "mlx-lm", "transformers", *WEB_PACKAGES):
        version = pkg_version(name)
        if version is None:
            section.add(
                name,
                CheckStatus.FAIL,
                f"not installed ({INSTALL_HINT})",
            )
            continue
        floor = PACKAGE_FLOORS.get(name)
        if floor is not None and not _version_at_least(version, floor):
            section.add(
                name,
                CheckStatus.WARN,
                f"{version} — below the {floor} floor in pyproject.toml",
            )
            continue
        if name == "transformers" and not _version_below_cap(version, TRANSFORMERS_CAP):
            section.add(
                name,
                CheckStatus.WARN,
                f"{version} — at/above the {TRANSFORMERS_CAP} cap "
                "(transformers 5.13 breaks mlx-lm's tokenizer "
                "registration; see pyproject.toml)",
            )
            continue
        section.add(name, CheckStatus.OK, version)


def check_backend(report: Report) -> None:
    section = report.add_section("Backend")
    system, machine = platform_info()
    if system == "Darwin" and machine == "arm64":
        section.add("Apple Silicon (Metal)", CheckStatus.OK, machine)
    elif system == "Darwin":
        section.add(
            "Apple Silicon (Metal)",
            CheckStatus.WARN,
            f"Intel Mac ({machine}) — qMLX is built for Apple Silicon",
        )
    else:
        section.add(
            "Apple Silicon (Metal)",
            CheckStatus.WARN,
            f"{system}/{machine} — qMLX targets Apple Silicon; mlx is "
            "Apple-only, so serving will not work here",
        )


def check_model(report: Report, model_name: str | None) -> None:
    section = report.add_section("Model")
    if model_name is None:
        section.add(
            "model",
            CheckStatus.WARN,
            "not specified (pass --model to check a model directory)",
        )
        return

    # Resolve the name: registered local model, then alias, then
    # local path, then HF repo name.
    from .local_model import (
        is_local_model_path,
        resolve_local_model,
        validate_local_model_dir,
    )

    resolved = resolve_local_model(model_name)
    if resolved is None:
        from .model_aliases import resolve_model

        resolved = Path(resolve_model(model_name))

    # An argument shaped like a filesystem path (absolute, home- or
    # cwd-relative) is a local dir even when it does not exist yet —
    # "org/repo" (relative, no leading dot) stays an HF name.
    looks_like_path = model_name.startswith(("/", "~", "./", ".."))
    if looks_like_path or is_local_model_path(str(resolved)):
        p = Path(resolved)
        section.add(
            "directory",
            CheckStatus.OK
            if p.is_dir() and os.access(p, os.R_OK | os.X_OK)
            else CheckStatus.FAIL,
            str(p),
        )
        report_ = validate_local_model_dir(p)
        section.add(
            "config.json",
            CheckStatus.OK if report_.config_ok else CheckStatus.FAIL,
            f"model_type={report_.model_type}"
            if report_.config_ok
            else "missing or unparseable",
        )
        section.add(
            "weights",
            CheckStatus.OK if report_.weight_files else CheckStatus.FAIL,
            f"{len(report_.weight_files)} shard(s), "
            f"{_format_bytes(report_.size_bytes)} total"
            if report_.weight_files
            else "no model*.safetensors found",
        )
        section.add(
            "tokenizer",
            CheckStatus.OK if report_.tokenizer_file else CheckStatus.FAIL,
            report_.tokenizer_file or "no tokenizer file found",
        )
        return

    # Not a local directory — probe the local HF cache only (no
    # network: a doctor run must work on an air-gapped machine).
    if "/" in str(resolved):
        cached = _hf_cache_probe(str(resolved))
        if cached is not None:
            section.add(
                "HF cache",
                CheckStatus.OK,
                f"{resolved} is cached locally at {cached}",
            )
        else:
            section.add(
                "HF cache",
                CheckStatus.WARN,
                f"{resolved} is not in the local HF cache — serving "
                "it will download from HuggingFace (network required)",
            )
        return

    section.add(
        "name",
        CheckStatus.WARN,
        f"{model_name} is not an alias, a registered local model, or a "
        "path — check the spelling (qmlx models lists aliases)",
    )


def _hf_cache_probe(repo_id: str) -> str | None:
    """Local-only presence probe for an HF repo (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id, "config.json")
        if isinstance(cached, str) and os.path.exists(cached):
            return os.path.dirname(cached)
    except Exception:
        pass
    return None


def check_disk(report: Report, model_name: str | None) -> None:
    section = report.add_section("Disk")

    # KV checkpoint volume — the disk KV store is the ONLY cache tier,
    # so its volume's free space directly bounds cross-turn reuse.
    # ``get_default_root`` is stdlib-only at import time, so this is
    # safe to call from a doctor run (no engine / mlx import).
    from .runtime.disk_kv_checkpoint import get_default_root

    kv_dir = Path(get_default_root())
    free = disk_free_bytes(kv_dir)
    if free is None:
        section.add(
            "KV checkpoint volume",
            CheckStatus.WARN,
            f"could not stat {kv_dir}",
        )
    elif free < 1024**3:
        section.add(
            "KV checkpoint volume",
            CheckStatus.FAIL,
            f"only {_format_bytes(free)} free on {kv_dir} — the disk KV "
            "store will thrash",
        )
    elif free < 10 * 1024**3:
        section.add(
            "KV checkpoint volume",
            CheckStatus.WARN,
            f"{_format_bytes(free)} free on {kv_dir} — small for the "
            "100 GiB default checkpoint cap",
        )
    else:
        section.add(
            "KV checkpoint volume",
            CheckStatus.OK,
            f"{_format_bytes(free)} free on {kv_dir}",
        )

    # Model volume — for a local model dir, the volume must be able to
    # hold the model plus headroom.
    if model_name is not None:
        from .local_model import (
            is_local_model_path,
            resolve_local_model,
            validate_local_model_dir,
        )

        resolved = resolve_local_model(model_name)
        if resolved is None:
            resolved = Path(model_name)
        if is_local_model_path(str(resolved)):
            report_ = validate_local_model_dir(Path(resolved))
            vol_free = disk_free_bytes(Path(resolved))
            if vol_free is None:
                section.add(
                    "model volume",
                    CheckStatus.WARN,
                    f"could not stat {resolved}",
                )
            elif vol_free < report_.size_bytes:
                section.add(
                    "model volume",
                    CheckStatus.FAIL,
                    f"only {_format_bytes(vol_free)} free but the model "
                    f"is {_format_bytes(report_.size_bytes)} — the "
                    "volume is nearly full",
                )
            else:
                section.add(
                    "model volume",
                    CheckStatus.OK,
                    f"{_format_bytes(vol_free)} free for a "
                    f"{_format_bytes(report_.size_bytes)} model",
                )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_checks(model: str | None = None) -> Report:
    """Run every section and return the aggregate report."""
    report = Report()
    check_python(report)
    check_packages(report)
    check_backend(report)
    check_model(report, model)
    check_disk(report, model)
    return report


def doctor_command(args) -> None:
    """``qmlx doctor`` handler — render the report, exit 0/1."""
    model = getattr(args, "model", None)
    report = run_checks(model)
    if getattr(args, "json", False):
        print(json.dumps(report.to_json(), indent=2))
    else:
        print()
        print(report.to_text())
        print()
    sys.exit(report.exit_code)
