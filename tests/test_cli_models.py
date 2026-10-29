# SPDX-License-Identifier: Apache-2.0
"""Tests for `qmlx models` — pins the per-alias profile rendering."""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import patch

from vllm_mlx import cli
from vllm_mlx.model_aliases import list_profiles, list_visible_profiles


def _capture_models_output() -> str:
    """Run `models_command` with stdout captured. Patches the version-check
    helper so the test stays hermetic (no PyPI lookup at test time)."""
    buf = io.StringIO()
    with (
        patch.object(sys, "stdout", buf),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.models_command(None)
    return buf.getvalue()


def test_models_command_lists_all_aliases():
    """Every *presented* (qwen3.*) alias must appear in the table.

    The available-models surface is qwen3.*-only (operator directive
    2026-10-23); the full registry still exists but is not listed.
    Non-qwen3 aliases remain resolvable by explicit name — they just
    don't show up here.
    """
    out = _capture_models_output()
    profiles = list_visible_profiles()
    assert len(profiles) >= 20, "expected 20+ visible aliases (per project goal)"
    for alias in profiles:
        assert alias in out, f"alias {alias!r} missing from `qmlx models` output"
    assert f"({len(profiles)} aliases)" in out


def test_models_command_shows_capability_columns():
    """Capability columns (Tools, Reasoning, Spec-Decode, Suffix Tier) appear."""
    out = _capture_models_output()
    for header in ("Tools", "Reasoning", "Spec-Decode", "Suffix Tier"):
        assert header in out, f"column header {header!r} missing"


def test_models_command_renders_hybrid_marker_for_qwen35_moe():
    """Hybrid MoE models (e.g. qwen3.5-35b-4bit, the A3B variant) must
    show the 'hybrid' marker and a BLANK suffix-tier cell (issue #62:
    'n/a' / 'unknown' rendered the data gap as data).

    r6-A R6-C1: the test was previously written against the DENSE
    ``qwen3.5-4b-4bit`` alias, which we've now flipped to non-hybrid (it
    was the metal::malloc wedge surface). The CLI column contract is the
    same — only the surface alias changes — so pivot to the A3B MoE
    Qwen3.5 variant which still legitimately wears the hybrid marker.

    The point of the column is to spare users an `info` round-trip when
    deciding whether spec-decode/suffix-decode will help. Trust the gate.
    """
    out = _capture_models_output()
    profiles = list_profiles()
    qwen35_moe = profiles.get("qwen3.5-35b-4bit")
    assert qwen35_moe is not None, "qwen3.5-35b-4bit alias missing — fixture drift"
    assert qwen35_moe.is_hybrid, (
        "qwen3.5-35b-4bit (A3B MoE) should remain is_hybrid=True"
    )

    matches = [line for line in out.splitlines() if "qwen3.5-35b-4bit " in line]
    assert matches, "no row found for qwen3.5-35b-4bit"
    row = matches[0]
    assert "hybrid" in row, f"expected 'hybrid' marker in row: {row!r}"
    # Absent tier renders blank — no 'n/a', no 'unknown' placeholder.
    assert "n/a" not in row, f"expected blank tier cell (was 'n/a'): {row!r}"
    assert "unknown" not in row, f"expected blank tier cell: {row!r}"


def test_models_command_renders_parser_for_qwen35_9b():
    """Non-hybrid model with parser + tier column renders both cells.

    Pinned to the canonical bring-up model (qwen3.5-9b-4bit) per the
    2026-10-23 operator directive — the table is qwen3.*-only, so the
    hermes3-8b-4bit pin no longer has a row to inspect. Two contracts:
    (1) the tool parser cell shows the value from the alias registry,
    (2) the suffix-tier cell shows the tier currently recorded in
    ``aliases.json``. Reading the expected tier from the registry (not
    hardcoding it) means a future bench re-sweep that reclassifies the
    alias doesn't break this test, while a *display* regression (tier
    dropped from the row entirely) still does.
    """
    out = _capture_models_output()
    alias = "qwen3.5-9b-4bit"
    matches = [line for line in out.splitlines() if f"{alias} " in line]
    assert matches, f"no row found for {alias}"
    row = matches[0]
    profile = list_visible_profiles().get(alias)
    assert profile is not None, f"{alias} alias missing — fixture drift"
    assert (profile.tool_call_parser or "") in row, (
        f"expected tool parser {profile.tool_call_parser!r} in row: {row!r}"
    )
    tier = profile.suffix_decoding_tier
    if tier in ("unknown", "n/a"):
        # Issue #62: an unbenched tier renders blank, not as data.
        assert str(tier) not in row, (
            f"unbenched tier {tier!r} should render blank: {row!r}"
        )
    else:
        assert str(tier) in row, f"expected suffix tier {tier!r} in row: {row!r}"


def test_models_command_renders_blank_for_unset_parsers():
    """An alias without tool_call_parser / reasoning_parser renders
    BLANK cells (issue #62) — the em-dash placeholder printed the data
    gap as data. Check the actual column cells, not a dash count (other
    columns no longer use '—' either)."""
    out = _capture_models_output()
    profiles = list_visible_profiles()
    # Find any *presented* alias that has neither parser populated.
    candidates = [
        a
        for a, p in profiles.items()
        if not p.tool_call_parser and not p.reasoning_parser
    ]
    if not candidates:
        # Schema may have tightened — skip cleanly.
        import pytest

        pytest.skip("no aliases without parsers (schema may have tightened)")
    alias = candidates[0]
    matches = [line for line in out.splitlines() if f"{alias} " in line]
    assert matches, f"no row found for {alias}"
    row = matches[0]
    # The plain path lays rows out as fixed-width fields; derive every
    # field boundary from the captured header so the test survives a
    # legitimate width re-size (issue #62 shrunk the fixed columns).
    header = next(
        ln
        for ln in out.splitlines()
        if ln.lstrip().startswith("Alias") and "Tools" in ln and "Reasoning" in ln
    )
    tools_col = header.index("Tools")
    reason_col = header.index("Reasoning")
    spec_col = header.index("Spec-Decode")
    tools_cell = row[tools_col : reason_col - 1].strip()
    reasoning_cell = row[reason_col : spec_col - 1].strip()
    assert tools_cell == "", f"expected blank Tools cell, got {tools_cell!r}: {row!r}"
    assert reasoning_cell == "", (
        f"expected blank Reasoning cell, got {reasoning_cell!r}: {row!r}"
    )


def test_models_command_mentions_chat_pull_serve_in_tip():
    """The footer should advertise the four canonical actions."""
    out = _capture_models_output()
    for cmd in ("info", "pull", "chat", "serve"):
        assert f"qmlx {cmd}" in out, f"footer tip missing 'qmlx {cmd}' suggestion"


def test_models_command_subparser_smoke():
    """`qmlx models --help` exits 0 (subparser still wired)."""
    import pytest

    with (
        patch.object(sys, "argv", ["qmlx", "models", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


# ----------------------------------------------------------------------
# D2 — --cached / ls view
# ----------------------------------------------------------------------


def test_ls_subcommand_registered():
    """``qmlx ls --help`` exits 0 (top-level alias is wired)."""
    import pytest

    with (
        patch.object(sys, "argv", ["qmlx", "ls", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


def test_ls_routes_to_models_with_cached(monkeypatch):
    """``qmlx ls`` invokes the cached view via ``models_command``
    with ``args.cached = True``. We patch ``models_command`` to capture
    the args namespace before the body runs."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["qmlx", "ls"]),
        patch.object(cli, "models_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    assert captured[0].cached is True


def test_models_cached_flag_routes_to_cached_view(monkeypatch, capsys):
    """``models --cached`` must call the cached-view path (not the
    full alias table). We assert via the printed header difference."""
    # No cached models will be found in a fresh tmp HF cache.
    monkeypatch.setenv("HF_HOME", "/nonexistent_path_for_this_test_xyz")
    # huggingface_hub.constants.HF_HUB_CACHE is evaluated at module load,
    # so we instead patch the helper directly.
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])

    with (
        patch.object(sys, "argv", ["qmlx", "models", "--cached"]),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.main()
    out = capsys.readouterr().out
    assert "No models cached yet" in out
    # And the full alias table header is absent.
    assert "Available models" not in out


def test_models_default_view_unchanged(monkeypatch, capsys):
    """Bare ``qmlx models`` still prints the capability table —
    --cached is opt-in. Backward-compat contract."""
    with (
        patch.object(sys, "argv", ["qmlx", "models"]),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.main()
    out = capsys.readouterr().out
    assert "Available models" in out
    assert "Tools" in out and "Reasoning" in out


def test_cached_view_renders_alias_for_known_repo(tmp_path, monkeypatch, capsys):
    """A cached HF repo whose path matches an alias should render under
    the alias name (e.g. ``qwen3.5-4b-4bit``), not the raw HF path."""
    from vllm_mlx.model_aliases import list_profiles

    profiles = list_profiles()
    # Pick any alias for the test; we'll synthesize a fake cache entry
    # at its hf_path.
    alias = next(iter(profiles))
    hf_path = profiles[alias].hf_path

    monkeypatch.setattr(
        cli, "_scan_hf_cache_models", lambda: [(hf_path, 1024 * 1024 * 100, 0.0)]
    )
    cli._print_cached_models()
    out = capsys.readouterr().out
    assert alias in out, f"expected alias {alias!r} in cached view"
    assert hf_path[:40] in out, "expected HF path in cached view"


def test_cached_view_renders_unmapped_for_unknown_repo(monkeypatch, capsys):
    """A cached HF repo with no alias entry must show ``(unmapped)``."""
    monkeypatch.setattr(
        cli,
        "_scan_hf_cache_models",
        lambda: [("some/totally-unmapped-repo", 1024, 0.0)],
    )
    cli._print_cached_models()
    out = capsys.readouterr().out
    assert "(unmapped)" in out
    assert "totally-unmapped-repo" in out


def test_format_bytes_unit_selection():
    """``_format_bytes`` picks the largest unit where value >= 1.

    Suffixes are IEC base-1024 (KiB/MiB/GiB) — aligned with
    ``_format_size`` in ``vllm_mlx._download_gate`` so the same byte
    count is rendered identically by ``ls --cached`` and the B2 prompt
    (DeepSeek round-3 NIT #4)."""
    assert cli._format_bytes(0) == "0 B"
    assert cli._format_bytes(512) == "512 B"
    assert cli._format_bytes(2048) == "2.0 KiB"
    assert cli._format_bytes(5 * 1024 * 1024) == "5.0 MiB"
    assert cli._format_bytes(int(2.5 * 1024**3)) == "2.5 GiB"


def test_scan_hf_cache_models_filters_to_models_only(tmp_path, monkeypatch):
    """Only ``models--*`` directories should show in the listing — not
    ``datasets--*`` or ``spaces--*``."""
    cache_root = tmp_path / "hub"
    cache_root.mkdir()
    (cache_root / "models--mlx-community--FakeModel").mkdir()
    (cache_root / "models--mlx-community--FakeModel" / "blob1").write_bytes(b"x" * 128)
    (cache_root / "datasets--squad").mkdir()
    (cache_root / "datasets--squad" / "data").write_bytes(b"y" * 999)
    (cache_root / "spaces--gradio--demo").mkdir()

    # Patch the constants lookup inside _scan_hf_cache_models.
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(cache_root), raising=False
    )
    rows = cli._scan_hf_cache_models()
    repos = [r[0] for r in rows]
    assert "mlx-community/FakeModel" in repos
    assert all("squad" not in r for r in repos)
    assert all("demo" not in r for r in repos)


# ---------------------------------------------------------------------------
# issue #62 defect 1 — overflow: installed rows first, TTY cap, --all
# ---------------------------------------------------------------------------


def _synthetic_profiles(n: int) -> dict:
    """A hermetic alias registry: n synthetic aliases, no capabilities."""
    from types import SimpleNamespace

    profs = {}
    for i in range(n):
        alias = f"syn-{i:02d}"
        profs[alias] = SimpleNamespace(
            hf_path=f"org/{alias}",
            tool_call_parser=None,
            reasoning_parser=None,
            is_hybrid=False,
            supports_spec_decode=False,
            suffix_decoding_tier="unknown",
            supports_dflash=False,
            supports_ddtree=False,
        )
    return profs


def test_models_installed_rows_sort_first(monkeypatch, capsys):
    """Installed aliases render before pullable ones (issue #62)."""
    profiles = _synthetic_profiles(6)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.list_visible_profiles", lambda: profiles
    )
    monkeypatch.setattr(cli, "_installed_alias_set", lambda p: {"syn-03"})
    cli.models_command(None)
    out = capsys.readouterr().out
    idx = {a: out.index(a) for a in ("syn-00", "syn-03", "syn-05")}
    assert idx["syn-03"] < idx["syn-00"] < idx["syn-05"]
    # Alphabetical order is preserved *within* each group.
    assert idx["syn-00"] < out.index("syn-01") < idx["syn-05"]


def test_models_tty_cap_collapses_overflow(monkeypatch, capsys):
    """On a 30-line TTY the table caps so the column header stays on
    screen; the remainder collapses into a pointer at `--all`."""
    from types import SimpleNamespace

    profiles = _synthetic_profiles(30)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.list_visible_profiles", lambda: profiles
    )
    monkeypatch.setattr(cli, "_installed_alias_set", lambda p: set())
    monkeypatch.setattr("vllm_mlx._ui.is_rich", lambda: True)
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((100, 30))
    )
    cli.models_command(SimpleNamespace(cached=False, all_models=False))
    out = capsys.readouterr().out
    # 30 lines - 12 chrome - 1 pointer line = 17 rows shown, 13 collapsed.
    assert "… 13 more" in out
    for i in range(17):
        assert f"syn-{i:02d}" in out
    for i in range(17, 30):
        assert f"syn-{i:02d}" not in out


def test_models_all_flag_disables_tty_cap(monkeypatch, capsys):
    """`--all` shows every row even on a small TTY."""
    from types import SimpleNamespace

    profiles = _synthetic_profiles(30)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.list_visible_profiles", lambda: profiles
    )
    monkeypatch.setattr(cli, "_installed_alias_set", lambda p: set())
    monkeypatch.setattr("vllm_mlx._ui.is_rich", lambda: True)
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((100, 30))
    )
    cli.models_command(SimpleNamespace(cached=False, all_models=True))
    out = capsys.readouterr().out
    assert "… " not in out
    assert f"Available models ({len(profiles)} aliases)" in out
    for i in range(30):
        assert f"syn-{i:02d}" in out


def test_models_pipe_output_uncapped(monkeypatch, capsys):
    """Plain (pipe) output is never row-capped — scripts must see every
    alias regardless of terminal size."""
    from types import SimpleNamespace

    profiles = _synthetic_profiles(30)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.list_visible_profiles", lambda: profiles
    )
    monkeypatch.setattr(cli, "_installed_alias_set", lambda p: set())
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((100, 12))
    )
    cli.models_command(SimpleNamespace(cached=False, all_models=False))
    out = capsys.readouterr().out
    assert "… " not in out
    for i in range(30):
        assert f"syn-{i:02d}" in out


def test_models_installed_rows_carry_marker(monkeypatch, capsys):
    """Installed rows lead with a '● ' prefix; the title reports the
    local count (issue #62, defect 3)."""
    profiles = _synthetic_profiles(5)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.list_visible_profiles", lambda: profiles
    )
    monkeypatch.setattr(cli, "_installed_alias_set", lambda p: {"syn-02"})
    cli.models_command(None)
    out = capsys.readouterr().out
    assert "● syn-02" in out
    assert "— 1 local" in out
    for a in ("syn-00", "syn-01", "syn-03", "syn-04"):
        assert f"● {a}" not in out


def test_models_title_has_no_local_suffix_when_none_local(monkeypatch, capsys):
    profiles = _synthetic_profiles(3)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.list_visible_profiles", lambda: profiles
    )
    monkeypatch.setattr(cli, "_installed_alias_set", lambda p: set())
    cli.models_command(None)
    out = capsys.readouterr().out
    assert "Available models (3 aliases)" in out
    assert "local" not in out
    assert "● " not in out


def test_models_rich_mode_dims_pullable_rows(monkeypatch, capsys):
    """rich path: row_styles plumbed through — installed rows get
    ``None`` (full strength), pullable rows get ``dim``."""
    from types import SimpleNamespace

    import vllm_mlx._ui as ui_mod

    profiles = _synthetic_profiles(4)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.list_visible_profiles", lambda: profiles
    )
    monkeypatch.setattr(cli, "_installed_alias_set", lambda p: {"syn-00"})
    monkeypatch.setattr("vllm_mlx._ui.is_rich", lambda: True)
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((100, 40))
    )
    captured_styles = {}

    def fake_rich_table(headers, rows, **kwargs):
        captured_styles.update(kwargs)
        # render nothing but the alias column, so the marker is visible
        for r in rows:
            print(r[0])

    monkeypatch.setattr(ui_mod, "rich_table", fake_rich_table)
    cli.models_command(SimpleNamespace(cached=False, all_models=False))
    out = capsys.readouterr().out
    styles = captured_styles.get("row_styles")
    assert styles is not None
    assert styles[0] is None  # installed
    assert all(s == "dim" for s in styles[1:])
    assert "● syn-00" in out
