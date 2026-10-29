# SPDX-License-Identifier: Apache-2.0
"""Tests for the `qmlx models` table column-alignment contract.

Dogfood-driven: 0.9.5 had a hardcoded 24-char alias column. The actual
registry has names up to 31 chars (``deepseek-coder-v2-lite-16b-4bit``),
which overflowed and shifted the rest of that row's columns. 0.9.6 sizes
the column from the data with a 24-char floor.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.cli import models_command
from vllm_mlx.model_aliases import list_visible_profiles


def _capture(capsys, **arg_overrides):
    args = SimpleNamespace(cached=False, **arg_overrides)
    models_command(args)
    return capsys.readouterr().out


def test_every_row_aligns_with_the_header_separator(capsys):
    """Each data row must have the same number of visible columns and
    the same column positions as the header. With the old fixed 24-char
    alias column, the 31-char ``deepseek-coder-v2-lite-16b-4bit`` row
    pushed Tools / Reasoning / Spec-Decode out one full column position.

    issue #62: installed rows lead with a "● " marker *inside* the Alias
    cell, and missing-data cells render blank (not "✗" / "unknown") —
    the check below validates the *padding* between the alias name and
    the header's Tools offset, so neither breaks it.
    """
    out = _capture(capsys)
    lines = [ln for ln in out.splitlines() if ln.startswith("  ")]
    # Find the header line ("  Alias ... DFlash") and the data rows
    # immediately following (between two separator lines).
    header_idx = next(
        i
        for i, ln in enumerate(lines)
        if ln.lstrip().startswith("Alias") and "DFlash" in ln and "HF id" not in ln
    )
    header = lines[header_idx]
    # The data rows start two lines after the header (separator, then rows)
    # and continue until the next separator line of box-drawing dashes.
    data_rows: list[str] = []
    for ln in lines[header_idx + 2 :]:
        if set(ln.strip()) == {"─"}:
            break
        data_rows.append(ln)
    assert len(data_rows) >= 50, "expected the full presented (qwen3.*) listing"
    # Row count must match the visible catalog exactly — the available-
    # models surface is qwen3.*-only (operator directive 2026-10-23).
    assert len(data_rows) == len(list_visible_profiles())

    # Column position of "Tools" in the header — every data row must
    # have its second column starting at the same offset. The Alias cell
    # is either "<alias>" or "● <alias>" (installed marker, issue #62);
    # the marker's internal space is part of the cell, and the fixed
    # column padding between the cell and tools_col must be blank.
    tools_col = header.index("Tools")
    for row in data_rows:
        cell = row[2:tools_col].rstrip()
        name = cell[2:] if cell.startswith("● ") else cell
        assert name and " " not in name, (
            f"Alias cell is not one name (optional '● ' marker): {row!r}"
        )
        assert row[2 + len(cell) : tools_col].isspace(), (
            f"Row mis-aligned: non-blank padding before Tools col. Row: {row!r}"
        )


def test_alias_column_width_floor_is_24(capsys, monkeypatch):
    """If the registry only has short names, the alias column must
    still be 24 wide so short tables don't feel cramped."""
    from vllm_mlx import model_aliases
    from vllm_mlx.model_aliases import AliasProfile

    short_profile = AliasProfile(hf_path="x/y")
    # Patch the *presented* catalog (models_command renders
    # list_visible_profiles, the qwen3.*-only operator-facing view).
    monkeypatch.setattr(
        model_aliases, "list_visible_profiles", lambda: {"qwen3": short_profile}
    )
    out = _capture(capsys)
    # Header has "Alias" followed by at least 19 spaces before "Tools"
    # → column starts at position 2 + 24 + 1 = 27.
    header_line = next(
        ln for ln in out.splitlines() if "Alias" in ln and "DFlash" in ln
    )
    assert header_line.index("Tools") - header_line.index("Alias") == 25, (
        "Alias-column floor regression: short registry should still pad "
        "to 24 chars (Tools header at offset 25 from Alias)."
    )


def test_longest_real_alias_does_not_overflow(capsys):
    """End-to-end: with the real registry, the longest *presented* alias
    still gets its column with at least 1 space before the next column."""
    out = _capture(capsys)
    longest_alias = max(list_visible_profiles().keys(), key=len)
    data_line = next(
        ln for ln in out.splitlines() if ln.lstrip().startswith(longest_alias)
    )
    after_alias = data_line[2 + len(longest_alias) :]
    # The character immediately after the alias must be a space and
    # what follows must be the Tools column (not another part of the
    # alias name).
    assert after_alias.startswith(" "), (
        f"No padding between alias and Tools column for {longest_alias!r}"
    )
