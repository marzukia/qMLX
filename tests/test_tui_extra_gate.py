# SPDX-License-Identifier: Apache-2.0
"""Issue #51 AC9 gate — the optional TUI extra fails friendly, not tracebacks.

A bare install (no ``[tui]`` extra) must print the one-line install hint and
exit 1 from ``qmlx tui`` / ``qmlx chat --tui`` — mirroring the Gradio hint
pattern. The hint check has to run BEFORE ``from vllm_mlx import tui``
because tui.py imports textual at module top level; before the import-order
fix a bare install died with an ImportError traceback instead.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from vllm_mlx import cli


def test_tui_command_without_textual_prints_hint_and_exits(capsys, monkeypatch):
    """``qmlx tui`` with textual blocked: hint on stdout, exit 1, no traceback."""
    # A None entry in sys.modules makes `import textual` raise ImportError.
    monkeypatch.setitem(sys.modules, "textual", None)
    args = SimpleNamespace(
        model="qwen3.5-9b-4bit", port=None, base_url=None, system=None
    )
    with pytest.raises(SystemExit) as exc:
        cli.tui_command(args)
    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "TUI extra not installed" in captured.out
    assert "pip install" in captured.out
    # No traceback on stderr — the friendly hint replaces it (AC9).
    assert captured.err == ""


def test_tui_available_reflects_missing_extra(monkeypatch):
    """``tui.tui_available()`` is the module-level probe the hint relies on."""
    try:
        import textual  # noqa: F401
    except Exception:  # noqa: BLE001 — base install: probe path still testable
        pytest.skip("textual extra not installed in this environment")
    from vllm_mlx import tui as _tui

    assert _tui.tui_available() is True
    monkeypatch.setitem(sys.modules, "textual", None)
    assert _tui.tui_available() is False
