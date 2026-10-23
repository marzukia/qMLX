# SPDX-License-Identifier: Apache-2.0
"""Textual TUI for qMLX (issue #51, phase 4 — optional ``[tui]`` extra).

Opt-in surface: ``qmlx tui [model]`` / ``qmlx chat --tui``. The base
install has zero TUI dependencies; the CLI calls
:func:`ensure_tui_available` (one-line install hint + ``exit(1)``)
before importing the rest of this module (acceptance #51-AC9).

The app is deliberately small: a header, a scrollable conversation log,
an input box, and a status strip that polls ``/v1/status`` on a worker
thread. Chat messages stream from ``/v1/chat/completions`` so the reply
grows token-by-token in the log.
"""

from __future__ import annotations

import json
import sys
import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, Label, Log

#: One-line hint printed (and the exit code returned) when the ``tui``
#: extra is missing — pinned by tests (AC9).
TUI_INSTALL_HINT = "TUI extra not installed — run: pip install 'qmlx-serve[tui]'"


def tui_available() -> bool:
    """True when the ``textual`` import succeeds in this install."""
    try:
        import textual  # noqa: F401

        return True
    except Exception:
        return False


def ensure_tui_available() -> None:
    """Exit(1) with the one-line install hint when textual is missing."""
    if not tui_available():
        print(TUI_INSTALL_HINT)
        sys.exit(1)


def _resolve_base_url(args) -> str:
    """Base URL for the TUI: ``--base-url`` > ``--port`` > 127.0.0.1:8000."""
    base_url = getattr(args, "base_url", None)
    if base_url:
        return base_url.rstrip("/")
    port = getattr(args, "port", None) or 8000
    return f"http://127.0.0.1:{port}"


def _status_line(stats: dict | None) -> str:
    """Plain one-line status from a /v1/status payload (AC6 fields)."""
    if stats is None:
        return "  ⚡ qmlx — server not reachable"
    model = stats.get("model") or "-"
    state = stats.get("status", "idle")
    line = f"  ⚡ {model} · {state} · run {stats.get('num_running', 0)}"
    tps = stats.get("generation_tps") or 0.0
    if tps > 0:
        line += f" · {tps:.1f} tok/s"
    up = stats.get("uptime_s")
    if up is not None:
        line += f" · up {int(up)}s"
    return line


class QMLXTUI(App):
    """Chat TUI: streaming completions + a 1.5s-poll /v1/status strip."""

    BINDINGS = [Binding("q", "quit", "Quit", show=False)]

    def __init__(self, base_url: str, model: str, *, system: str | None = None):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system = system
        self.history: list[dict] = []
        if system:
            self.history.append({"role": "system", "content": system})

    def compose(self) -> ComposeResult:
        yield Header("🐆 qMLX TUI")
        yield Log(auto_scroll=True)
        yield Label(_status_line(None), id="qmlx-status")
        yield Input(
            placeholder=f"Message {self.model}…  (Ctrl-Q quits)",
            id="qmlx-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one(Log)
        log.write(f"model: {self.model} · endpoint: {self.base_url}/v1")
        self.run_worker(self._status_loop(), group="tui", exclusive=False)
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self.run_worker(self._send(text), group="tui", exclusive=False)

    def _send(self, message: str) -> None:
        import requests

        log = self.query_one(Log)
        self.history.append({"role": "user", "content": message})
        log.write(f"🐆 {message}", markup=False)
        reply = ""
        try:
            with requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": self.history,
                    "stream": True,
                    "max_tokens": 2048,
                },
                timeout=600,
                stream=True,
            ) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    delta = (
                        (chunk.get("choices") or [{}])[0].get("delta") or {}
                    ).get("content")
                    if delta:
                        reply += delta
                        log.write(delta, markup=False)
        except Exception as e:  # noqa: BLE001 — surface, don't crash the app
            log.write(f"\n[error] {e}\n", markup=False)
            return
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        log.write("\n", markup=False)

    def _poll_status(self) -> dict | None:
        import requests

        try:
            r = requests.get(f"{self.base_url}/v1/status", timeout=2)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _status_loop(self):
        while True:
            label = self.query_one(Label)
            label.update(_status_line(self._poll_status()))
            time.sleep(1.5)


def run(args) -> None:
    """CLI entry: ``qmlx tui`` / ``qmlx chat --tui``.

    ``textual`` is guaranteed importable here — the CLI calls
    :func:`ensure_tui_available` before dispatching to this function.
    """
    base_url = _resolve_base_url(args)
    model = getattr(args, "model", None) or "qwen3.5-4b-4bit"
    system = getattr(args, "system", None)
    QMLXTUI(base_url, model, system=system).run()
