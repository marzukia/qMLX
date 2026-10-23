# SPDX-License-Identifier: Apache-2.0
"""Textual TUI for qMLX (issue #51, phase 4 — optional ``[tui]`` extra).

Opt-in surface: ``qmlx tui [model]`` / ``qmlx chat --tui``. The base
install has zero TUI dependencies; the CLI runs its textual-import check
(``vllm_mlx.cli._tui_hint_or_exit`` — one-line install hint + ``exit(1)``)
BEFORE importing this module, which pulls textual at top level (AC9).

The app is deliberately small: a header, a scrollable conversation log,
an input box, and a status strip that polls ``/v1/status`` on a worker
thread. Chat messages stream from ``/v1/chat/completions`` so the reply
grows token-by-token in the log.
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, Label, Log

#: One-line hint printed (and the exit code returned) when the ``tui``
#: extra is missing — pinned by tests (AC9).
TUI_INSTALL_HINT = "TUI extra not installed — run: pip install 'qmlx-serve[tui]'"

#: Empty-state starter prompts (issue #63.3): the Log used to sit at
#: ~20 blank rows until the first message. These are example prompts
#: the operator can copy/type as-is; they are plain text, not sent.
STARTER_PROMPTS = (
    "> summarize the latest commit on git log",
    "> explain how disk KV restore works in vllm_mlx",
    "> draft release notes for the current version",
)


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
    """Plain one-line status from a /v1/status payload (AC6 fields).

    ASCII only (issue #63.1): no emoji/middot glyphs — a terminal font
    without coverage renders them as tofu boxes.
    """
    if stats is None:
        return "  qmlx | server not reachable"
    model = stats.get("model") or "-"
    state = stats.get("status", "idle")
    line = f"  qmlx | {model} | {state} | run {stats.get('num_running', 0)}"
    tps = stats.get("generation_tps") or 0.0
    if tps > 0:
        line += f" | {tps:.1f} tok/s"
    up = stats.get("uptime_s")
    if up is not None:
        line += f" | up {int(up)}s"
    return line


def _session_banner(model: str, base_url: str) -> str:
    """Empty-state banner (issue #63.3): session line + starter prompts,
    so an empty session never shows ~20 blank Log rows."""
    return (
        f"-- qMLX session | model: {model} | endpoint: {base_url}/v1\n"
        "-- starter prompts (type your own):\n"
        + "\n".join(f"   {p}" for p in STARTER_PROMPTS)
        + "\n"
    )


class QMLXTUI(App):
    """Chat TUI: streaming completions + a 1.5s-poll /v1/status strip."""

    #: One title for the whole surface (issue #63.2): the window title,
    #: the in-app Header and (before #63) the clock were three separate
    #: title-ish rows. The Header renders ``App.TITLE`` verbatim and
    #: keeps its clock off, so the operator sees exactly one title row.
    TITLE = "QMLX TUI"

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
        # ASCII icon: the Header default ("⭘") is a tofu box on
        # non-emoji fonts (issue #63.1). The title comes from App.TITLE.
        yield Header(icon=" ")
        yield Log(auto_scroll=True)
        yield Label(_status_line(None), id="qmlx-status")
        yield Input(
            placeholder=f"Message {self.model}...  (Ctrl-Q quits)",
            id="qmlx-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one(Log)
        label = self.query_one(Label)
        log.write(_session_banner(self.model, self.base_url))
        # Pass the loop UN-CALLED and run it in a worker THREAD — calling
        # ``self._status_loop()`` here ran the blocking while-True loop in
        # place on the app thread, so on_mount never returned and the app
        # froze at startup (t258). ``run_worker`` in textual 0.89 takes no
        # *args, so the widget reference is bound via ``partial``.
        self.run_worker(
            partial(self._status_loop, label), group="tui", exclusive=False, thread=True
        )
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        # Same contract as on_mount: pass the worker UN-CALLED with
        # thread=True. ``self._send(text)`` used to run the blocking HTTP
        # stream inline on the app thread and feed ``run_worker`` its
        # ``None`` result, which raised a WorkerError and exited the app
        # after the first message (default exit_on_error=True). Widgets are
        # captured on the app thread; ``run_worker`` takes no *args, so the
        # turn args are bound via ``partial``.
        self.run_worker(
            partial(self._send, text, self.query_one(Log)),
            group="tui",
            exclusive=False,
            thread=True,
        )

    def _write(self, log: Log, msg: str) -> None:
        """Append one line to the conversation log, from any thread.

        Worker threads must not touch widgets directly — the write is
        serialized onto the app thread via ``call_from_thread``.
        """
        self.call_from_thread(log.write, msg)

    def _send(self, message: str, log: Log) -> None:
        """Run one chat turn (streamed) in a worker thread.

        ``log`` is captured on the app thread before the worker starts;
        all writes go through ``_write`` → ``call_from_thread``.
        """
        import requests

        self.history.append({"role": "user", "content": message})
        self._write(log, f"> {message}")
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
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get(
                        "content"
                    )
                    if delta:
                        reply += delta
                        self._write(log, delta)
        except Exception as e:  # noqa: BLE001 — surface, don't crash the app
            self._write(log, f"\n[error] {e}\n")
            return
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        self._write(log, "\n")

    def _poll_status(self) -> dict | None:
        import requests

        try:
            r = requests.get(f"{self.base_url}/v1/status", timeout=2)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _status_loop(self, label: Label) -> None:
        """Poll /v1/status every ~1.5s — runs in a worker thread.

        The thread only sleeps and does HTTP; the widget update is
        serialized onto the app thread via ``call_from_thread``.
        """
        while self.is_running:
            try:
                self.call_from_thread(label.update, _status_line(self._poll_status()))
            except Exception:  # noqa: BLE001 — app gone: stop the loop quietly
                return
            time.sleep(1.5)


def run(args) -> None:
    """CLI entry: ``qmlx tui`` / ``qmlx chat --tui``.

    ``textual`` is guaranteed importable here — the CLI ran its
    ``_tui_hint_or_exit`` check (hint + exit 1) before importing this
    module and dispatching to this function.
    """
    base_url = _resolve_base_url(args)
    model = getattr(args, "model", None) or "qwen3.5-4b-4bit"
    system = getattr(args, "system", None)
    QMLXTUI(base_url, model, system=system).run()
