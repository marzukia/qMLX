# SPDX-License-Identifier: Apache-2.0
"""Worker-thread pin for the qMLX TUI (t258 defect, textual 0.89.1).

Diagnosed in t257 §2 / t258 against textual 0.89.1 source:

1. ``on_mount`` called ``run_worker(self._status_loop(), ...)`` — the
   blocking ``while True`` loop ran *in place* on the app thread before
   ``run_worker`` was ever reached, so ``on_mount`` never returned and
   the app froze at startup (Input never focused, keys never processed).
2. ``on_input_submitted`` called ``run_worker(self._send(text), ...)`` —
   the blocking HTTP stream ran inline on the app thread, and the
   ``None`` it returned was fed to ``run_worker``, raising a WorkerError
   that exited the app after the first message (``exit_on_error=True``).

Pins (hermetic: ``App.run_test()`` with stubbed ``_send`` and
``_poll_status`` — no real HTTP, no model server):
- mount completes and the Input is focused (a blocking ``on_mount``
  hangs ``run_test``'s ready-wait forever, so each test carries a
  wall-clock guard);
- a submitted message round-trips back into the log through
  ``call_from_thread`` and the app stays alive (a WorkerError would
  have exited it);
- the status worker updates the strip in a thread without blocking
  mount.
"""

from __future__ import annotations

import asyncio
import json as _json

import pytest

pytest.importorskip("textual")

from textual.widgets import Footer, Header, Input, Label, Log  # noqa: E402

from vllm_mlx import tui as _tui  # noqa: E402

#: Wall-clock guard: run_test's ready-wait has no timeout of its own, so a
#: blocking on_mount must fail the test (not hang the suite).
_TIMEOUT = 20.0


def _log_text(app) -> str:
    """Plain text of the conversation Log (0.89 stores lines as str)."""
    return "".join(app.query_one(Log).lines)


@pytest.fixture
def tui_app(monkeypatch):
    """QMLXTUI with all network I/O stubbed (no real HTTP, no server)."""
    app = _tui.QMLXTUI("http://127.0.0.1:1", model="test-model")
    monkeypatch.setattr(
        _tui.QMLXTUI,
        "_poll_status",
        lambda self: {
            "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "status": "idle",
            "num_running": 0,
        },
    )
    # Capture the REAL _send before the class attribute is replaced, so
    # tests that exercise the genuine streaming path (fake requests
    # module) can restore it on the instance.
    app._real_send = _tui.QMLXTUI._send  # noqa: SLF001

    def _stub_send(self, message: str, log) -> None:
        # Mirrors the worker contract: the turn runs in a worker thread
        # and its UI write comes back through call_from_thread.
        self.call_from_thread(log.write, f"stub-reply to: {message}")

    monkeypatch.setattr(_tui.QMLXTUI, "_send", _stub_send)
    return app


def test_mount_completes_and_input_focused(tui_app):
    """on_mount must RETURN — a blocking ``self._status_loop()`` call
    (the t258 freeze) would hang ``run_test``'s ready-wait forever."""

    async def body() -> None:
        async with tui_app.run_test() as pilot:
            await pilot.pause()
            assert tui_app.is_running
            assert tui_app.focused is tui_app.query_one(Input)
            # The mount banner proves on_mount ran past the run_worker
            # call and reached the focus line.
            assert "model: test-model" in _log_text(tui_app)

    asyncio.run(asyncio.wait_for(body(), timeout=_TIMEOUT))


def test_submitted_message_round_trips_via_call_from_thread(tui_app):
    """A submitted message runs its (stubbed) turn in a worker thread;
    the reply posts back into the Log through ``call_from_thread`` and
    the app stays alive — a WorkerError would have exited it."""

    async def body() -> None:
        async with tui_app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("h", "e", "l", "l", "o")
            await pilot.press("enter")
            for _ in range(200):
                if "stub-reply to: hello" in _log_text(tui_app):
                    break
                await pilot.pause()
            assert "stub-reply to: hello" in _log_text(tui_app)
            assert tui_app.is_running

    asyncio.run(asyncio.wait_for(body(), timeout=_TIMEOUT))


def test_empty_session_shows_starter_block(tui_app):
    """An empty session shows a starter block, not ~20 blank rows
    (issue #63.3): the session line + example prompts land in the Log
    at mount."""

    async def body() -> None:
        async with tui_app.run_test() as pilot:
            await pilot.pause()
            text = _log_text(tui_app)
            assert "model: test-model" in text
            assert "starter prompts" in text
            for prompt in _tui.STARTER_PROMPTS:
                assert prompt in text
            assert tui_app.is_running

    asyncio.run(asyncio.wait_for(body(), timeout=_TIMEOUT))


def test_single_title_line(tui_app):
    """One title row (issue #63.2): window title == in-app Header text,
    and the Header clock is off — the old layout showed the window
    title, a second in-app 'QMLXTUI' header row, and a clock."""

    async def body() -> None:
        async with tui_app.run_test() as pilot:
            await pilot.pause()
            assert tui_app.TITLE == "QMLX TUI"
            header = tui_app.query_one(Header)
            # textual 0.89: the clock is opt-in (show_clock flag, kept in
            # ``_show_clock``); the old Header("...") call passed the
            # title string positionally and switched it on.
            assert header._show_clock is False  # noqa: SLF001
            assert header.screen_title == tui_app.TITLE
            assert tui_app.is_running

    asyncio.run(asyncio.wait_for(body(), timeout=_TIMEOUT))


def test_quit_affordance_visible(tui_app):
    """The quit key must be VISIBLE (issue #63.5): the binding is
    show=True so the Footer lists it, and ctrl+q (not a bare 'q', which
    the focused Input swallows) actually quits while typing."""

    async def body() -> None:
        async with tui_app.run_test() as pilot:
            await pilot.pause()
            quit_bindings = [b for b in tui_app.BINDINGS if b.action == "quit"]
            assert len(quit_bindings) == 1
            binding = quit_bindings[0]
            assert binding.show is True
            assert binding.key == "ctrl+q"
            from textual.widgets._footer import FooterKey  # noqa: PLC0415

            keys = {
                fk.key_display: fk.description
                for fk in tui_app.query_one(Footer).query(FooterKey)
            }
            # The Footer renders the ctrl+q binding as "^q".
            assert keys.get("^q") == "Quit"
            await pilot.press("ctrl+q")
        assert not tui_app.is_running

    asyncio.run(asyncio.wait_for(body(), timeout=_TIMEOUT))


def test_streaming_indicator_in_log(tui_app, monkeypatch):
    """While generating, the Log itself shows the turn is in flight
    (issue #63.6 — before, only the status strip said so): the real
    ``_send`` writes the user line, a 'generating...' line, then the
    first streamed token REPLACES the indicator and the reply stays
    one clean line. The fake ``requests`` module delays its first SSE
    chunk so the indicator is observable."""
    import sys as _sys
    import time as _time
    import types as _types

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            _time.sleep(0.3)  # time-to-first-token: indicator is visible
            for token in ("Hel", "lo!"):
                chunk = {"choices": [{"delta": {"content": token}}]}
                yield b"data: " + _json.dumps(chunk).encode()
            yield b"data: [DONE]"

    def _fake_post(*a, **k):
        return _FakeResp()

    fake_requests = _types.ModuleType("requests")
    fake_requests.post = _fake_post
    monkeypatch.setitem(_sys.modules, "requests", fake_requests)
    # Rebind the captured (unbound) method to this instance.
    tui_app._send = _types.MethodType(tui_app._real_send, tui_app)  # noqa: SLF001

    async def body() -> None:
        async with tui_app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("h", "e", "l", "l", "o")
            await pilot.press("enter")
            saw_indicator = False
            for _ in range(400):
                text = _log_text(tui_app)
                if "generating..." in text:
                    saw_indicator = True
                if "Hello!" in text:
                    break
                await pilot.pause()
            text = _log_text(tui_app)
            assert saw_indicator  # indicator was visible before first token
            assert "Hello!" in text
            assert "generating..." not in text  # replaced by the reply
            assert "> hello" in text  # user line, on its own line
            assert tui_app.is_running

    asyncio.run(asyncio.wait_for(body(), timeout=_TIMEOUT))


def test_headless_screenshot_no_tofu(tmp_path):
    """A 100x30 headless screenshot of the idle TUI renders without tofu
    (issue #63.1). The SVG from ``App.save_screenshot`` contains the
    screen's text; any non-ASCII codepoint outside the box-drawing /
    block-elements ranges is a tofu candidate on a basic terminal font
    (emoji, dingbats, CJK…). Box drawing (U+2500-257F) and block
    elements (U+2580-259F) are whitelisted: the SVG renderer draws the
    focused Input's cursor/underline with them, and they are part of
    standard console fonts (not the tofu-prone symbol sets).
    The HF repo id in the stubbed /v1/status must not leak into the
    strip (issue #63.4), which the assertion also covers."""
    app = _tui.QMLXTUI("http://127.0.0.1:1", model="test-model")
    app._poll_status = lambda self: {  # noqa: SLF001 — hermetic stub
        "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "status": "idle",
        "num_running": 0,
    }
    screenshot = tmp_path / "tui_100x30.svg"

    async def body() -> None:
        async with app.run_test(headless=True, size=(100, 30)) as pilot:
            await pilot.pause()
            app.save_screenshot(str(screenshot))
            assert app.is_running

    asyncio.run(asyncio.wait_for(body(), timeout=_TIMEOUT))
    assert screenshot.exists()
    text = screenshot.read_text(encoding="utf-8")
    tofu = sorted(
        {ch for ch in text if ord(ch) > 127 and not (0x2500 <= ord(ch) <= 0x259F)}
    )
    assert tofu == [], f"tofu candidates in 100x30 screenshot: {tofu!r}"
    # Strip density pin (issue #63.4): the alias, never the HF repo id.
    assert "test-model" in text
    assert "mlx-community" not in text


def test_status_line_uses_alias_not_hf_path():
    """Strip shape (issue #63.4): the alias the operator launched with,
    never the full HF repo id from /v1/status, and a lighter layout
    (no run count / uptime; tok/s only while generating)."""
    assert (
        _tui._status_line(None, "qwen3.5-4b-4bit") == "qmlx | qwen3.5-4b-4bit | offline"
    )
    stats = {
        "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "status": "running",
        "num_running": 2,
        "generation_tps": 12.5,
        "uptime_s": 30.0,
    }
    line = _tui._status_line(stats, "qwen3.5-4b-4bit")
    assert line == "qmlx | qwen3.5-4b-4bit | running | 12.5 tok/s"
    assert "mlx-community" not in line
    assert "run 2" not in line
    assert "up 30s" not in line
    idle = _tui._status_line({"model": "x", "status": "idle"}, "qwen3.5-4b-4bit")
    assert idle == "qmlx | qwen3.5-4b-4bit | idle"


def test_status_loop_does_not_block_mount(tui_app):
    """The status worker must run in a thread: mount completes (already
    proven by test 1) and the first poll refreshes the strip — the
    initial "offline" line is replaced by the stubbed status without
    holding the app thread. The strip shows the launched alias, not
    the full HF repo id the stubbed /v1/status reports (issue #63.4)."""

    async def body() -> None:
        async with tui_app.run_test() as pilot:
            await pilot.pause()
            label = tui_app.query_one(Label)
            for _ in range(400):
                if "test-model | idle" in str(label.visual):
                    break
                await pilot.pause()
            assert "test-model | idle" in str(label.visual)
            assert "mlx-community" not in str(label.visual)
            assert "offline" not in str(label.visual)
            assert tui_app.is_running

    asyncio.run(asyncio.wait_for(body(), timeout=_TIMEOUT))
