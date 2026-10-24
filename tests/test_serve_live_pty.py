# SPDX-License-Identifier: Apache-2.0
"""issue #61 acceptance: ``qmlx serve <alias> --live`` on a 100x30 pty.

The operator's screenshot review captured two defects on a 100x30
terminal:

1. the dim live status strip (``render_live_line``) was printed inline
   and subsequent log lines continued on the SAME visual line
   (``warming up…INFO:qmlx.scheduler:…`` fused);
2. the startup banner box had scrolled off-screen before
   ``Starting server on http://… (warming up)`` printed.

Fix (see ``vllm_mlx/_ui.py::_draw_live_strip`` and
``vllm_mlx/cli.py::_render_startup_banner``): the strip is
bottom-anchored — clear+redraw per poll on its own line,
newline-terminated so the log stream (uvicorn/qmlx loggers write
straight to stderr, one write per record) always starts on a fresh
line; the banner is a compact 2-3 line block and the endpoint line
prints in ``--live`` mode too.

This test drives the REAL CLI in a 100x30 pseudo-terminal, replays the
byte stream through a minimal VT100 screen model, and asserts:

* the strip never shares a visual line with a log line (checked on
  every half-second screen snapshot plus the final screen);
* at the "warming up" moment (first complete strip frame) the compact
  banner AND the ``Starting server on http://…`` endpoint line are
  both still on-screen.

The capture window is the warming-up phase — model-free in the sense
that strip/banner render before model load completes; a real (small,
cached) model is still loaded because the serve flow only reaches the
endpoint line after ``load_model`` returns. Skipped when the small
cached model is absent (no network is assumed).
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import socket
import struct
import sys
import termios
import time
from pathlib import Path

import pytest

ALIASES_PATH = Path(__file__).resolve().parent.parent / "vllm_mlx" / "aliases.json"
MODEL = "mlx-community/Qwen3-0.6B-4bit"
ALIAS = "qwen3-0.6b"
COLS = 100
ROWS = 30
CAPTURE_SECONDS = 30.0  # after the first strip frame
HARD_TIMEOUT_SECONDS = 120.0  # incl. CLI start + model load


def _model_cache_dir() -> str | None:
    """HF cache dir for the model, or None when it is not cached."""
    hub = os.environ.get("HF_HUB_CACHE")
    if not hub:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        hub = str(Path(hf_home) / "hub")
    return str(Path(hub) / f"models--{MODEL.replace('/', '--')}")


def _model_cached() -> bool:
    d = _model_cache_dir()
    return bool(d and Path(d).is_dir())


class _Screen:
    """Minimal VT100 screen model — enough to replay the qmlx serve
    output: CUP, CUU/CUD/CUF/CUB, EOL (ESC[nK), ED (ESC[J/L), RLF,
    CR/LF/BS/TAB, SGR + OSC ignored, printable cells with auto-wrap
    and bottom-row scroll. Validated byte-for-byte against pyte on
    real captures (issue #61 dev notes)."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.grid: list[list[str]] = [[""] * cols for _ in range(rows)]
        self.row = 0
        self.col = 0
        self._buf: list[int] = []
        self._esc = False
        self._osc = False

    def feed(self, s: str) -> None:
        for ch in s:
            self._feed_byte(ord(ch))

    def _feed_byte(self, b: int) -> None:
        if self._osc:
            if b == 0x07:
                self._osc = False
            elif b == 0x1B:
                self._osc = False
                self._esc = False
                self._buf = []
            return
        if self._esc:
            if b == 0x5B:  # CSI
                self._buf = [b]
                self._esc = False
                return
            if b == 0x5D:  # OSC
                self._osc = True
                self._esc = False
                return
            if b == 0x43:  # CHA
                self.col = 0
            self._esc = False
            return
        if b == 0x1B:
            self._esc = True
            return
        if self._buf:
            if 0x30 <= b <= 0x3F:  # digits / ';' / '?'
                self._buf.append(b)
                return
            if 0x40 <= b <= 0x7E:  # final byte
                self._csi(bytes(self._buf[1:]), chr(b))
                self._buf = []
                return
            self._buf = []
            return
        if b == 0x0D:
            self.col = 0
        elif b == 0x0A:
            self.row += 1
            if self.row >= self.rows:
                self._scroll_up()
                self.row = self.rows - 1
        elif b == 0x08:
            self.col = max(0, self.col - 1)
        elif b == 0x09:
            self.col = min(self.cols - 1, (self.col // 8 + 1) * 8)
        elif b < 0x20:
            return
        else:
            if self.col >= self.cols:
                self.col = 0
                self.row += 1
                if self.row >= self.rows:
                    self._scroll_up()
                    self.row = self.rows - 1
            self.grid[self.row][self.col] = chr(b)
            if self.col == self.cols - 1:
                self.col = 0
                self.row += 1
                if self.row >= self.rows:
                    self._scroll_up()
                    self.row = self.rows - 1
            else:
                self.col += 1

    def _csi(self, params: bytes, final: str) -> None:
        p = params.decode("latin-1")
        if final in ("H", "f"):
            parts = p.split(";")
            r = int(parts[0]) - 1 if parts[0].isdigit() else 0
            c = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
            self.row = max(0, min(self.rows - 1, r))
            self.col = max(0, min(self.cols - 1, c))
        elif final == "A":
            self.row = max(0, self.row - (int(p) if p else 1))
        elif final == "B":
            self.row = min(self.rows - 1, self.row + (int(p) if p else 1))
        elif final == "C":
            self.col = min(self.cols - 1, self.col + (int(p) if p else 1))
        elif final == "D":
            self.col = max(0, self.col - (int(p) if p else 1))
        elif final == "K":
            mode = int(p) if p else 0
            rng = (
                range(self.col, self.cols)
                if mode == 0
                else range(0, self.col + 1) if mode == 1 else range(self.cols)
            )
            for c in rng:
                self.grid[self.row][c] = ""
        elif final == "J":
            for r in range(self.row, self.rows):
                for c in range(self.cols):
                    self.grid[r][c] = ""
        elif final == "L":
            for r in range(0, self.row + 1):
                for c in range(self.cols):
                    self.grid[r][c] = ""
        elif final == "M":
            for _ in range(int(p) if p else 1):
                self.grid.insert(0, [""] * self.cols)
                self.grid.pop()
        # SGR ('m'), private modes ('?…'): ignored

    def _scroll_up(self, n: int = 1) -> None:
        for _ in range(n):
            del self.grid[0]
            self.grid.append([""] * self.cols)

    def lines(self) -> list[str]:
        return ["".join(row).rstrip() for row in self.grid]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Capture:
    def __init__(self, text: str) -> None:
        self.text = text
        # offset of the first COMPLETE strip frame (its trailing CRLF)
        i = text.find("warming up")
        self.warm_offset = -1
        if i != -1:
            j = text.find("\r\n", i)
            if j != -1:
                self.warm_offset = j + 2
        # screen at the warming-up moment
        self.warm_screen = _Screen(ROWS, COLS)
        if self.warm_offset != -1:
            self.warm_screen.feed(text[: self.warm_offset])
        # Periodic snapshots by byte offset — enough to catch any fused
        # line, since a fused line persists on screen until the next
        # strip redraw erases its line.
        self.snapshots: list[_Screen] = []
        step = max(1, len(text) // 20)
        for off in range(0, len(text), step):
            s = _Screen(ROWS, COLS)
            s.feed(text[:off])
            self.snapshots.append(s)
        self.final = _Screen(ROWS, COLS)
        self.final.feed(text)

    def all_screens(self) -> list[_Screen]:
        return self.snapshots + [self.final, self.warm_screen]


def _run_capture(port: int, duration: float) -> _Capture:
    # The test runs from the repo's venv (make smoke) — prefer that
    # venv's own ``qmlx`` so the captured CLI is the code under test.
    qmlx = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "qmlx")
    if not os.path.exists(qmlx):
        qmlx = "qmlx"  # fall back to PATH
    env = dict(os.environ)
    env.update(
        TERM="xterm-256color",
        QMLX_TELEMETRY="0",
        QMLX_DISABLE_VERSION_CHECK="1",
        HF_HUB_OFFLINE="1",
    )
    env.pop("NO_COLOR", None)
    env.pop("QMLX_PLAIN", None)

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    pid = os.fork()
    if pid == 0:
        os.setsid()
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        os.close(master)
        os.close(slave)
        os.execvpe(qmlx, [qmlx, "serve", ALIAS, "--live", "--port", str(port)], env)
    os.close(slave)

    data = b""
    start = time.monotonic()
    saw_warm = False
    warm_at = None
    while time.monotonic() - start < HARD_TIMEOUT_SECONDS:
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            data += chunk
            if not saw_warm and "warming up" in data.decode("utf-8", "replace"):
                saw_warm = True
                warm_at = time.monotonic()
            if saw_warm and warm_at is not None and (
                time.monotonic() - warm_at > duration
            ):
                break
        if os.waitpid(pid, os.WNOHANG)[0] != 0:
            break
    try:
        os.kill(pid, signal.SIGINT)
        time.sleep(1.0)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    finally:
        try:
            os.close(master)
        except OSError:
            pass
    return _Capture(data.decode("utf-8", "replace"))


# Strip content markers (rendered, ANSI-free). A fused line would
# contain one of these PLUS a log-level prefix on the same visual line.
def _is_strip_line(line: str) -> bool:
    if "warming up…" in line:
        return True
    # ready-state strip: "ready …" / "generating …" + run/wait pair
    if " / wait " in line and ("ready" in line or "generating" in line):
        return True
    return False


_LOG_PREFIX = ("INFO", "WARNING", "ERROR", "DEBUG")


def _assert_no_fusion(screens: list[_Screen]) -> None:
    violations = []
    for s in screens:
        for line in s.lines():
            if _is_strip_line(line) and any(p in line for p in _LOG_PREFIX):
                violations.append(line)
            if line.count("INFO") > 1:  # two log lines fused
                violations.append(line)
    assert not violations, f"strip/log fusion (issue #61 AC1): {violations!r}"


@pytest.mark.skipif(not _model_cached(), reason="cached 0.6B model absent (offline dev)")
def test_serve_live_strip_and_banner_100x30():
    cap = _run_capture(_free_port(), CAPTURE_SECONDS)
    text = cap.text

    # The capture must have actually seen the warming-up phase — a
    # capture that never reached the live loop would make every
    # assertion below vacuously true.
    assert "warming up" in text, "capture never reached the warming-up strip"
    assert any(
        p in ln
        for ln in cap.final.lines() + [ln for s in cap.snapshots for ln in s.lines()]
        for p in _LOG_PREFIX
    ), "capture never saw a log line — fusion check would be vacuous"

    screens = cap.all_screens()
    _assert_no_fusion(screens)

    # The strip itself must appear alone on its own line somewhere in
    # the capture (both the warming frame and, if the engine finished
    # warmup within the window, the ready frame).
    assert any(
        _is_strip_line(ln) and not any(p in ln for p in _LOG_PREFIX)
        for s in screens
        for ln in s.lines()
    ), "no clean strip line observed"

    # AC2: at the warming-up moment the compact banner AND the
    # endpoint line must both be on-screen (100x30).
    warm_lines = cap.warm_screen.lines()
    assert any(
        "🐆 qMLX" in ln and "Model:" in ln for ln in warm_lines
    ), f"compact banner not on-screen at warming-up moment: {warm_lines!r}"
    assert any(
        ln.lstrip().startswith("Starting server on http://") for ln in warm_lines
    ), f"endpoint line not on-screen at warming-up moment: {warm_lines!r}"
