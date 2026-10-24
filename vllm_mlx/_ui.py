# SPDX-License-Identifier: Apache-2.0
"""Terminal-UI facade for qMLX CLI commands (issue #51).

One rendering layer with two modes:

* **rich** — ``rich`` output (titled panels, tables, spinners, live
  status strips). Selected when stdout is a TTY, ``NO_COLOR`` is unset,
  ``QMLX_PLAIN`` is not ``"1"``, ``TERM`` is not ``"dumb"``, and ``rich``
  is importable.
* **plain** — the hand-rolled layout code that predates this module.
  Byte-identical to the pre-#51 output: zero ANSI bytes, same spacing.
  Chosen for pipes, CI, ``NO_COLOR``, ``QMLX_PLAIN=1``, dumb terminals,
  and minimal installs where ``rich`` is missing.

Contract (issue #51 "invariants"):

* In plain mode nothing this module renders may contain ``\\x1b``.
* Logging (``logger.*``) stays untouched — rich output goes to stdout
  via a ``rich.Console``; spinner/status lines go to stderr.
* faulthandler dumps continue to reach stderr (never the Console).

The facade is deliberately small: call sites keep their plain-mode
strings and ask the facade for a rich render when ``ui.rich``. That is
what keeps the plain path auditable line-by-line.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

__all__ = [
    "rich",
    "is_rich",
    "emit",
    "emit_err",
    "panel",
    "rich_table",
    "plain_table",
    "spinner",
    "progress",
    "live",
    "render_live_line",
    "render_status_table",
    "status_kv_rows",
]

_rich_importable: bool | None = None


def rich_available() -> bool:
    """True when the ``rich`` package is importable in this install."""
    global _rich_importable
    if _rich_importable is None:
        try:
            import rich  # noqa: F401

            _rich_importable = True
        except Exception:  # pragma: no cover - rich is a base dep now
            _rich_importable = False
    return _rich_importable


def is_rich() -> bool:
    """Decide the render mode for this invocation (issue #51).

    Plain mode wins when ANY of: ``QMLX_PLAIN=1``, ``NO_COLOR`` set,
    ``TERM=dumb``, stdout is not a TTY, or ``rich`` is not installed.
    The check is re-evaluated on every call (cheap) so tests that swap
    ``sys.stdout`` or env vars do not need cache invalidation.
    """
    if os.environ.get("QMLX_PLAIN") == "1":
        return False
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    if not rich_available():
        return False
    return True


#: Live module-level flag, read the way the issue describes: ``if ui.rich``.
# Bound to a callable via a small class so ``ui.rich`` stays attribute-style
# while still re-evaluating per call.
class _RichFlag:
    def __bool__(self) -> bool:
        return is_rich()


rich = _RichFlag()


def _console():
    from rich.console import Console

    return Console()


def _stderr_console():
    from rich.console import Console

    return Console(stderr=True)


def emit(text: str = "", *, markup: bool = False) -> None:
    """Print one line of operator-facing output.

    plain: ``print(text)`` — byte-identical to the pre-facade output.
    rich: the same line through a ``Console`` with markup and highlight
    disabled so existing strings (arrows, dashes, ``**``) pass through
    verbatim; the Console adds only terminal-aware width handling.
    ``markup=True`` (issue #61 compact banner) enables rich markup on
    the TTY path; plain mode always ignores it.
    """
    if is_rich():
        _console().print(text, markup=markup, highlight=False, soft_wrap=False)
    else:
        print(text)


def emit_err(text: str = "") -> None:
    """Like :func:`emit` but on stderr (status/spinner side-channel)."""
    if is_rich():
        _stderr_console().print(text, markup=False, highlight=False)
    else:
        print(text, file=sys.stderr)


def panel(
    title: str,
    body: Sequence[str],
    *,
    plain_lines: Sequence[str],
    style: str = "cyan",
    subtitle: str | None = None,
) -> None:
    """Render a titled block.

    rich: a ``rich.panel.Panel`` titled ``title`` containing ``body``
    lines (markup allowed in ``body``).
    plain: prints ``plain_lines`` exactly — the pre-#51 byte-for-byte
    layout for this call site. Callers pass both; the plain list is
    what the legacy code used to ``print``.
    """
    if is_rich():
        from rich.panel import Panel

        _console().print(
            Panel(
                "\n".join(body),
                title=title,
                subtitle=subtitle,
                style=style,
                expand=False,
                padding=(0, 1),
            )
        )
    else:
        for line in plain_lines:
            print(line)


def rich_table(
    headers: Sequence[str | tuple[str, ...]],
    rows: Sequence[Sequence[str]],
    *,
    title: str | None = None,
    style: str = "cyan",
    header_style: str = "bold",
    footer: Sequence[str] | None = None,
) -> None:
    """Render a ``rich.table.Table`` on a TTY.

    ``headers`` entries may be plain strings or ``(name, hint)`` tuples
    (the hint is dropped — callers that share a plain/rich header spec
    keep one source of truth). ``footer`` is an optional list of hint
    lines rendered below the table.

    Plain-mode callers should instead use :func:`plain_table` (fixed
    widths) or keep their legacy hand-rolled layout.
    """
    from rich.table import Table

    table = Table(title=title, title_style=style, header_style=header_style,
                  show_lines=False, pad_edge=False)
    for h in headers:
        name = h[0] if isinstance(h, tuple) else h
        table.add_column(name, overflow="fold")
    for row in rows:
        table.add_row(*row)
    _console().print(table)
    if footer:
        for line in footer:
            _console().print(f"[dim]{line}[/dim]")


def plain_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Fixed-width plain table — used by NEW commands (``status``) whose
    plain output has no pre-#51 bytes to stay compatible with."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = [
        "  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    ]
    out.append(
        "  " + "  ".join("-" * widths[i] for i in range(len(headers))).rstrip()
    )
    for row in rows:
        out.append(
            "  "
            + "  ".join(
                str(c).ljust(widths[i]) for i, c in enumerate(row)
            ).rstrip()
        )
    return "\n".join(out)


@contextmanager
def spinner(text: str, *, interval: float = 0.1) -> Iterator[None]:
    """Loading indicator on stderr.

    rich: a ``rich`` status line (dots spinner) on the stderr console.
    plain: the legacy ANSI frame spinner when stderr is a TTY (byte-
    identical to the pre-facade ``_wait_for_chat_server`` behavior), or
    a no-op on non-TTY stderr.
    """
    import threading
    import time

    if is_rich():
        with _stderr_console().status(f"[cyan]{text}[/cyan]"):
            yield
        return

    # Plain mode: replicate the legacy frames exactly.
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    active = sys.stderr.isatty()
    cyan = "\x1b[36m" if active else ""
    dim = "\x1b[2m" if active else ""
    reset = "\x1b[0m" if active else ""
    state = {"tick": 0, "stop": False, "start": time.monotonic()}

    def _draw() -> None:
        if not active:
            return
        while not state["stop"]:
            if state["tick"] % 10 == 0:
                elapsed = int(time.monotonic() - state["start"])
                ch = frames[state["tick"] % len(frames)]
                sys.stderr.write(
                    f"\r  {cyan}{ch}{reset} {text} {dim}{elapsed}s{reset}"
                )
                sys.stderr.flush()
            state["tick"] += 1
            time.sleep(interval)
        if active:
            sys.stderr.write("\r" + " " * 40 + "\r")
            sys.stderr.flush()

    t = threading.Thread(target=_draw, daemon=True)
    t.start()
    try:
        yield
    finally:
        state["stop"] = True
        t.join()


@contextmanager
def progress(total: float = 100.0, description: str = "") -> Iterator[None]:
    """Multi-step progress for model download/load/warmup.

    rich: a ``rich.progress`` bar on stderr; plain: a no-op (the legacy
    code printed nothing here — HF's own tqdm already shows download
    progress, and piped output must stay clean).
    """
    if is_rich():
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )

        with Progress(
            TextColumn(f"[bold]{description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=_stderr_console(),
            transient=False,
        ) as prog:
            task = prog.add_task("working", total=total)
            yield task
            prog.update(task, completed=total)
    else:
        yield None


def render_live_line(stats: dict | None) -> str:
    """One-line live status strip (rich markup) from a stats dict.

    ``stats is None`` = engine not up yet ("warming up…"). ``status``
    of ``"idle"`` renders as ``ready`` (the strip only polls after
    ``cfg.ready`` flipped, so idle ⇒ the port is accepting traffic).
    """
    if stats is None:
        return "[dim]warming up…[/dim]"
    parts = []
    state = stats.get("status", "idle")
    if state == "generating":
        parts.append("[bold]generating[/bold]")
    else:
        parts.append("[bold green]ready[/bold green]")
    model = stats.get("model")
    if model:
        parts.append(f"[cyan]{model}[/cyan]")
    endpoint = stats.get("endpoint")
    if endpoint:
        parts.append(f"[cyan]{endpoint}[/cyan]")
    tps = stats.get("generation_tps")
    if tps:
        parts.append(f"[green]{tps:.1f} tok/s[/green]")
    run = stats.get("num_running", 0)
    wait = stats.get("num_waiting", 0)
    parts.append(f"[dim]run {run} / wait {wait}[/dim]")
    active_gb = stats.get("metal_active_gb")
    if active_gb:
        parts.append(f"[dim]{active_gb:.1f} GB[/dim]")
    uptime = stats.get("uptime_s")
    if uptime:
        parts.append(f"[dim]up {int(uptime)}s[/dim]")
    return "  ".join(parts)


def status_kv_rows(stats: dict) -> list[tuple[str, str]]:
    """(field, value) rows for a ``/v1/status`` payload — the single
    source of truth for ``qmlx status`` in BOTH render modes.

    Field-by-field mapping is pinned by tests (issue #51 AC6).
    """
    metal = stats.get("metal") or {}
    cache = stats.get("cache") or {"enabled": False}

    def _opt(v) -> str:
        return "-" if v is None else str(v)

    def _gb(v) -> str:
        return "-" if v is None else f"{v:.1f}"

    rows = [
        ("status", _opt(stats.get("status", "?"))),
        ("model", _opt(stats.get("model")) or "-"),
        ("uptime_s", _opt(stats.get("uptime_s"))),
        ("steps_executed", _opt(stats.get("steps_executed", 0))),
        ("num_running", _opt(stats.get("num_running", 0))),
        ("num_waiting", _opt(stats.get("num_waiting", 0))),
        ("total_requests_processed", _opt(stats.get("total_requests_processed", 0))),
        ("total_prompt_tokens", _opt(stats.get("total_prompt_tokens", 0))),
        ("total_completion_tokens", _opt(stats.get("total_completion_tokens", 0))),
        ("generation_tps", f"{stats.get('generation_tps') or 0.0:.1f}"),
        ("prompt_tps", f"{stats.get('prompt_tps') or 0.0:.1f}"),
        ("metal_active_gb", _gb(metal.get("active_memory_gb"))),
        ("metal_peak_gb", _gb(metal.get("peak_memory_gb"))),
        ("cache", "enabled" if cache.get("enabled") else "disabled"),
    ]
    return rows


def render_status_table(stats: dict) -> str:
    """Fixed-width plain rendering of a /v1/status payload (``qmlx status``).

    Plain-safe by construction: no ANSI bytes, no rich objects.
    """
    out = ["  "]
    for key, val in status_kv_rows(stats):
        out.append(f"  {key:<20} {val}")
    return "\n".join(out)


def _strip_lines(console, markup: str) -> list[str]:
    """Render a strip markup string to wrapped ANSI lines at console width.

    Used to (a) know exactly how many lines the strip occupies (it may
    wrap on narrow terminals) and (b) dedup unchanged redraws — while
    the engine warms up the strip is constant (``warming up…``) and
    redrawing it every poll would scroll the screen for nothing.
    """
    with console.capture() as capture:
        console.print(markup, markup=True, highlight=False, soft_wrap=False)
    lines = capture.get().split("\n")
    while lines and lines[-1] in ("", "\x1b[0m"):
        lines.pop()
    return lines or [""]


def _draw_live_strip(console, lines: list[str]) -> None:
    """Redraw the live status strip on its own line (issue #61).

    The strip is anchored to the BOTTOM of the screen and is the only
    writer of that line. Each poll repositions there (CUP), erases the
    line (``ESC[2K``), redraws the strip, and finishes with a newline
    so the cursor lands at column 0 of a fresh line. Log writers that
    bypass this console — uvicorn's and qmlx's loggers write straight
    to stderr, one ``write`` per record — therefore always start on a
    line of their own: the strip and the log stream can never share a
    visual line.

    The whole redraw (position + clear + text + newline) is a single
    ``write`` well under PIPE_BUF, so it is atomic with respect to
    those one-write log records: a log line lands before or after the
    strip, never inside it.

    Because the strip region always includes the bottom row, the
    trailing newline scrolls the screen one row per redraw — the
    bottom-of-screen position the issue calls for: the strip stays
    pinned to the bottom while log output scrolls above it. Redraws
    of an UNCHANGED strip are skipped by :func:`live`, so the screen
    only scrolls when the strip actually changes.

    On a non-terminal or degenerate-sized terminal (tests, pipes)
    falls back to a plain newline-terminated line — still a line of
    its own, still no ANSI, per the plain-mode contract.
    """
    _width, height = console.size
    if not console.is_terminal or height < 2:
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        return
    top = max(1, height - len(lines) + 1)
    chunk = f"\x1b[{top};1H"
    for line in lines:
        chunk += "\x1b[2K" + line + "\n"
    sys.stdout.write(chunk)
    sys.stdout.flush()


@contextmanager
def live(
    render_fn,
    *,
    interval: float = 0.5,
    stop_fn=lambda: False,
) -> Iterator[None]:
    """Polling live display for ``serve --live`` / ``status --watch``.

    ``render_fn`` returns the current renderable (rich markup string or
    plain text); ``stop_fn`` is polled each tick — when it returns
    true the display ends (e.g. the server thread exited).

    rich: a bottom-anchored status line driven on the calling thread —
    see :func:`_draw_live_strip` for the line discipline that keeps
    the strip off the log stream's lines (issue #61). Redraws of an
    unchanged strip are skipped so the screen only scrolls when the
    strip content actually changes.
    plain: ``render_fn`` is re-printed every tick (no ANSI erase
    sequences — plain output stays a clean byte stream for pipes).
    """
    import time

    if is_rich():
        console = _console()
        last: list[str] | None = None
        while not stop_fn():
            lines = _strip_lines(console, render_fn())
            if lines != last:
                _draw_live_strip(console, lines)
                last = lines
            time.sleep(interval)
        lines = _strip_lines(console, render_fn())
        if lines != last:
            _draw_live_strip(console, lines)
        return

    while not stop_fn():
        print(render_fn(), flush=True)
        time.sleep(interval)
    print(render_fn(), flush=True)
