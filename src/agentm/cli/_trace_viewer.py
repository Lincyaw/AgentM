# code-health: ignore-file[AM025] -- CLI renders typed-union trace records from query/store boundaries
"""Interactive trace viewer — pager-style TUI for session trajectories.

Navigation:
  Level 1  Turn list      ↑↓ select, Enter/→ expand, q quit
  Level 2  Message list   ↑↓ select, Enter toggle collapse,
                          e expand-all, c collapse-all, ←/Esc back
"""

from __future__ import annotations

import json
import os
import select
import sys
import termios
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from io import StringIO

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from agentm.core.abi.trajectory import Turn, TurnCheckpoint
from agentm.presenter.trajectory.model import (
    TraceRow,
    TraceSnapshot,
    TraceTurnSummary,
    build_trace_snapshot,
)


# -- Data model --------------------------------------------------------------


@dataclass(slots=True)
class _Message:
    role: str  # system | user | assistant | tool_call | tool_result | error
    tool_name: str | None = None
    is_error: bool = False
    content: str = ""
    thinking: str | None = None
    args_json: str | None = None


@dataclass(slots=True)
class _TurnView:
    summary: TraceTurnSummary
    trigger_label: str
    tools_str: str
    cause: str
    in_tok: int
    out_tok: int
    model: str
    messages: list[_Message] = field(default_factory=list)


@dataclass(slots=True)
class _ViewItem:
    """One collapsible row in the message-list view."""

    msg: _Message
    body_lines: list[str]
    expanded: bool


def _build_views(snapshot: TraceSnapshot) -> list[_TurnView]:
    views: list[_TurnView] = []
    rows_by_turn: dict[int, list[TraceRow]] = {}
    for row in snapshot.rows:
        if row.turn_index is None:
            continue
        rows_by_turn.setdefault(row.turn_index, []).append(row)

    for summary in snapshot.turns:
        trigger_label = _trigger_label(summary)
        tools_str = f"{summary.tool_calls} tools" if summary.tool_calls else "no tools"
        if summary.tool_errors:
            tools_str += f" ({summary.tool_errors} err)"
        cause = summary.cause or summary.status
        model = summary.model or "?"
        messages = _messages_from_rows(rows_by_turn.get(summary.turn_index, ()))
        views.append(
            _TurnView(
                summary=summary,
                trigger_label=trigger_label,
                tools_str=tools_str,
                cause=cause,
                in_tok=summary.input_tokens,
                out_tok=summary.output_tokens,
                model=model,
                messages=messages,
            )
        )
    return views


def _trigger_label(summary: TraceTurnSummary) -> str:
    text = " ".join(
        line.strip() for line in summary.trigger.splitlines() if line.strip()
    )
    if text:
        preview = text[:50]
        return f'"{preview}"'
    return summary.trigger_source or summary.status


def _messages_from_rows(rows: Sequence[TraceRow]) -> list[_Message]:
    msgs: list[_Message] = []
    for row in rows:
        if row.kind == "system":
            msgs.append(
                _Message(
                    role="system",
                    content=row.content,
                )
            )
        elif row.kind in {"user", "trigger"}:
            msgs.append(
                _Message(
                    role="user",
                    content=row.content,
                )
            )
        elif row.kind == "assistant":
            msgs.append(
                _Message(
                    role="assistant",
                    content=row.content,
                )
            )
        elif row.kind == "thinking":
            msgs.append(
                _Message(
                    role="assistant",
                    thinking=row.content,
                )
            )
        elif row.kind == "tool_call":
            msgs.append(
                _Message(
                    role="tool_call",
                    tool_name=row.tool_name,
                    args_json=row.content,
                )
            )
        elif row.kind == "tool_result":
            msgs.append(
                _Message(
                    role="tool_result",
                    tool_name=row.tool_name,
                    is_error=row.is_error,
                    content=row.content,
                )
            )
        elif row.kind == "error" or (row.kind == "control" and row.is_error):
            msgs.append(
                _Message(
                    role="error",
                    is_error=True,
                    content=row.content,
                )
            )

    return msgs


# -- Key reading -------------------------------------------------------------


_KEY_UP = "UP"
_KEY_DOWN = "DOWN"
_KEY_ENTER = "ENTER"
_KEY_SPACE = "SPACE"
_KEY_QUIT = "q"
_KEY_ESC = "ESC"
_KEY_LEFT = "LEFT"
_KEY_RIGHT = "RIGHT"
_KEY_PAGE_DOWN = "PGDN"
_KEY_PAGE_UP = "PGUP"
_KEY_HOME = "HOME"
_KEY_END = "END"
_KEY_TAB = "TAB"
_KEY_IGNORE = "IGNORE"
_KEY_SCROLL_UP = "SCROLL_UP"
_KEY_SCROLL_DOWN = "SCROLL_DOWN"
_ESCAPE_READ_TIMEOUT = 0.05
_WHEEL_SCROLL_LINES = 5
_ENABLE_MOUSE_REPORTING = "\033[?1000h\033[?1006h"
_DISABLE_MOUSE_REPORTING = "\033[?1006l\033[?1000l"


def _read_byte_if_ready(fd: int, timeout: float) -> bytes:
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return b""
    return os.read(fd, 1)


def _read_escape_sequence(fd: int) -> bytes:
    seq = _read_byte_if_ready(fd, _ESCAPE_READ_TIMEOUT)
    if not seq:
        return b""

    if seq == b"[":
        while len(seq) < 64:
            nxt = _read_byte_if_ready(fd, _ESCAPE_READ_TIMEOUT)
            if not nxt:
                break
            seq += nxt
            if 0x40 <= nxt[0] <= 0x7E:
                if seq == b"[M":
                    for _ in range(3):
                        more = _read_byte_if_ready(fd, _ESCAPE_READ_TIMEOUT)
                        if not more:
                            break
                        seq += more
                break
    elif seq == b"O":
        nxt = _read_byte_if_ready(fd, _ESCAPE_READ_TIMEOUT)
        if nxt:
            seq += nxt

    return seq


def _decode_mouse_sequence(seq: bytes) -> str:
    if seq.startswith(b"[<"):
        try:
            button_code = int(seq[2:-1].split(b";", 1)[0])
        except (TypeError, ValueError):
            return _KEY_IGNORE
        if button_code >= 64:
            button = button_code & 0b11
            if button == 0:
                return _KEY_SCROLL_UP
            if button == 1:
                return _KEY_SCROLL_DOWN
        return _KEY_IGNORE

    if len(seq) == 5 and seq.startswith(b"[M"):
        button_code = seq[2] - 32
        if button_code >= 64:
            button = button_code & 0b11
            if button == 0:
                return _KEY_SCROLL_UP
            if button == 1:
                return _KEY_SCROLL_DOWN
        return _KEY_IGNORE

    return _KEY_IGNORE


def _read_key(fd: int) -> str:
    ch = os.read(fd, 1)
    if ch == b"\x1b":
        seq = _read_escape_sequence(fd)
        if not seq:
            return _KEY_ESC
        if seq == b"[A":
            return _KEY_UP
        if seq == b"[B":
            return _KEY_DOWN
        if seq == b"[C":
            return _KEY_RIGHT
        if seq == b"[D":
            return _KEY_LEFT
        if seq == b"[5~":
            return _KEY_PAGE_UP
        if seq == b"[6~":
            return _KEY_PAGE_DOWN
        if seq == b"[H":
            return _KEY_HOME
        if seq == b"[F":
            return _KEY_END
        if seq in (b"OA", b"[1~", b"[7~"):
            return _KEY_HOME
        if seq in (b"OF", b"[4~", b"[8~"):
            return _KEY_END
        if seq.startswith((b"[<", b"[M")):
            return _decode_mouse_sequence(seq)
        return _KEY_IGNORE
    if ch in (b"\r", b"\n"):
        return _KEY_ENTER
    if ch == b" ":
        return _KEY_SPACE
    if ch == b"\t":
        return _KEY_TAB
    if ch in (b"q", b"Q"):
        return _KEY_QUIT
    if ch in (b"j",):
        return _KEY_DOWN
    if ch in (b"k",):
        return _KEY_UP
    if ch in (b"h",):
        return _KEY_LEFT
    if ch in (b"l",):
        return _KEY_RIGHT
    if ch == b"\x03":  # Ctrl+C
        return _KEY_QUIT
    return ch.decode("utf-8", errors="replace")


def _try_read_key(fd: int, timeout: float) -> str | None:
    """Non-blocking key read.  Returns ``None`` if *timeout* elapses."""
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    return _read_key(fd)


# -- Turn summary rendering -------------------------------------------------


def _render_turn_summary(view: _TurnView, width: int) -> str:
    """Render a turn summary line that fits within *width* columns.

    Layout priority (right-to-left trim):
      fixed:  "T 0 | "  (6 chars)
      tail:   " | <tools> | <cause> | in:N out:N | <model>"
      flex:   trigger_label gets whatever remains
    """
    prefix = f"T{view.summary.turn_index:>2} | "
    tail = (
        f" | {view.tools_str} | {view.cause}"
        f" | in:{view.in_tok:>7,} out:{view.out_tok:>5,}"
        f" | {view.model}"
    )

    avail = width - len(prefix) - len(tail)
    if avail < 12:
        tail = f" | {view.tools_str} | in:{view.in_tok:,} out:{view.out_tok:,}"
        avail = width - len(prefix) - len(tail)
    if avail < 8:
        tail = ""
        avail = width - len(prefix)

    label = view.trigger_label
    if len(label) > avail:
        label = label[: max(0, avail - 3)] + "..."
    else:
        label = f"{label:<{avail}}"

    return f"{prefix}{label}{tail}"


# -- Collapsible item rendering ----------------------------------------------

_MAX_BODY_LINES = 60

_ANSI = {
    "system": "\033[1;35m",
    "user": "\033[1;32m",
    "assistant": "\033[1;34m",
    "tool_call": "\033[1;33m",
    "tool_result": "\033[36m",
    "error": "\033[1;31m",
}
_RST = "\033[0m"
_DIM = "\033[2m"
_REV = "\033[7m"


def _collapsed_info(msg: _Message) -> str:
    """Short info shown on the header when an item is collapsed."""
    if msg.role == "tool_result":
        if not msg.content:
            return "(empty)"
        n = msg.content.count("\n") + 1
        chars = len(msg.content)
        if chars >= 1024:
            return f"({n} lines, {chars / 1024:.1f}K)"
        return f"({n} lines)"
    text = msg.content or msg.args_json or ""
    text = text.replace("\n", " ").strip()
    if not text:
        return ""
    if len(text) > 50:
        text = text[:47] + "..."
    return text


def _render_item_header(item: _ViewItem, width: int, *, selected: bool = False) -> str:
    msg = item.msg
    marker = "▾" if item.expanded else "▸"
    color = _ANSI.get(msg.role, "")

    label = msg.role.upper()
    if msg.tool_name:
        label += f": {msg.tool_name}"
    if msg.is_error and msg.role != "error":
        label += " [ERR]"

    if item.expanded:
        inner = f" {marker} {label} "
    else:
        info = _collapsed_info(msg)
        if info:
            inner = f" {marker} {label} {_DIM}{info}{_RST}{color} "
        else:
            inner = f" {marker} {label} "

    # The visible width ignores ANSI sequences for the rule padding.
    # Approximate: strip DIM/RST/color that appear inside `inner`.
    vis_len = len(label) + 4  # marker + spaces
    if not item.expanded:
        info = _collapsed_info(msg)
        vis_len += len(info) + 1 if info else 0
    rule_len = max(0, width - vis_len - 1)
    line = f"{inner}{'─' * rule_len}"

    if selected:
        return f"{_REV}{color}{line}{_RST}"
    return f"{color}{line}{_RST}"


def _looks_like_json(s: str) -> bool:
    stripped = s.strip()
    if len(stripped) < 2:
        return False
    return (stripped[0] == "{" and stripped[-1] == "}") or (
        stripped[0] == "[" and stripped[-1] == "]"
    )


def _render_body_lines(msg: _Message, width: int) -> list[str]:
    """Pre-render message content into indented ANSI lines."""
    con = Console(
        file=StringIO(),
        width=max(20, width - 4),
        highlight=False,
        force_terminal=True,
        color_system="truecolor",
    )

    if msg.thinking:
        lines = msg.thinking.split("\n")
        if len(lines) > 8:
            display = lines[:3] + [f"... ({len(lines) - 6} lines) ..."] + lines[-3:]
        else:
            display = lines
        for line in display:
            con.print(Text(f"\U0001f4ad {line}", style="dim italic"))
        con.print()

    if msg.args_json:
        try:
            syn = Syntax(
                msg.args_json,
                "json",
                theme="monokai",
                word_wrap=True,
                padding=(0, 0),
            )
            con.print(syn)
        except Exception:
            con.print(Text(msg.args_json, style="dim"))

    content = msg.content
    if content:
        raw_lines = content.split("\n")
        if len(raw_lines) > _MAX_BODY_LINES:
            half = _MAX_BODY_LINES // 2
            omitted = len(raw_lines) - _MAX_BODY_LINES
            content = "\n".join(
                raw_lines[:half]
                + [f"\n... ({omitted} lines omitted) ...\n"]
                + raw_lines[-half:]
            )

        if msg.role == "tool_result" and not msg.is_error:
            if _looks_like_json(content):
                try:
                    formatted = json.dumps(
                        json.loads(content),
                        indent=2,
                        ensure_ascii=False,
                    )
                    flines = formatted.split("\n")
                    if len(flines) > _MAX_BODY_LINES:
                        half = _MAX_BODY_LINES // 2
                        formatted = "\n".join(
                            flines[:half]
                            + [f"  ... ({len(flines) - _MAX_BODY_LINES} lines) ..."]
                            + flines[-half:]
                        )
                    syn = Syntax(
                        formatted,
                        "json",
                        theme="monokai",
                        word_wrap=True,
                        padding=(0, 0),
                    )
                    con.print(syn)
                except (json.JSONDecodeError, ValueError):
                    for line in content.split("\n"):
                        con.print(Text(line, style="dim"))
            else:
                for line in content.split("\n"):
                    con.print(Text(line, style="dim"))
        elif msg.role == "error":
            con.print(Text(content, style="bold red"))
        elif msg.role == "system":
            con.print(Text(content, style="magenta"))
        elif msg.role == "user":
            con.print(Text(content, style="green"))
        else:
            con.print(Text(content))

    raw = con.file.getvalue()  # type: ignore[attr-defined]
    result: list[str] = []
    for line in raw.split("\n"):
        result.append(f"  {line}" if line.rstrip() else "")
    while result and not result[-1].strip():
        result.pop()
    return result


def _build_view_items(messages: list[_Message], width: int) -> list[_ViewItem]:
    items: list[_ViewItem] = []
    for msg in messages:
        default_expanded = msg.role != "tool_result"
        items.append(
            _ViewItem(
                msg=msg,
                body_lines=_render_body_lines(msg, width),
                expanded=default_expanded,
            )
        )
    return items


# -- Interactive viewer ------------------------------------------------------


class TraceViewer:
    """Interactive pager for session trace turns."""

    def __init__(
        self,
        turns: list[Turn],
        session_id: str,
        *,
        checkpoints: list[TurnCheckpoint] | None = None,
        reload: Callable[[], tuple[list[Turn], list[TurnCheckpoint]]] | None = None,
    ) -> None:
        self._views = _build_views(
            build_trace_snapshot(session_id, turns, checkpoints or ())
        )
        self._session_id = session_id
        self._reload = reload
        # Turn-list state
        self._cursor = 0
        self._expanded = False
        # Message-list state (active when _expanded is True)
        self._msg_cursor = 0
        self._items: list[_ViewItem] | None = None
        self._scroll = 0
        self._manual_scroll = False

    def run(self) -> None:
        if not self._views:
            Console(stderr=True).print("[dim]No turns to display.[/dim]")
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            new_settings = termios.tcgetattr(fd)
            new_settings[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
            new_settings[1] |= termios.OPOST | termios.ONLCR
            new_settings[6][termios.VMIN] = 1
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
            sys.stdout.write("\033[?1049h")  # alt screen
            sys.stdout.write(_ENABLE_MOUSE_REPORTING)
            sys.stdout.write("\033[?25l")  # hide cursor
            sys.stdout.flush()
            self._loop(fd)
        finally:
            sys.stdout.write(_DISABLE_MOUSE_REPORTING)
            sys.stdout.write("\033[?25h")
            sys.stdout.write("\033[?1049l")
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _term_size(self) -> tuple[int, int]:
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 120, 40
        return cols, rows

    def _enter_expanded(self) -> None:
        self._expanded = True
        self._items = None
        self._msg_cursor = 0
        self._scroll = 0
        self._manual_scroll = False

    def _leave_expanded(self) -> None:
        self._expanded = False
        self._items = None
        self._scroll = 0
        self._manual_scroll = False

    def _maybe_reload(self) -> None:
        if self._reload is None:
            return
        new_turns, new_checkpoints = self._reload()
        new_views = _build_views(
            build_trace_snapshot(self._session_id, new_turns, new_checkpoints)
        )
        if len(new_views) != len(self._views):
            was_at_end = self._cursor >= len(self._views) - 1
            self._views = new_views
            if was_at_end:
                self._cursor = max(0, len(self._views) - 1)
                if self._expanded:
                    self._items = None
                    self._msg_cursor = 0
                    self._scroll = 0
                    self._manual_scroll = False
        else:
            self._views = new_views

    def _loop(self, fd: int) -> None:
        while True:
            self._draw()

            if self._reload is not None:
                key = _try_read_key(fd, 1.5)
                if key is None:
                    self._maybe_reload()
                    continue
            else:
                key = _read_key(fd)

            if key == _KEY_QUIT:
                break
            if key == _KEY_ESC and not self._expanded:
                break
            if key == _KEY_IGNORE:
                continue

            if self._expanded:
                self._handle_expanded_key(key)
            else:
                self._handle_list_key(key)

    def _handle_list_key(self, key: str) -> None:
        _, rows = self._term_size()
        page = max(1, rows - 4)
        if key == _KEY_UP:
            self._cursor = max(0, self._cursor - 1)
        elif key == _KEY_DOWN:
            self._cursor = min(len(self._views) - 1, self._cursor + 1)
        elif key == _KEY_SCROLL_UP:
            self._cursor = max(0, self._cursor - page)
        elif key == _KEY_SCROLL_DOWN:
            self._cursor = min(len(self._views) - 1, self._cursor + page)
        elif key in (_KEY_ENTER, _KEY_SPACE, _KEY_RIGHT):
            self._enter_expanded()
        elif key == _KEY_PAGE_DOWN:
            self._cursor = min(len(self._views) - 1, self._cursor + page)
        elif key == _KEY_PAGE_UP:
            self._cursor = max(0, self._cursor - page)
        elif key == _KEY_HOME:
            self._cursor = 0
        elif key == _KEY_END:
            self._cursor = len(self._views) - 1

    def _expanded_total_lines(self, items: list[_ViewItem]) -> int:
        return sum(2 + (len(item.body_lines) if item.expanded else 0) for item in items)

    def _scroll_expanded(self, delta: int) -> None:
        cols, rows = self._term_size()
        if self._items is None:
            view = self._views[self._cursor]
            self._items = _build_view_items(view.messages, cols)
        usable = max(1, rows - 3)
        total = self._expanded_total_lines(self._items)
        max_scroll = max(0, total - usable)
        self._scroll = max(0, min(max_scroll, self._scroll + delta))
        self._manual_scroll = True

    def _handle_expanded_key(self, key: str) -> None:
        items = self._items
        if items is None:
            return

        if key in (_KEY_ESC, _KEY_LEFT):
            self._leave_expanded()
        elif key == _KEY_UP:
            self._manual_scroll = False
            self._msg_cursor = max(0, self._msg_cursor - 1)
        elif key == _KEY_DOWN:
            self._manual_scroll = False
            self._msg_cursor = min(len(items) - 1, self._msg_cursor + 1)
        elif key in (_KEY_ENTER, _KEY_SPACE):
            items[self._msg_cursor].expanded = not items[self._msg_cursor].expanded
        elif key == _KEY_PAGE_DOWN:
            _, rows = self._term_size()
            self._scroll_expanded(max(1, rows - 5))
        elif key == _KEY_PAGE_UP:
            _, rows = self._term_size()
            self._scroll_expanded(-max(1, rows - 5))
        elif key == _KEY_SCROLL_DOWN:
            self._scroll_expanded(_WHEEL_SCROLL_LINES)
        elif key == _KEY_SCROLL_UP:
            self._scroll_expanded(-_WHEEL_SCROLL_LINES)
        elif key == _KEY_HOME:
            self._manual_scroll = False
            self._msg_cursor = 0
            self._scroll = 0
        elif key == _KEY_END:
            self._manual_scroll = False
            self._msg_cursor = len(items) - 1
        elif key == _KEY_TAB:
            if self._cursor < len(self._views) - 1:
                self._cursor += 1
                self._enter_expanded()
        elif key == "e":
            for item in items:
                item.expanded = True
        elif key == "c":
            for item in items:
                item.expanded = False

    # -- Drawing -------------------------------------------------------------

    def _draw(self) -> None:
        cols, rows = self._term_size()
        sys.stdout.write("\033[2J\033[H")

        if self._expanded:
            self._draw_expanded(cols, rows)
        else:
            self._draw_list(cols, rows)

        sys.stdout.flush()

    def _draw_list(self, cols: int, rows: int) -> None:
        follow_tag = " [following]" if self._reload is not None else ""
        header = (
            f" agentm trace | session: {self._session_id}"
            f" | {len(self._views)} turn(s){follow_tag}"
        )
        footer = " ↑↓/jk: move  Enter/→: expand  q: quit  Home/End: jump"

        sys.stdout.write(f"\033[7m{header:<{cols}}\033[0m\n")

        usable = rows - 3
        start = max(0, self._cursor - usable // 2)
        end = min(len(self._views), start + usable)
        if end - start < usable:
            start = max(0, end - usable)

        for i in range(start, end):
            view = self._views[i]
            selected = i == self._cursor
            line = _render_turn_summary(view, cols - 3)

            if selected:
                sys.stdout.write(f"\033[1;7m ▸ {line}\033[0m\n")
            else:
                sys.stdout.write(f"   {line}\n")

        remaining = usable - (end - start)
        for _ in range(remaining):
            sys.stdout.write(" " * cols + "\n")

        sys.stdout.write(f"\033[2m{footer:<{cols}}\033[0m")

    def _draw_expanded(self, cols: int, rows: int) -> None:
        view = self._views[self._cursor]

        if self._items is None:
            self._items = _build_view_items(view.messages, cols)
            self._msg_cursor = 0
            self._scroll = 0

        items = self._items

        # Build flat line list with item ownership.
        all_lines: list[str] = []
        item_header_line: list[int] = []  # line index of each item's header

        for idx, item in enumerate(items):
            item_header_line.append(len(all_lines))
            hdr = _render_item_header(item, cols, selected=(idx == self._msg_cursor))
            all_lines.append(hdr)

            if item.expanded:
                all_lines.extend(item.body_lines)

            all_lines.append("")  # separator

        total = len(all_lines)
        usable = rows - 3

        # Auto-scroll: keep cursor item's header visible.
        cursor_line = item_header_line[self._msg_cursor]
        if not self._manual_scroll:
            if cursor_line < self._scroll:
                self._scroll = cursor_line
            elif cursor_line >= self._scroll + usable:
                self._scroll = cursor_line - usable + 1
        self._scroll = max(0, min(self._scroll, max(0, total - usable)))

        visible = all_lines[self._scroll : self._scroll + usable]

        # Header bar
        pct = (
            int(self._scroll / max(1, total - usable) * 100) if total > usable else 100
        )
        turn_label = f"Turn {self._cursor + 1}/{len(self._views)}"
        n_collapsed = sum(1 for it in items if not it.expanded)
        header = (
            f" {turn_label} | {len(items)} msg(s)"
            f"{f', {n_collapsed} collapsed' if n_collapsed else ''}"
            f" | {pct}%"
        )
        sys.stdout.write(f"\033[7m{header:<{cols}}\033[0m\n")

        for line in visible:
            sys.stdout.write(f"{line}\n")

        remaining = usable - len(visible)
        for _ in range(remaining):
            sys.stdout.write("\n")

        footer = (
            " ↑↓: move  Enter: toggle  "
            "e: expand all  c: collapse all  "
            "PgUp/PgDn/wheel: scroll  ←/Esc: back  Tab: next turn"
        )
        sys.stdout.write(f"\033[2m{footer:<{cols}}\033[0m")


def run_interactive_viewer(
    turns: list[Turn],
    session_id: str,
    *,
    checkpoints: list[TurnCheckpoint] | None = None,
    reload: Callable[[], tuple[list[Turn], list[TurnCheckpoint]]] | None = None,
) -> None:
    """Entry point for the interactive trace viewer."""
    viewer = TraceViewer(turns, session_id, checkpoints=checkpoints, reload=reload)
    viewer.run()
