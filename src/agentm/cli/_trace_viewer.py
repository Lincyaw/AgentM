"""Interactive trace viewer — pager-style TUI for session trajectories."""

from __future__ import annotations

import json
import os
import sys
import termios
from dataclasses import dataclass, field
from io import StringIO

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from agentm.core.abi.messages import (
    ImageContent,
    OpaqueThinkingBlock,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
)
from agentm.core.abi.termination import ProviderRequestFailed
from agentm.core.abi.trajectory import Turn
from agentm.core.abi.trigger import UserInput


# -- Data model --------------------------------------------------------------


@dataclass(slots=True)
class _Message:
    role: str  # user | assistant | tool_call | tool_result | error
    tool_name: str | None = None
    is_error: bool = False
    content: str = ""
    thinking: str | None = None
    args_json: str | None = None


@dataclass(slots=True)
class _TurnView:
    turn: Turn
    summary_line: str
    messages: list[_Message] = field(default_factory=list)


def _build_views(turns: list[Turn]) -> list[_TurnView]:
    views: list[_TurnView] = []
    for turn in turns:
        tool_names: list[str] = []
        tool_errors = 0
        for rnd in turn.rounds:
            for rec in rnd.tool_results:
                tool_names.append(rec.call.name)
                if rec.result.is_error:
                    tool_errors += 1

        trigger_label = turn.trigger.source
        if isinstance(turn.trigger, UserInput):
            parts = []
            for block in turn.trigger.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
            preview = " ".join(parts).replace("\n", " ")[:50]
            if preview:
                trigger_label = f'"{preview}"'

        tools_str = f"{len(tool_names)} tools" if tool_names else "no tools"
        if tool_errors:
            tools_str += f" ({tool_errors} err)"
        cause = type(turn.outcome.cause).__name__
        in_tok = turn.meta.total_input_tokens
        out_tok = turn.meta.total_output_tokens
        model = turn.meta.model_id or "?"

        summary = f"T{turn.index:>2} | {trigger_label:<52} | {tools_str:<16} | {cause:<16} | in:{in_tok:>8,} out:{out_tok:>6,} | {model}"
        msgs = _extract_messages(turn)
        views.append(_TurnView(turn=turn, summary_line=summary, messages=msgs))
    return views


def _extract_messages(turn: Turn) -> list[_Message]:
    msgs: list[_Message] = []

    if isinstance(turn.trigger, UserInput):
        parts = []
        for block in turn.trigger.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ImageContent):
                parts.append(f"[image {block.mime_type}, {len(block.data)} bytes]")
            else:
                parts.append(f"[{type(block).__name__}]")
        msgs.append(_Message(role="user", content="\n".join(parts)))

    for rnd in turn.rounds:
        text_parts: list[str] = []
        thinking_parts: list[str] = []

        for response_block in rnd.response.content:
            if isinstance(response_block, TextContent):
                text_parts.append(response_block.text)
            elif isinstance(response_block, ThinkingBlock):
                thinking_parts.append(response_block.text)
            elif isinstance(response_block, OpaqueThinkingBlock):
                thinking_parts.append(
                    f"[opaque reasoning: {response_block.provider}]"
                )
            elif isinstance(response_block, ToolCallBlock):
                args_str = json.dumps(
                    dict(response_block.arguments),
                    ensure_ascii=False,
                    indent=2,
                )
                if text_parts or thinking_parts:
                    msgs.append(_Message(
                        role="assistant", content="\n".join(text_parts),
                        thinking="\n".join(thinking_parts) if thinking_parts else None,
                    ))
                    text_parts = []
                    thinking_parts = []
                msgs.append(_Message(
                    role="tool_call",
                    tool_name=response_block.name,
                    args_json=args_str,
                ))

        if text_parts or thinking_parts:
            msgs.append(_Message(
                role="assistant", content="\n".join(text_parts),
                thinking="\n".join(thinking_parts) if thinking_parts else None,
            ))

        for rec in rnd.tool_results:
            txt = "".join(
                result_block.text
                for result_block in rec.result.content
                if isinstance(result_block, TextContent)
            )
            msgs.append(_Message(
                role="tool_result", tool_name=rec.call.name,
                is_error=rec.result.is_error, content=txt,
            ))

    if isinstance(turn.outcome.cause, ProviderRequestFailed):
        msgs.append(_Message(
            role="error", is_error=True,
            content=f"{turn.outcome.cause.error_type}: {turn.outcome.cause.detail}",
        ))

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


def _read_key(fd: int) -> str:
    ch = os.read(fd, 1)
    if ch == b"\x1b":
        seq = os.read(fd, 2)
        if seq == b"[A":
            return _KEY_UP
        if seq == b"[B":
            return _KEY_DOWN
        if seq == b"[C":
            return _KEY_RIGHT
        if seq == b"[D":
            return _KEY_LEFT
        if seq == b"[5":
            os.read(fd, 1)
            return _KEY_PAGE_UP
        if seq == b"[6":
            os.read(fd, 1)
            return _KEY_PAGE_DOWN
        if seq == b"[H":
            return _KEY_HOME
        if seq == b"[F":
            return _KEY_END
        return _KEY_ESC
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


# -- Rendering helpers -------------------------------------------------------


_ROLE_COLORS = {
    "user": "green",
    "assistant": "blue",
    "tool_call": "yellow",
    "tool_result": "cyan",
    "error": "red",
}

_MAX_LINES = 50


def _render_expanded_turn(view: _TurnView, width: int) -> list[str]:
    """Render a turn's messages into ANSI-colored lines."""
    con = Console(
        file=StringIO(), width=width, highlight=False,
        force_terminal=True, color_system="truecolor",
    )

    con.print(Text(f"Turn {view.turn.index}", style="bold underline"))
    con.print()

    for msg in view.messages:
        _render_message(con, msg, width)
        con.print()

    output = con.file.getvalue()  # type: ignore[attr-defined]
    return output.split("\n")


def _render_message(con: Console, msg: _Message, width: int) -> None:
    msg = _Message(
        role=msg.role, tool_name=msg.tool_name, is_error=msg.is_error,
        content=msg.content.expandtabs(4) if msg.content else "",
        thinking=msg.thinking.expandtabs(4) if msg.thinking else None,
        args_json=msg.args_json,
    )
    color = _ROLE_COLORS.get(msg.role, "white")
    label = msg.role.upper()
    if msg.tool_name:
        label = f"{label}: {msg.tool_name}"
    if msg.is_error:
        label += " [ERROR]"
        color = "red"

    rule_char = "─"
    label_str = f" {label} "
    remaining = max(0, width - len(label_str) - 2)
    left = remaining // 2
    right = remaining - left
    header_line = f"{rule_char * left}{label_str}{rule_char * right}"
    con.print(Text(header_line, style=f"bold {color}"))

    if msg.thinking:
        lines = msg.thinking.split("\n")
        if len(lines) > 8:
            display = lines[:3] + [f"    ... ({len(lines) - 6} lines) ..."] + lines[-3:]
        else:
            display = lines
        for line in display:
            con.print(Text(f"  💭 {line}", style="dim italic"))

    if msg.args_json:
        try:
            syn = Syntax(msg.args_json, "json", theme="monokai", word_wrap=True, padding=(0, 2))
            con.print(syn)
        except Exception:
            con.print(Text(msg.args_json, style="dim"))

    if msg.content:
        content = msg.content
        lines = content.split("\n")
        if len(lines) > _MAX_LINES:
            half = _MAX_LINES // 2
            content = "\n".join(lines[:half] + [f"\n    ... ({len(lines) - _MAX_LINES} lines omitted) ...\n"] + lines[-half:])

        if msg.role == "tool_result" and not msg.is_error:
            if _looks_like_json(content):
                try:
                    formatted = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
                    flines = formatted.split("\n")
                    if len(flines) > _MAX_LINES:
                        half = _MAX_LINES // 2
                        formatted = "\n".join(flines[:half] + [f"  ... ({len(flines) - _MAX_LINES} lines) ..."] + flines[-half:])
                    syn = Syntax(formatted, "json", theme="monokai", word_wrap=True, padding=(0, 2))
                    con.print(syn)
                except (json.JSONDecodeError, ValueError):
                    con.print(Text(content, style="dim"))
            else:
                for line in content.split("\n"):
                    con.print(Text(f"  {line}", style="dim"))
        elif msg.role == "error":
            con.print(Text(content, style="bold red"))
        elif msg.role == "user":
            con.print(Text(content, style="green"))
        else:
            con.print(Text(content))


def _looks_like_json(s: str) -> bool:
    stripped = s.strip()
    if len(stripped) < 2:
        return False
    return (stripped[0] == "{" and stripped[-1] == "}") or \
           (stripped[0] == "[" and stripped[-1] == "]")


# -- Interactive viewer ------------------------------------------------------


class TraceViewer:
    """Interactive pager for session trace turns."""

    def __init__(self, turns: list[Turn], session_id: str) -> None:
        self._views = _build_views(turns)
        self._session_id = session_id
        self._cursor = 0
        self._expanded = False
        self._scroll = 0
        self._cached_lines: list[str] | None = None

    def run(self) -> None:
        if not self._views:
            Console(stderr=True).print("[dim]No turns to display.[/dim]")
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            new_settings = termios.tcgetattr(fd)
            # Disable canonical mode, echo, and signals for raw key reading
            new_settings[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
            # Keep output processing (OPOST) so tabs expand and \n→\n works
            new_settings[1] |= termios.OPOST | termios.ONLCR
            # Read one byte at a time, no timeout
            new_settings[6][termios.VMIN] = 1
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
            sys.stdout.write("\033[?1049h")  # alt screen
            sys.stdout.write("\033[?25l")  # hide cursor
            sys.stdout.flush()
            self._loop(fd)
        finally:
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

    def _loop(self, fd: int) -> None:
        while True:
            self._draw()
            key = _read_key(fd)

            if key == _KEY_QUIT:
                break
            elif key == _KEY_ESC:
                if self._expanded:
                    self._expanded = False
                    self._scroll = 0
                    self._cached_lines = None
                else:
                    break
            elif key == _KEY_UP:
                if self._expanded:
                    self._scroll = max(0, self._scroll - 1)
                else:
                    self._cursor = max(0, self._cursor - 1)
            elif key == _KEY_DOWN:
                if self._expanded:
                    self._scroll += 1
                else:
                    self._cursor = min(len(self._views) - 1, self._cursor + 1)
            elif key in (_KEY_ENTER, _KEY_SPACE, _KEY_RIGHT):
                if not self._expanded:
                    self._expanded = True
                    self._scroll = 0
                    self._cached_lines = None
            elif key in (_KEY_LEFT,):
                if self._expanded:
                    self._expanded = False
                    self._scroll = 0
                    self._cached_lines = None
            elif key == _KEY_PAGE_DOWN:
                if self._expanded:
                    self._scroll += 20
                else:
                    self._cursor = min(len(self._views) - 1, self._cursor + 10)
            elif key == _KEY_PAGE_UP:
                if self._expanded:
                    self._scroll = max(0, self._scroll - 20)
                else:
                    self._cursor = max(0, self._cursor - 10)
            elif key == _KEY_HOME:
                if self._expanded:
                    self._scroll = 0
                else:
                    self._cursor = 0
            elif key == _KEY_END:
                if self._expanded:
                    self._scroll = 99999
                else:
                    self._cursor = len(self._views) - 1
            elif key == _KEY_TAB:
                if self._expanded:
                    # next turn while staying expanded
                    if self._cursor < len(self._views) - 1:
                        self._cursor += 1
                        self._scroll = 0
                        self._cached_lines = None

    def _draw(self) -> None:
        cols, rows = self._term_size()
        sys.stdout.write("\033[2J\033[H")  # clear + cursor home

        if self._expanded:
            self._draw_expanded(cols, rows)
        else:
            self._draw_list(cols, rows)

        sys.stdout.flush()

    def _draw_list(self, cols: int, rows: int) -> None:
        header = f" agentm trace | session: {self._session_id} | {len(self._views)} turn(s)"
        footer = " ↑↓/jk: move  Enter/→: expand  q: quit  Home/End: jump"

        sys.stdout.write(f"\033[7m{header:<{cols}}\033[0m\n")

        usable = rows - 3
        start = max(0, self._cursor - usable // 2)
        end = min(len(self._views), start + usable)
        if end - start < usable:
            start = max(0, end - usable)

        for i in range(start, end):
            view = self._views[i]
            selected = (i == self._cursor)
            line = view.summary_line[:cols - 3]

            if selected:
                sys.stdout.write(f"\033[1;7m ▸ {line}\033[0m")
            else:
                sys.stdout.write(f"   {line}")

            # pad to full width and newline
            written = len(line) + 3
            sys.stdout.write(" " * max(0, cols - written))
            sys.stdout.write("\n")

        # fill remaining rows
        remaining = usable - (end - start)
        for _ in range(remaining):
            sys.stdout.write(" " * cols + "\n")

        sys.stdout.write(f"\033[2m{footer:<{cols}}\033[0m")

    def _draw_expanded(self, cols: int, rows: int) -> None:
        view = self._views[self._cursor]

        if self._cached_lines is None:
            self._cached_lines = _render_expanded_turn(view, cols - 2)

        all_lines = self._cached_lines
        total = len(all_lines)
        usable = rows - 3

        self._scroll = min(self._scroll, max(0, total - usable))

        visible = all_lines[self._scroll:self._scroll + usable]

        pct = int(self._scroll / max(1, total - usable) * 100) if total > usable else 100
        turn_label = f"Turn {view.turn.index + 1}/{len(self._views)}"
        header = f" {turn_label} | {len(view.messages)} msg(s) | {pct}%"
        footer = " ↑↓: scroll  ←/Esc: back  PgUp/PgDn: page  Tab: next turn"

        sys.stdout.write(f"\033[7m{header:<{cols}}\033[0m\n")

        for line in visible:
            sys.stdout.write(f" {line}\n")

        remaining = usable - len(visible)
        for _ in range(remaining):
            sys.stdout.write("\n")

        sys.stdout.write(f"\033[2m{footer:<{cols}}\033[0m")


def run_interactive_viewer(turns: list[Turn], session_id: str) -> None:
    """Entry point for the interactive trace viewer."""
    viewer = TraceViewer(turns, session_id)
    viewer.run()
