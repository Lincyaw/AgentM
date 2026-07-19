"""Live TUI dashboard for monitoring and steering a running agentm session.

Usage::

    agentm dashboard <session-id>

Requires ``textual>=3.0`` (install with ``pip install textual``).
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import sys
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog


def _agentm_home() -> Path:
    import os
    return Path(os.environ.get("AGENTM_HOME", Path.home() / ".agentm"))


def _socket_path(session_id: str) -> Path:
    return _agentm_home() / "live" / f"{session_id}.sock"


def _send_message(session_id: str, message: str, mode: str = "wait") -> str:
    sock_path = _socket_path(session_id)
    if not sock_path.exists():
        return f"error: socket not found ({sock_path})"
    prefix = {"now": "!now\n", "interrupt": "!interrupt\n", "wait": ""}.get(mode, "")
    payload = f"{prefix}{message}"
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(str(sock_path))
        s.sendall(payload.encode("utf-8"))
        s.shutdown(socket.SHUT_WR)
        ack = s.recv(64).decode("utf-8", errors="replace").strip()
        s.close()
        return ack
    except Exception as exc:
        return f"error: {exc}"


def _fetch_messages(session_id: str) -> list[dict[str, Any]]:
    try:
        from agentm.core.observability import clickhouse as ch
        url = ch.get_url()
        if url is None:
            return []
        return list(ch.messages(url, session_id, include_system_prompt=False))
    except Exception:
        return []


def _fetch_usage(session_id: str) -> dict[str, Any] | None:
    try:
        from agentm.core.observability import clickhouse as ch
        url = ch.get_url()
        if url is None:
            return None
        return ch.usage(url, session_id)
    except Exception:
        return None


def _trunc(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _extract_child_session_id(entry: dict[str, Any]) -> str | None:
    """If this entry is a tool_result from dispatch_agent containing
    child_session_id, return it."""
    payload = entry.get("payload", {})
    if not isinstance(payload, dict):
        return None
    role = payload.get("role", "")
    if role != "tool_result":
        return None
    content = payload.get("content", [])
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if "child_session_id" in text:
            try:
                data = json.loads(text)
                return data.get("child_session_id")
            except (json.JSONDecodeError, TypeError):
                m = re.search(r'"child_session_id"\s*:\s*"([a-f0-9]+)"', text)
                if m:
                    return m.group(1)
    return None


def _render_entry(entry: dict[str, Any], prefix: str = "") -> list[str]:
    payload = entry.get("payload", {})
    if not isinstance(payload, dict):
        return []
    role = payload.get("role", "?")
    content = payload.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []

    out: list[str] = []
    p = prefix

    for item in content:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "")

        if itype == "thinking":
            thinking_text = item.get("text", "")
            if not thinking_text:
                continue
            out.append(f"{p}  [italic dim]💭 thinking[/]")
            lines = thinking_text.strip().split("\n")
            for line in lines[:15]:
                out.append(f"{p}  [dim]{_trunc(line, 140)}[/]")
            if len(lines) > 15:
                out.append(f"{p}  [dim]... ({len(lines)} lines total)[/]")

        elif itype == "text":
            text = item.get("text", "").strip()
            if not text:
                continue
            if role == "user":
                out.append(f"{p}[bold green]👤 user[/]")
                for line in text.split("\n")[:20]:
                    out.append(f"{p}  {_trunc(line, 140)}")
            elif role == "assistant":
                out.append(f"{p}[bold white]🤖 assistant[/]")
                for line in text.split("\n")[:10]:
                    out.append(f"{p}  {_trunc(line, 140)}")

        elif itype == "tool_call":
            name = item.get("name", "?")
            args = item.get("arguments", item.get("input", {}))
            out.append(f"{p}[bold cyan]🔧 {name}[/]")
            if isinstance(args, dict):
                for k, v in args.items():
                    sv = str(v)
                    if len(sv) > 120:
                        sv = sv[:117] + "..."
                    out.append(f"{p}  [cyan]{k}:[/] {sv}")
            else:
                out.append(f"{p}  {_trunc(str(args), 200)}")

        elif itype == "tool_result" or role == "tool_result":
            is_err = item.get("is_error", False)
            inner = item.get("content", "")
            if isinstance(inner, list) and inner:
                first = inner[0]
                text = first.get("text", str(first)) if isinstance(first, dict) else str(first)
            elif isinstance(inner, str):
                text = inner
            else:
                text = str(inner)
            lines = text.split("\n")
            tag = f"{p}  [red]❌ error[/]" if is_err else f"{p}  [dim]→ result[/]"
            if len(lines) > 6:
                out.append(f"{tag} ({len(lines)} lines):")
                for line in lines[:5]:
                    out.append(f"{p}    [dim]{_trunc(line, 120)}[/]")
                out.append(f"{p}    [dim]...[/]")
            else:
                out.append(f"{tag}:")
                for line in lines:
                    out.append(f"{p}    [dim]{_trunc(line, 120)}[/]")

    if role == "tool_result" and not out:
        for item in content:
            if not isinstance(item, dict):
                continue
            inner = item.get("content", "")
            is_err = item.get("is_error", False)
            if isinstance(inner, list) and inner:
                first = inner[0]
                text = first.get("text", str(first)) if isinstance(first, dict) else str(first)
            elif isinstance(inner, str):
                text = inner
            else:
                text = str(inner)
            lines = text.split("\n")
            tag = f"{p}  [red]❌[/]" if is_err else f"{p}  [dim]→[/]"
            if len(lines) > 6:
                for line in lines[:5]:
                    out.append(f"{tag} [dim]{_trunc(line, 120)}[/]")
                out.append(f"{p}  [dim]  ... ({len(lines)} lines)[/]")
            else:
                for line in lines[:3]:
                    out.append(f"{tag} [dim]{_trunc(line, 120)}[/]")

    return out


class Dashboard(App):
    TITLE = "agentm session dashboard"
    CSS = """
    #trajectory {
        height: 1fr;
        border: solid $primary;
    }
    #input-box {
        dock: bottom;
        height: 3;
        margin: 0 1;
    }
    """
    BINDINGS = [
        Binding("ctrl+n", "send_now", "Send (now)", show=True),
        Binding("ctrl+t", "send_interrupt", "Send (interrupt)", show=True),
    ]

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self._seen_ids: set[str] = set()
        self._child_sessions: dict[str, set[str]] = {}
        self._poll_count = 0
        self._entry_count = 0
        self._ctrl_c_pending = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield RichLog(id="trajectory", highlight=True, markup=True, wrap=True)
        yield Input(
            placeholder="Enter=send (wait) | Ctrl+N=now | Ctrl+T=interrupt | 2×Ctrl+C=quit",
            id="input-box",
        )
        yield Footer()

    def key_ctrl_c(self) -> None:
        if self._ctrl_c_pending:
            self.exit()
        else:
            self._ctrl_c_pending = True
            log = self.query_one("#trajectory", RichLog)
            log.write("[dim]Press Ctrl+C again to quit[/]")
            self.set_timer(1.5, self._reset_ctrl_c)

    def _reset_ctrl_c(self) -> None:
        self._ctrl_c_pending = False

    def on_mount(self) -> None:
        self.title = f"session: {self.session_id[:16]}..."
        log = self.query_one("#trajectory", RichLog)
        log.write(f"[bold]Session:[/] {self.session_id}")
        sock = _socket_path(self.session_id)
        alive = "✓ live" if sock.exists() else "✗ not found"
        log.write(f"[bold]Socket:[/]  {alive}")
        log.write("─" * 60)
        self._poll_trajectory()

    @work(exclusive=True, group="poll")
    async def _poll_trajectory(self) -> None:
        log = self.query_one("#trajectory", RichLog)
        while True:
            try:
                await self._poll_session(log, self.session_id, prefix="")
                await self._poll_children(log)

                self._poll_count += 1
                if self._poll_count % 5 == 0:
                    usage = await asyncio.to_thread(_fetch_usage, self.session_id)
                    if usage:
                        inp = usage.get("input_tokens", 0)
                        out = usage.get("output_tokens", 0)
                        cache = usage.get("cache_read", 0)
                        children = len(self._child_sessions)
                        child_label = f" | children: {children}" if children else ""
                        self.sub_title = (
                            f"msgs: {self._entry_count} | "
                            f"in: {inp:,} out: {out:,} cache: {cache:,}"
                            f"{child_label}"
                        )

                alive = _socket_path(self.session_id).exists()
                if not alive and self._poll_count > 3:
                    log.write("[bold red]━━━ Session ended ━━━[/]")
                    break
            except Exception as exc:
                log.write(f"[red]poll error: {exc}[/]")

            await asyncio.sleep(3)

    async def _poll_session(
        self, log: RichLog, session_id: str, prefix: str = ""
    ) -> None:
        msgs = await asyncio.to_thread(_fetch_messages, session_id)
        for m in msgs:
            mid = m.get("id", "")
            if not mid or mid in self._seen_ids:
                continue
            self._seen_ids.add(mid)

            child_sid = _extract_child_session_id(m)
            if child_sid and child_sid not in self._child_sessions:
                self._child_sessions[child_sid] = set()
                log.write(
                    f"{prefix}[bold magenta]🔀 child session spawned:[/] "
                    f"{child_sid[:16]}..."
                )
                log.write("")

            payload = m.get("payload", {})
            if not isinstance(payload, dict):
                continue
            role = payload.get("role", "")
            if role == "system":
                continue

            lines = _render_entry(m, prefix=prefix)
            if lines:
                self._entry_count += 1
                for line in lines:
                    log.write(line)
                log.write("")

    async def _poll_children(self, log: RichLog) -> None:
        for child_sid, seen in list(self._child_sessions.items()):
            try:
                msgs = await asyncio.to_thread(_fetch_messages, child_sid)
                for m in msgs:
                    mid = m.get("id", "")
                    if not mid or mid in seen or mid in self._seen_ids:
                        continue
                    seen.add(mid)

                    payload = m.get("payload", {})
                    if not isinstance(payload, dict):
                        continue
                    role = payload.get("role", "")
                    if role == "system":
                        continue

                    lines = _render_entry(m, prefix="  [magenta]│[/] ")
                    if lines:
                        self._entry_count += 1
                        for line in lines:
                            log.write(line)
                        log.write("")
            except Exception:
                pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        log = self.query_one("#trajectory", RichLog)
        ack = await asyncio.to_thread(_send_message, self.session_id, text, "wait")
        log.write(f"[bold green]📨 you → wait:[/] {text}")
        log.write(f"  [dim]{ack}[/]")
        log.write("")

    def action_send_now(self) -> None:
        inp = self.query_one("#input-box", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.value = ""
        log = self.query_one("#trajectory", RichLog)
        ack = _send_message(self.session_id, text, "now")
        log.write(f"[bold yellow]⚡ you → now:[/] {text}")
        log.write(f"  [dim]{ack}[/]")
        log.write("")

    def action_send_interrupt(self) -> None:
        inp = self.query_one("#input-box", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.value = ""
        log = self.query_one("#trajectory", RichLog)
        ack = _send_message(self.session_id, text, "interrupt")
        log.write(f"[bold red]🛑 you → interrupt:[/] {text}")
        log.write(f"  [dim]{ack}[/]")
        log.write("")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: agentm dashboard <session-id>")
        sys.exit(1)
    session_id = sys.argv[1]
    app = Dashboard(session_id)
    app.run()


if __name__ == "__main__":
    main()
