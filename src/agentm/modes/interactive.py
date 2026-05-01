"""Minimal interactive TUI for AgentM.

Built on ``rich.live`` (already a transitive dep via typer). Renders the
streaming assistant text in a live panel, prints tool calls/results inline,
and reads user input with ``console.input`` at each prompt boundary. No
multi-pane layout, no keybindings beyond Ctrl-C; this is the smallest
thing that proves the streaming-event plumbing works end-to-end.

Wire-up shape (per pluggable-architecture §5):

* the ``run`` coroutine takes an ``AgentSessionConfig`` and owns the
  session lifecycle (``create`` → loop ``prompt`` → ``shutdown``);
* it subscribes only to public bus channels — ``stream_delta``,
  ``tool_call``, ``tool_result`` — never reaches into harness internals;
* ``rich`` is imported lazily at function call so the SDK stays usable
  without a TUI dependency on import.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import (
    StreamDeltaEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallStart,
)
from agentm.harness import AgentSession, AgentSessionConfig
from agentm.harness.events import (
    ChildSessionEndEvent,
    ChildSessionStartEvent,
)
from agentm.core.abi.events import ToolCallEvent, ToolResultEvent


_QUIT_COMMANDS = frozenset({"/quit", "/exit", "/q", "exit", "quit"})


async def run(config: AgentSessionConfig) -> int:
    """Drive a multi-turn TUI session against ``config``.

    Returns process exit code (``0`` on clean quit, ``1`` on internal
    error). Top-level ``asyncio.CancelledError`` is treated as Ctrl-C.
    """

    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel

    console = Console()
    session = await AgentSession.create(config)

    # Per-turn rendering state. Reset on each ``stream_delta`` with
    # turn_index that exceeds what we last saw, so a single ``run``
    # spans many prompt cycles cleanly.
    pending_text: list[str] = []
    pending_thinking: list[str] = []
    last_turn_index = -1

    def _on_delta(event: StreamDeltaEvent) -> None:
        nonlocal last_turn_index
        if event.turn_index != last_turn_index:
            pending_text.clear()
            pending_thinking.clear()
            last_turn_index = event.turn_index
        delta = event.delta
        if isinstance(delta, TextDelta):
            pending_text.append(delta.text)
        elif isinstance(delta, ThinkingDelta):
            pending_thinking.append(delta.text)
        elif isinstance(delta, ToolCallStart):
            # Show tool name as soon as the model commits to it; args
            # arrive later via ToolCallArgsDelta and are rendered when
            # ``tool_call`` fires below.
            pending_text.append(f"\n[dim]→ {delta.name}…[/dim]\n")

    def _on_tool_call(event: ToolCallEvent) -> None:
        args_preview = _truncate(repr(event.args), 80)
        console.print(f"  [yellow]→ {event.tool_name}[/yellow] {args_preview}")

    def _on_tool_result(event: ToolResultEvent) -> None:
        ok = not getattr(event.result, "is_error", False)
        glyph = "[green]✓[/green]" if ok else "[red]✗[/red]"
        text = _result_preview(event.result)
        console.print(f"  {glyph} {text}")

    def _on_child_start(event: ChildSessionStartEvent) -> None:
        console.print(f"  [cyan]↳ subagent: {event.purpose}[/cyan]")

    def _on_child_end(event: ChildSessionEndEvent) -> None:
        if event.error:
            console.print(f"  [red]subagent failed: {event.error}[/red]")

    session.bus.on("stream_delta", _on_delta)
    session.bus.on("tool_call", _on_tool_call)
    session.bus.on("tool_result", _on_tool_result)
    session.bus.on("child_session_start", _on_child_start)
    session.bus.on("child_session_end", _on_child_end)

    console.print(
        f"[bold]agentm[/bold] interactive — model={config.provider[1].get('model','?')}  "
        f"(/quit to exit)"
    )

    rc = 0
    try:
        while True:
            try:
                text = console.input("\n[bold cyan]> [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not text:
                continue
            if text.lower() in _QUIT_COMMANDS:
                break

            pending_text.clear()
            pending_thinking.clear()

            prompt_task = asyncio.create_task(session.prompt(text))
            try:
                with Live(
                    Panel("", title="assistant", border_style="dim"),
                    console=console,
                    refresh_per_second=20,
                    transient=False,
                ) as live:
                    while not prompt_task.done():
                        live.update(_render_panel(pending_text, pending_thinking))
                        await asyncio.sleep(0.05)
                    live.update(_render_panel(pending_text, pending_thinking))
                await prompt_task
            except KeyboardInterrupt:
                prompt_task.cancel()
                console.print("[red]^C interrupted[/red]")
                try:
                    await prompt_task
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]error: {exc!r}[/red]")
                rc = 1
    finally:
        await session.shutdown()
    return rc


# --- helpers ---------------------------------------------------------------


def _render_panel(text_parts: list[str], thinking_parts: list[str]) -> Any:
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    blocks = []
    if thinking_parts:
        thinking = "".join(thinking_parts).strip()
        if thinking:
            blocks.append(
                Panel(
                    Text(thinking, style="dim italic"),
                    title="thinking",
                    border_style="grey50",
                )
            )
    body = "".join(text_parts) or "[dim]…[/dim]"
    blocks.append(Panel.fit(body, title="assistant", border_style="cyan"))
    return Group(*blocks)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _result_preview(result: Any) -> str:
    """Best-effort one-line summary of a ``ToolResult``."""

    content = getattr(result, "content", None)
    if not content:
        return ""
    first = content[0]
    text = getattr(first, "text", None)
    if isinstance(text, str):
        flat = text.replace("\n", " ⏎ ")
        return _truncate(flat, 100)
    return _truncate(repr(first), 100)


__all__ = ["run"]
