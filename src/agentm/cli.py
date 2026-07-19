"""Interactive CLI — typer + rich, streaming thinking display.

Usage::

    agentm-chat                          # default scenario + config
    agentm-chat -s interrupt_demo        # named scenario from manifest.yaml

Ctrl+C interrupts the current turn. Ctrl+D or 'exit' quits.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown

from agentm import AgentSession, AgentSessionConfig
from agentm.config.resolver import DefaultSessionSpecResolver
from agentm.core.abi.messages import TextContent, ThinkingBlock, ToolCallBlock
from agentm.core.abi.stream import TextDelta, ThinkingDelta
from agentm.core.abi.events import StreamDeltaEvent
from agentm.core.abi.termination import SignalAborted
from agentm.scenarios import builtin_scenario_loader

app = typer.Typer(add_completion=False)
console = Console()


class _SessionStats:
    """Tracks cumulative session statistics."""

    def __init__(self) -> None:
        self.turns = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

    def update_from_turn(self, turn: Any) -> None:
        self.turns += 1
        meta = turn.meta
        self.input_tokens += meta.total_input_tokens
        self.output_tokens += meta.total_output_tokens
        self.cache_read_tokens += meta.cache_read_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def title_string(self) -> str:
        return (
            f"agentm | turn {self.turns} | "
            f"↑{self.input_tokens:,} ↓{self.output_tokens:,} "
            f"(cache:{self.cache_read_tokens:,})"
        )

    def status_line(self) -> str:
        return (
            f"turn {self.turns} | "
            f"in:{self.input_tokens:,} out:{self.output_tokens:,} "
            f"cache:{self.cache_read_tokens:,} total:{self.total_tokens:,}"
        )


class _StreamCollector:
    """Collects streaming deltas for live display."""

    def __init__(self, stats: _SessionStats) -> None:
        self.thinking_buf: list[str] = []
        self.text_buf: list[str] = []
        self._live: Live | None = None
        self._stats = stats

    def attach_live(self, live: Live) -> None:
        self._live = live

    def on_delta(self, event: StreamDeltaEvent) -> None:
        delta = event.delta
        if isinstance(delta, ThinkingDelta):
            self.thinking_buf.append(delta.text)
            self._refresh()
        elif isinstance(delta, TextDelta):
            self.text_buf.append(delta.text)
            self._refresh()

    def _refresh(self) -> None:
        if self._live is None:
            return
        parts: list[Any] = []
        if self.thinking_buf:
            thinking_text = "".join(self.thinking_buf)
            lines = thinking_text.split("\n")
            if len(lines) > 6:
                lines = lines[-6:]
            preview = "\n".join(lines)
            parts.append(Text(f"💭 {preview}", style="dim italic"))
        if self.text_buf:
            parts.append(Text("".join(self.text_buf)))
        if parts:
            from rich.console import Group
            self._live.update(Group(*parts))

    def reset(self) -> None:
        self.thinking_buf.clear()
        self.text_buf.clear()


async def _run(
    scenario: str | None,
    extensions: list[str],
    project_config: str | None,
    user_config: str | None,
    system_prompt: str | None,
) -> None:
    if project_config is None:
        candidate = Path.cwd() / "agentm.toml"
        project_config = str(candidate) if candidate.exists() else None

    resolver = DefaultSessionSpecResolver(
        project_config=project_config,
        user_config=user_config,
    )
    config = AgentSessionConfig(
        scenario=scenario,
        scenario_loader=builtin_scenario_loader,
        spec_resolver=resolver,
        extra_extensions=[(mod, {}) for mod in extensions],
        system=system_prompt,
    )
    session = await AgentSession.create(config)
    session.start()

    loop = asyncio.get_running_loop()
    interrupted = False

    def _on_sigint() -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        session.interrupt("user_cancel")

    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    stats = _SessionStats()
    collector = _StreamCollector(stats)
    session.on(StreamDeltaEvent.CHANNEL, collector.on_delta)

    spec = resolver.resolve(config)
    name = spec.provider_identity.name if spec.provider_identity else "?"
    model_id = spec.provider_identity.model_id if spec.provider_identity else "?"
    console.print(f"[dim]session {session.session_id} | {name}:{model_id}[/dim]")
    console.print("[dim]Ctrl+C = interrupt | Ctrl+D or 'exit' = quit[/dim]\n")

    try:
        while True:
            try:
                text = await asyncio.to_thread(
                    console.input, "[bold green]you>[/bold green] "
                )
            except EOFError:
                break
            except KeyboardInterrupt:
                continue

            text = text.strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit", "/quit", "/exit"):
                break

            interrupted = False
            collector.reset()

            with Live(Text("..."), console=console, refresh_per_second=12, transient=True) as live:
                collector.attach_live(live)
                try:
                    receipt = await session.prompt(text)
                    await receipt.wait()
                except Exception as exc:
                    console.print(f"[red]error: {exc}[/red]\n")
                    collector.attach_live(None)
                    continue
                collector.attach_live(None)

            turns = session.get_turns()
            if not turns:
                continue
            last = turns[-1]
            stats.update_from_turn(last)
            console.set_window_title(stats.title_string())

            if isinstance(last.outcome.cause, SignalAborted):
                console.print(f"[yellow]⚡ interrupted: {last.outcome.cause.reason}[/yellow]")
                for rnd in last.rounds:
                    for tr in rnd.tool_results:
                        t = "".join(b.text for b in tr.result.content if isinstance(b, TextContent))
                        if t:
                            err = " ERROR" if tr.result.is_error else ""
                            console.print(f"[dim]  [tool{err}] {t}[/dim]")
                console.print(f"[dim]─ {stats.status_line()}[/dim]\n")
                continue

            for rnd in last.rounds:
                thinking_parts: list[str] = []
                text_parts: list[str] = []
                for block in rnd.response.content:
                    if isinstance(block, ThinkingBlock):
                        thinking_parts.append(block.text)
                    elif isinstance(block, TextContent):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolCallBlock):
                        console.print(f"[dim]⚙ {block.name}({dict(block.arguments)})[/dim]")

                if thinking_parts:
                    thinking_text = "\n".join(thinking_parts)
                    lines = thinking_text.strip().split("\n")
                    if len(lines) > 8:
                        lines = lines[:3] + ["  ..."] + lines[-3:]
                    console.print(f"[dim italic]💭 {'\\n   '.join(lines)}[/dim italic]")

                if text_parts:
                    console.print("\n".join(text_parts))

                for tr in rnd.tool_results:
                    t = "".join(b.text for b in tr.result.content if isinstance(b, TextContent))
                    if tr.result.is_error:
                        console.print(f"[red]  ✗ {t}[/red]")
                    else:
                        preview = t[:200] + ("..." if len(t) > 200 else "")
                        console.print(f"[dim]  → {preview}[/dim]")
            console.print(f"[dim]─ {stats.status_line()}[/dim]\n")
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        await session.shutdown()


@app.command()
def chat(
    scenario: Optional[str] = typer.Option(None, "-s", "--scenario", help="Named scenario"),
    extension: Optional[list[str]] = typer.Option(None, "-e", "--extension", help="Extra extension modules"),
    project_config: Optional[str] = typer.Option(None, "--project-config", help="Project config TOML path"),
    user_config: Optional[str] = typer.Option(None, "--user-config", help="User config TOML path"),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt override"),
) -> None:
    """Interactive agent chat session."""
    asyncio.run(_run(
        scenario=scenario,
        extensions=extension or [],
        project_config=project_config,
        user_config=user_config,
        system_prompt=system,
    ))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
