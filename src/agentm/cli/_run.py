"""``agentm run`` — non-interactive single-shot execution."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from agentm import AgentSession, AgentSessionConfig
from agentm.config.resolver import DefaultSessionSpecResolver
from agentm.core.abi.messages import TextContent
from agentm.scenarios import builtin_scenario_loader

from agentm.cli._display import (
    EXIT_CANCELLED,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    SessionStats,
    stderr_console,
)
from agentm.cli._store import resolve_trajectory_store_or_create


async def _execute(
    prompt_text: str,
    *,
    scenario: str | None,
    extensions: list[str],
    project_config: str | None,
    user_config: str | None,
    system_prompt: str | None,
    output_format: str,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    if project_config is None:
        candidate = Path.cwd() / "agentm.toml"
        project_config = str(candidate) if candidate.exists() else None

    resolver = DefaultSessionSpecResolver(
        project_config=project_config,
        user_config=user_config,
    )
    store = resolve_trajectory_store_or_create()
    config = AgentSessionConfig(
        cwd=str(Path.cwd()),
        scenario=scenario,
        scenario_loader=builtin_scenario_loader,
        spec_resolver=resolver,
        extra_extensions=[(mod, {}) for mod in extensions],
        system=system_prompt,
        store=store,
    )

    try:
        session = await AgentSession.create(config)
    except Exception as exc:
        stderr_console.print(f"[red]error: session creation failed: {exc}[/red]")
        raise typer.Exit(EXIT_ERROR)

    session.start()
    stats = SessionStats()

    try:
        receipt = await session.prompt(prompt_text)
        await receipt.wait()
    except Exception as exc:
        stderr_console.print(f"[red]error: {exc}[/red]")
        await session.shutdown()
        raise typer.Exit(EXIT_ERROR)

    turns = session.get_turns()
    await session.shutdown()

    if not turns:
        stderr_console.print("[yellow]warning: no turns produced[/yellow]")
        raise typer.Exit(EXIT_ERROR)

    last = turns[-1]
    stats.update_from_turn(last)

    text_parts: list[str] = []
    for rnd in last.rounds:
        for block in rnd.response.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
    text = "\n".join(text_parts)

    if output_format == "json":
        result = {
            "text": text,
            "session_id": session.session_id,
            "stats": stats.to_dict(),
        }
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(text)
        if text and not text.endswith("\n"):
            sys.stdout.write("\n")

    stderr_console.print(f"[dim]{stats.status_line()}[/dim]")


def run(
    message: Optional[str] = typer.Option(None, "-m", "--message", help="Prompt text (reads stdin if omitted)"),
    scenario: Optional[str] = typer.Option(None, "-s", "--scenario", help="Named scenario"),
    extension: Optional[list[str]] = typer.Option(None, "-e", "--extension", help="Extra extension modules"),
    project_config: Optional[str] = typer.Option(None, "--project-config", help="Project config TOML path"),
    user_config: Optional[str] = typer.Option(None, "--user-config", help="User config TOML path"),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt override"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
) -> None:
    """Run a single prompt and exit.

    Reads the prompt from -m or stdin. Writes the assistant response to
    stdout (text) or structured JSON (--format json). Token stats go to
    stderr.

    Examples:

        agentm run -m "hello"

        echo "summarize this" | agentm run -s minimal

        agentm run -m "translate" --system "You are a translator." --format json
    """
    if format not in ("text", "json"):
        stderr_console.print(f"[red]error: --format must be 'text' or 'json', got {format!r}[/red]")
        raise typer.Exit(EXIT_USAGE)

    if message is None:
        if sys.stdin.isatty():
            stderr_console.print("[red]error: no message — pass -m or pipe stdin[/red]")
            raise typer.Exit(EXIT_USAGE)
        message = sys.stdin.read().strip()

    if not message:
        stderr_console.print("[red]error: empty message[/red]")
        raise typer.Exit(EXIT_USAGE)

    asyncio.run(_execute(
        message,
        scenario=scenario,
        extensions=extension or [],
        project_config=project_config,
        user_config=user_config,
        system_prompt=system,
        output_format=format,
    ))
