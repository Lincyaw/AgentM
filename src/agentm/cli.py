"""AgentM CLI (typer-based).

Loads a scenario recipe, sends one prompt, and prints the assistant's final
text output. Any richer surface (dashboards, eval harnesses, batch runs)
remains an out-of-tree concern — this CLI is intentionally thin.
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated

import typer

from agentm.core.kernel.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
)
from agentm.extensions.loader import ScenarioLoadError, load_scenario
from agentm.harness import AgentSession, AgentSessionConfig

def _print_final(final_messages: list) -> None:
    text_blocks: list[str] = []
    tool_calls = 0
    for msg in final_messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_blocks.append(block.text)
                elif isinstance(block, ToolCallBlock):
                    tool_calls += 1

    typer.echo("\n" + "=" * 60)
    typer.echo("AGENT FINAL OUTPUT")
    typer.echo("=" * 60)
    typer.echo("\n\n".join(text_blocks) if text_blocks else "<no text output>")
    typer.echo("=" * 60)
    typer.echo(f"messages={len(final_messages)} tool_calls={tool_calls}")


async def _run(
    prompt: str,
    scenario_name: str,
    model: str,
    cwd: str,
    quiet: bool,
) -> int:
    extensions = load_scenario(scenario_name)
    config = AgentSessionConfig(
        cwd=cwd,
        extensions=extensions,
        provider=("agentm.llm.anthropic", {"model": model}),
    )
    session = await AgentSession.create(config)
    try:
        final = await session.prompt(prompt)
    finally:
        await session.shutdown()

    if not quiet:
        _print_final(final)
    return 0


def run_cmd(
    prompt: Annotated[str, typer.Argument(help="User prompt to send to the agent.")],
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario",
            help="Scenario recipe name under agentm.extensions.scenarios.",
        ),
    ] = "general_purpose",
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Model id passed to the Anthropic-compatible provider.",
        ),
    ] = os.environ.get("AGENTM_MODEL", "claude-sonnet-4-6"),
    cwd: Annotated[
        str,
        typer.Option("--cwd", help="Working directory exposed to extensions."),
    ] = os.getcwd(),
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress final-output printout."),
    ] = False,
) -> None:
    """Send a single prompt and print the agent's final text."""

    try:
        rc = asyncio.run(_run(prompt, scenario, model, cwd, quiet))
    except ScenarioLoadError as exc:
        typer.echo(f"agentm: scenario load failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    raise typer.Exit(code=rc)


def main() -> None:
    """Entry point referenced by the ``agentm`` console script."""

    typer.run(run_cmd)
