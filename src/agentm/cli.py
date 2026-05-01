"""Minimal AgentM CLI.

After Phase 2.5 the legacy multi-command typer app was removed. This module
provides a small ``agentm`` entry point that boots an :class:`AgentSession`
from a named scenario recipe and runs a single ``prompt(...)`` call against
the Anthropic provider.

The CLI deliberately stays thin: the real value lives in the kernel +
extensions + scenario recipes. Anything richer (dashboards, eval harnesses,
etc.) is now an out-of-tree concern.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agentm.extensions.loader import ScenarioLoadError, load_scenario
from agentm.harness import AgentSession, AgentSessionConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentm",
        description=(
            "AgentM v2 thin CLI: load a scenario recipe, send one prompt, "
            "stream the result."
        ),
    )
    parser.add_argument("prompt", help="User prompt to send to the agent.")
    parser.add_argument(
        "--scenario",
        default="general_purpose",
        help="Scenario recipe name under agentm.extensions.scenarios "
        "(default: general_purpose).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENTM_MODEL", "claude-sonnet-4-5"),
        help="Anthropic model id.",
    )
    parser.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory exposed to extensions (default: cwd).",
    )
    return parser


async def _run(prompt: str, scenario_name: str, model: str, cwd: str) -> int:
    extensions = load_scenario(scenario_name)
    config = AgentSessionConfig(
        cwd=cwd,
        extensions=extensions,
        provider=("agentm.llm.anthropic", {"model": model}),
    )
    session = await AgentSession.create(config)
    try:
        await session.prompt(prompt)
    finally:
        await session.shutdown()
    return 0


def main() -> None:
    """Entry point referenced by the ``agentm`` console script."""

    args = _build_parser().parse_args()
    try:
        rc = asyncio.run(_run(args.prompt, args.scenario, args.model, args.cwd))
    except ScenarioLoadError as exc:
        print(f"agentm: scenario load failed: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(rc)
