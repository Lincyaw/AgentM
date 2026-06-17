"""TEL prompt evolution: aggregate reflections → propose prompt changes.

Usage::

    llmharness-evolve --reflections ./failed_cases/reflections/
    llmharness-evolve --reflections ./reflections/ --apply
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "agents" / "tel" / "prompts"

app = typer.Typer(help="Evolve TEL prompts from reflection reports.")


def _load_reflections(directory: Path) -> list[dict[str, str]]:
    """Read all reflection markdown files from *directory*."""
    reflections = []
    for md in sorted(directory.glob("*.md")):
        text = md.read_text(encoding="utf-8").strip()
        if text:
            reflections.append({"instance_id": md.stem, "content": text})
    return reflections


def _load_current_prompts() -> dict[str, str]:
    """Read current notepad/reason prompts."""
    prompts = {}
    for name in ("notepad", "reason"):
        p = _PROMPTS_DIR / f"{name}.md"
        if p.is_file():
            prompts[name] = p.read_text(encoding="utf-8")
    return prompts


async def _run_evolve(
    reflections: list[dict[str, str]],
    current_prompts: dict[str, str],
    provider: tuple[str, dict[str, Any]] | None,
    cwd: str,
) -> str:
    """Create an AgentM session to aggregate reflections and propose changes."""
    import contextlib

    from agentm.core.abi import AgentSessionConfig, AssistantMessage, TextContent
    from agentm.core.runtime import AgentSession

    evolve_prompt = (_PROMPTS_DIR / "evolve.md").read_text(encoding="utf-8")

    reflections_block = "\n\n---\n\n".join(
        f"### Case: {r['instance_id']}\n\n{r['content']}" for r in reflections
    )

    prompts_block = "\n\n---\n\n".join(
        f"### {name}.md\n\n```markdown\n{content}\n```"
        for name, content in current_prompts.items()
    )

    user_message = (
        f"{evolve_prompt}\n\n"
        f"# Reflection reports ({len(reflections)} cases)\n\n"
        f"{reflections_block}\n\n"
        f"# Current prompts\n\n"
        f"{prompts_block}"
    )

    config = AgentSessionConfig(
        cwd=cwd,
        provider=provider,
        extensions=[
            ("agentm.extensions.builtin.observability", {}),
            ("agentm.extensions.builtin.operations", {}),
        ],
        purpose="tel_evolve",
    )

    session = await AgentSession.create(config)

    try:
        messages = await session.prompt(user_message)
    finally:
        with contextlib.suppress(Exception):
            await session.shutdown()

    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            parts = [
                b.text for b in msg.content
                if isinstance(b, TextContent) and b.text.strip()
            ]
            if parts:
                return "\n".join(parts)

    return "(no output from evolve agent)"


@app.command()
def evolve(
    reflections: Annotated[
        Path, typer.Option("--reflections", help="Directory containing reflection .md files")
    ],
    cwd: Annotated[Path, typer.Option("--cwd", help="Working directory")] = Path("."),
    provider: Annotated[
        str | None, typer.Option("--provider", help="LLM provider spec")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="config.toml profile name")
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write proposal to file (default: stdout)"),
    ] = None,
) -> None:
    """Read reflection reports and propose prompt improvements."""
    import os

    try:
        from agentm.cli import autoload_dotenv
        autoload_dotenv()
    except ImportError:
        pass

    if not reflections.is_dir():
        typer.echo(f"Error: {reflections} is not a directory", err=True)
        raise typer.Exit(1)

    reflection_data = _load_reflections(reflections)
    if not reflection_data:
        typer.echo(f"No reflection files found in {reflections}", err=True)
        raise typer.Exit(1)

    current_prompts = _load_current_prompts()
    typer.echo(
        f"Loaded {len(reflection_data)} reflections, "
        f"{len(current_prompts)} current prompts"
    )

    resolved_provider: tuple[str, dict[str, Any]] | None = None
    if model:
        try:
            from agentm.ai import DEFAULT_PROVIDER_REGISTRY
            from agentm.core.lib import resolve_model_profile

            profile = resolve_model_profile(model)
            if profile is not None:
                resolved_provider = DEFAULT_PROVIDER_REGISTRY.build(
                    profile.provider, profile.to_build_config(), env=os.environ
                )
        except ImportError:
            pass
    if resolved_provider is None and provider:
        if ":" in provider:
            mod, payload = provider.split(":", 1)
            resolved_provider = mod.strip(), json.loads(payload)
        else:
            resolved_provider = provider, {}
    if resolved_provider is None:
        try:
            from agentm.ai import DEFAULT_PROVIDER_REGISTRY
            from agentm.core.lib import resolve_model_profile

            profile = resolve_model_profile(None)
            if profile is not None:
                resolved_provider = DEFAULT_PROVIDER_REGISTRY.build(
                    profile.provider, profile.to_build_config(), env=os.environ
                )
        except ImportError:
            pass

    typer.echo("Running evolve agent…")
    proposal = asyncio.run(
        _run_evolve(
            reflection_data,
            current_prompts,
            resolved_provider,
            str(cwd.resolve()),
        )
    )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(proposal, encoding="utf-8")
        typer.echo(f"\nProposal written to {output}")
    else:
        typer.echo("\n" + "=" * 60)
        typer.echo(proposal)
        typer.echo("=" * 60)

    typer.echo(
        f"\nPrompt files at:\n"
        f"  {_PROMPTS_DIR / 'notepad.md'}\n"
        f"  {_PROMPTS_DIR / 'reason.md'}\n"
        f"Review the proposals above and edit the prompts manually."
    )


def main() -> None:
    app()


__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
