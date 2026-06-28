"""TEL prompt evolution: aggregate reflections → edit prompt files directly.

Usage::

    llmharness-evolve --reflections ./failed_cases/reflections/
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

_PROMPTS_DIR = Path(__file__).parents[2] / "agents" / "tel" / "prompts"

app = typer.Typer(help="Evolve TEL prompts from reflection reports.")


def _load_reflections(directory: Path) -> list[dict[str, str | float]]:
    """Read all reflection markdown files, sorted by F1 ascending (worst first)."""
    import re

    reflections: list[dict[str, str | float]] = []
    f1_pat = re.compile(r"F1=([\d.]+)")
    for md in sorted(directory.glob("*.md")):
        text = md.read_text(encoding="utf-8").strip()
        if not text:
            continue
        f1 = 1.0
        m = f1_pat.search(text.split("\n", 1)[0])
        if m:
            f1 = float(m.group(1))
        reflections.append({"instance_id": md.stem, "content": text, "f1": f1})
    reflections.sort(key=lambda r: r["f1"])
    return reflections


async def _run_evolve(
    reflections: list[dict[str, str | float]],
    provider: tuple[str, dict[str, Any]] | None,
    cwd: str,
) -> str:
    """Create an AgentM session that edits prompt files directly."""
    import contextlib

    from agentm.core.abi import AgentSessionConfig, AssistantMessage, TextContent
    from agentm.core.runtime import AgentSession

    evolve_prompt = (_PROMPTS_DIR / "evolve.md").read_text(encoding="utf-8")

    reflections_block = "\n\n---\n\n".join(
        f"### Case: {r['instance_id']} (F1={r['f1']:.3f})\n\n{r['content']}"
        for r in reflections
    )

    notepad_path = _PROMPTS_DIR / "notepad.md"
    reason_path = _PROMPTS_DIR / "reason.md"

    user_message = (
        f"{evolve_prompt}\n\n"
        f"# Reflection reports ({len(reflections)} cases)\n\n"
        f"{reflections_block}\n\n"
        f"# Prompt file paths\n\n"
        f"Read and edit these files directly:\n"
        f"- `{notepad_path}`\n"
        f"- `{reason_path}`\n\n"
        f"After editing, summarise what you changed and why."
    )

    config = AgentSessionConfig(
        cwd=cwd,
        provider=provider,
        extensions=[
            ("agentm.extensions.builtin.observability", {}),
            ("agentm.extensions.builtin.operations", {}),
            ("agentm.extensions.builtin.file_tools", {}),
            ("agentm.extensions.builtin.tool_bash", {}),
        ],
        purpose="tel_evolve",
        auto_commit=False,
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
    summary_file: Annotated[
        Path | None, typer.Option("--summary-file", help="Write raw agent summary to this file")
    ] = None,
) -> None:
    """Read reflection reports and evolve prompt files in-place."""
    import os

    try:
        from agentm.env import autoload_dotenv

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

    typer.echo(f"Loaded {len(reflection_data)} reflections")

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
    summary = asyncio.run(
        _run_evolve(reflection_data, resolved_provider, str(cwd.resolve()))
    )

    if summary_file:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(summary, encoding="utf-8")

    typer.echo(f"\n{summary}")
    typer.echo(
        f"\nPrompt files edited in-place:\n"
        f"  {_PROMPTS_DIR / 'notepad.md'}\n"
        f"  {_PROMPTS_DIR / 'reason.md'}\n"
        f"Review changes with: git diff"
    )


def main() -> None:
    app()


__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
