"""``agentm config`` — configuration inspection subcommands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from agentm import AgentSessionConfig
from agentm.config.resolver import DefaultSessionSpecResolver
from agentm.core.abi.session_api import ExtensionSpec
from agentm.core.lib.redact import redact_config
from agentm.scenarios import builtin_scenario_loader

from agentm.cli._display import EXIT_ERROR, stderr_console

config_app = typer.Typer(
    name="config",
    help="Configuration inspection and scaffolding.",
    no_args_is_help=True,
    add_completion=False,
)


@config_app.command("show")
def show(
    scenario: Optional[str] = typer.Option(None, "-s", "--scenario", help="Scenario to resolve"),
    project_config: Optional[str] = typer.Option(None, "--project-config", help="Project config TOML path"),
    user_config: Optional[str] = typer.Option(None, "--user-config", help="User config TOML path"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
) -> None:
    """Display the resolved session configuration.

    Shows what scenario, provider, extensions, and atom config would be
    used for a session with the given options. Useful for debugging the
    config resolution chain.

    Examples:

        agentm config show

        agentm config show -s chat --format json

        agentm config show --project-config ./agentm.toml
    """
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    if project_config is None:
        candidate = Path.cwd() / "agentm.toml"
        project_config = str(candidate) if candidate.exists() else None

    resolver = DefaultSessionSpecResolver(
        project_config=project_config,
        user_config=user_config,
    )
    config = AgentSessionConfig(
        cwd=str(Path.cwd()),
        scenario=scenario,
        scenario_loader=builtin_scenario_loader,
    )

    try:
        spec = resolver.resolve(config)
    except Exception as exc:
        stderr_console.print(f"[red]error: config resolution failed: {exc}[/red]")
        raise typer.Exit(EXIT_ERROR)

    result = {
        "scenario": spec.scenario,
        "provider": {
            "name": spec.provider_identity.name if spec.provider_identity else None,
            "model": spec.provider_identity.model_id if spec.provider_identity else None,
            "source": (
                _extension_record(spec.provider)
                if spec.provider is not None
                else None
            ),
        },
        "extensions": [
            _extension_record(extension)
            for extension in spec.extensions
        ],
        "atom_config": dict(spec.atom_config),
        "provenance": [
            {
                "path": p.path,
                "source": p.source,
                "source_ref": p.source_ref,
            }
            for p in spec.value_provenance
        ],
    }

    if format == "json":
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
        return

    from rich.table import Table
    from rich.console import Console

    console = Console()

    console.print(f"[bold]scenario:[/bold] {spec.scenario or '(none)'}")

    if spec.provider_identity:
        console.print(
            f"[bold]provider:[/bold] {spec.provider_identity.name} "
            f"model={spec.provider_identity.model_id or '?'}"
        )
    else:
        console.print("[bold]provider:[/bold] (none)")

    if spec.extensions:
        table = Table(title="Extensions", show_lines=False)
        table.add_column("Module", style="cyan")
        table.add_column("Config")
        for extension in spec.extensions:
            cfg = redact_config(extension.config)
            cfg_str = json.dumps(cfg, ensure_ascii=False) if cfg else ""
            table.add_row(_extension_label(extension), cfg_str)
        console.print(table)

    if spec.atom_config:
        table = Table(title="Atom Config", show_lines=False)
        table.add_column("Atom", style="cyan")
        table.add_column("Config")
        for name, atom_config in spec.atom_config.items():
            table.add_row(
                name,
                json.dumps(dict(atom_config), ensure_ascii=False),
            )
        console.print(table)

    if spec.value_provenance:
        table = Table(title="Provenance", show_lines=False)
        table.add_column("Path", style="cyan")
        table.add_column("Source", style="dim")
        table.add_column("Ref")
        for p in spec.value_provenance:
            table.add_row(p.path, p.source, p.source_ref or "")
        console.print(table)


@config_app.command("init")
def init(
    path: Optional[str] = typer.Argument(None, help="Output path (default: ./agentm.toml)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing file"),
) -> None:
    """Create a starter agentm.toml in the current directory.

    Examples:

        agentm config init

        agentm config init ./my-project/agentm.toml --force
    """
    target = Path(path) if path else Path.cwd() / "agentm.toml"

    if target.exists() and not force:
        stderr_console.print(f"[red]error: {target} already exists (use --force to overwrite)[/red]")
        raise typer.Exit(EXIT_ERROR)

    template = """\
# AgentM project configuration
# Docs: https://github.com/AoyangSpace/AgentM

# Default scenario (contrib/scenarios/<name>/manifest.yaml)
# default_scenario = "minimal"

# Default provider name — matches a [providers.<name>] section below
# default_provider = "openai"

# [providers.openai]
# model = "gpt-4.1"
# api_key_env = "OPENAI_API_KEY"

# [providers.anthropic]
# model = "claude-sonnet-4-20250514"
# api_key_env = "ANTHROPIC_API_KEY"

# Per-atom config overrides
# [atoms.system_prompt]
# prompt = "You are a helpful assistant."
"""
    target.write_text(template, encoding="utf-8")
    stderr_console.print(f"[green]created {target}[/green]")


def _extension_record(extension: ExtensionSpec) -> dict[str, object]:
    return {
        "kind": extension.source.kind,
        "location": extension.source.location,
        "digest": extension.source.digest,
        "module": extension.module_path,
        "config": redact_config(extension.config),
    }


def _extension_label(extension: ExtensionSpec) -> str:
    if extension.source.kind == "module":
        return extension.source.location
    return f"{extension.source.location} ({extension.source.digest})"
