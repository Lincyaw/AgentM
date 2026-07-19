"""``agentm scenario`` — scenario discovery and inspection."""

from __future__ import annotations

import json
import sys

import typer

from agentm.core.abi.session_api import (
    ExtensionSpec,
    ScenarioSpec,
    normalize_extension_spec,
)
from agentm.core.lib.redact import redact_config
from agentm.scenarios import builtin_scenario_loader, packaged_scenario_names

from agentm.cli._display import EXIT_NOT_FOUND, stderr_console

scenario_app = typer.Typer(
    name="scenario",
    help="Scenario discovery and inspection.",
    no_args_is_help=True,
    add_completion=False,
)


@scenario_app.command("list")
def list_scenarios(
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
) -> None:
    """List available scenarios.

    Discovers built-in scenarios and manifest.yaml files under
    contrib/scenarios/.

    Examples:

        agentm scenario list

        agentm scenario list --format json
    """
    names = packaged_scenario_names()

    if format == "json":
        sys.stdout.write(json.dumps(list(names)) + "\n")
        return

    if not names:
        stderr_console.print("[dim]No scenarios found.[/dim]")
        return

    from rich.console import Console
    console = Console()
    for name in names:
        console.print(f"  {name}")


@scenario_app.command("show")
def show_scenario(
    name: str = typer.Argument(..., help="Scenario name (e.g. 'chat', 'minimal', 'arl:harbor')"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
) -> None:
    """Show a scenario's extensions and base directory.

    Examples:

        agentm scenario show minimal

        agentm scenario show chat --format json
    """
    try:
        spec = builtin_scenario_loader(name)
    except ValueError as exc:
        stderr_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)

    extensions = tuple(
        normalize_extension_spec(extension)
        for extension in (
            spec.extensions
            if isinstance(spec, ScenarioSpec)
            else spec
        )
    )

    result = {
        "name": name,
        "base_dir": spec.base_dir if isinstance(spec, ScenarioSpec) else None,
        "extensions": [
            _extension_record(extension)
            for extension in extensions
        ],
    }

    if format == "json":
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"[bold]scenario:[/bold] {name}")
    if isinstance(spec, ScenarioSpec) and spec.base_dir:
        console.print(f"[bold]base_dir:[/bold] {spec.base_dir}")

    table = Table(title="Extensions", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Module", style="cyan")
    table.add_column("Config")
    for i, extension in enumerate(extensions, 1):
        cfg = redact_config(extension.config)
        cfg_str = json.dumps(cfg, ensure_ascii=False) if cfg else ""
        table.add_row(str(i), _extension_label(extension), cfg_str)
    console.print(table)


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
