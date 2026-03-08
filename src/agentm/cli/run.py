"""CLI run subcommand — execute an RCA investigation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from rich.console import Console

from agentm.builder import AgentSystemBuilder
from agentm.config.loader import load_scenario_config, load_system_config


console = Console()


async def run_investigation(
    data_dir: str,
    incident: str,
    scenario_dir: str,
    config_path: str,
    debug_mode: bool = False,
    verbose: bool = False,
) -> None:
    """Run the full RCA investigation."""
    import agentm.tools.observability as obs_tools

    # Apply env-var overrides
    if api_key := os.environ.get("AGENTM_API_KEY"):
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url := os.environ.get("AGENTM_API_BASE_URL"):
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    project_root = Path(config_path).resolve().parent.parent
    scenario_path = Path(scenario_dir)
    if not scenario_path.is_absolute():
        scenario_path = project_root / scenario_path

    # Load configs
    system_config = load_system_config(config_path)
    scenario_config = load_scenario_config(scenario_path / "scenario.yaml")

    # Override from env
    if model := os.environ.get("AGENTM_ORCHESTRATOR_MODEL"):
        scenario_config.orchestrator.model = model
    if model := os.environ.get("AGENTM_WORKER_MODEL"):
        scenario_config.agents["worker"].model = model

    # Override debug settings from CLI flags
    if debug_mode:
        system_config.debug.console_live = True
    if verbose:
        system_config.debug.verbose = True

    # Initialize data
    console.rule("AgentM — RCA Investigation")
    console.print(f"Data directory: [cyan]{data_dir}[/]")
    console.print(f"Orchestrator: [cyan]{scenario_config.orchestrator.model}[/]")
    console.print(f"Worker: [cyan]{scenario_config.agents['worker'].model}[/]")

    result = obs_tools.set_data_directory(data_dir)
    init_info = json.loads(result)
    if "error" in init_info:
        console.print(f"[red]ERROR: {init_info['error']}[/]")
        sys.exit(1)
    console.print(f"Data initialized: {len(init_info['files'])} parquet files")

    # Build
    system = AgentSystemBuilder.build(
        system_type="hypothesis_driven",
        scenario_config=scenario_config,
        system_config=system_config,
        scenario_dir=scenario_path,
    )

    # Set up debug console if requested
    debug_console = None
    if system_config.debug.console_live:
        from agentm.core.debug_console import DebugConsole
        debug_console = DebugConsole(verbose=verbose)
        debug_console.start()

    # Stream execution
    initial_state = {"messages": [HumanMessage(content=incident)]}
    console.rule("Starting investigation")
    console.print(f"Incident: {incident[:200]}...")
    console.print()

    step = 0
    try:
        async for event in system.stream(
            initial_state,
            on_event=debug_console.on_event if debug_console else None,
        ):
            step += 1
            if not system_config.debug.console_live:
                _print_event(event, step, verbose)
            if step > 100:
                console.print("\n[yellow][!] Reached step limit (100), stopping.[/]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow][!] Investigation interrupted by user.[/]")
    except Exception as e:
        console.print(f"\n[red][ERROR] {e}[/]")
        if verbose:
            import traceback
            traceback.print_exc()
    finally:
        if debug_console is not None:
            debug_console.stop()
        if system.trajectory is not None:
            path = await system.trajectory.close()
            console.print()
            console.rule("Investigation complete")
            console.print(f"Steps: {step}")
            if path:
                console.print(f"Trajectory saved: [green]{path}[/]")
            console.print(f"Thread ID: [dim]{system.thread_id}[/]")


def _print_event(event: dict, step: int, verbose: bool) -> None:
    """Print a streaming event in non-debug mode."""
    for node_name, node_data in event.items():
        if node_name == "__interrupt__":
            continue
        messages = node_data.get("messages", [])
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")

            if role == "ai":
                if content:
                    console.print(f"\n[bold cyan][Orchestrator step {step}][/]")
                    limit = 2000 if verbose else 500
                    console.print(content[:limit])
                    if len(content) > limit:
                        console.print(f"  [dim]... ({len(content)} chars total)[/]")

                tool_calls = getattr(msg, "tool_calls", [])
                for tc in tool_calls:
                    args_str = json.dumps(tc["args"], ensure_ascii=False)
                    console.print(f"\n  [yellow]-> {tc['name']}[/]({args_str[:300]})")

            elif role == "tool":
                tool_name = getattr(msg, "name", "?")
                if content:
                    preview = content[:500] if verbose else content[:200]
                    console.print(f"\n  [green]<- {tool_name}:[/] {preview}")
                    if len(content) > len(preview):
                        console.print(f"     [dim]... ({len(content)} chars total)[/]")
