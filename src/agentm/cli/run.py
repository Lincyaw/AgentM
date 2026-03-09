"""CLI run subcommand — execute an RCA investigation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

import uvicorn
from langchain_core.messages import HumanMessage
from rich.console import Console

import agentm.tools.observability as obs_tools
from agentm.builder import AgentSystemBuilder
from agentm.config.loader import load_scenario_config, load_system_config
from agentm.core.debug_console import DebugConsole
from agentm.server.app import broadcast_event, create_dashboard_app


console = Console()


async def run_investigation(
    data_dir: str,
    incident: str,
    scenario_dir: str,
    config_path: str,
    debug_mode: bool = False,
    verbose: bool = False,
    dashboard: bool = False,
    dashboard_port: int = 8765,
    dashboard_host: str = "127.0.0.1",
    max_steps: int = 100,
) -> None:
    """Run the full RCA investigation."""
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

    # Resolve relative prompt paths against scenario_dir
    _resolve_prompt_paths(scenario_config, scenario_path)

    # Build
    system = AgentSystemBuilder.build(
        system_type="hypothesis_driven",
        scenario_config=scenario_config,
        system_config=system_config,
    )

    # Set up debug console if requested
    debug_console = None
    if system_config.debug.console_live:
        debug_console = DebugConsole(verbose=verbose)
        debug_console.start()

    # Set up dashboard server if requested
    dashboard_server_task = None
    dashboard_broadcast = None
    if dashboard:
        app = create_dashboard_app(
            graph=system.graph,
            scenario_config=system.scenario_config,
            task_manager=system.task_manager,
            trajectory=system.trajectory,
            thread_id=system.thread_id,
        )
        if system.task_manager is not None:
            system.task_manager.set_broadcast_callback(broadcast_event)
        dashboard_broadcast = broadcast_event

        uvi_config = uvicorn.Config(
            app, host=dashboard_host, port=dashboard_port, log_level="warning",
        )
        server = uvicorn.Server(uvi_config)
        dashboard_server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)
        console.print(f"Dashboard: [link=http://localhost:{dashboard_port}]http://localhost:{dashboard_port}[/link]")

    # Register trajectory listeners — all events flow through TrajectoryCollector
    if system.trajectory is not None:
        if debug_console is not None:
            system.trajectory.add_listener(debug_console.on_trajectory_event)
        if dashboard_broadcast is not None:
            async def _traj_to_ws(event: dict) -> None:
                await dashboard_broadcast({
                    "event_type": event.get("event_type", ""),
                    "agent_path": event.get("agent_path", []),
                    "data": event.get("data", {}),
                    "timestamp": event.get("timestamp", ""),
                    "mode": "trajectory",
                })
            system.trajectory.add_listener(_traj_to_ws)

    # Stream execution
    initial_state = {"messages": [HumanMessage(content=incident)]}
    console.rule("Starting investigation")
    console.print(f"Incident: {incident[:200]}...")
    console.print()

    step = 0
    try:
        async for event in system.stream(initial_state):
            step += 1
            if not system_config.debug.console_live:
                _print_event(event, step, verbose)
            if step > max_steps:
                console.print(f"\n[yellow][!] Reached step limit ({max_steps}), stopping.[/]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow][!] Investigation interrupted by user.[/]")
    except Exception as e:
        console.print(f"\n[red][ERROR] {e}[/]")
        if verbose:
            traceback.print_exc()
    finally:
        if debug_console is not None:
            debug_console.stop()
        await system._close_checkpointer()
        if system.trajectory is not None:
            path = await system.trajectory.close()
            console.print()
            console.rule("Investigation complete")
            console.print(f"Steps: {step}")
            if path:
                console.print(f"Trajectory saved: [green]{path}[/]")
            console.print(f"Thread ID: [dim]{system.thread_id}[/]")

    # Keep dashboard alive for post-hoc inspection
    if dashboard_server_task is not None:
        console.print(
            f"\nDashboard running at [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link] — press Ctrl+C to stop."
        )
        try:
            await dashboard_server_task
        except asyncio.CancelledError:
            pass


def _print_event(event: dict, step: int, verbose: bool) -> None:
    """Print a streaming event in non-debug mode."""
    for node_name, node_data in event.items():
        if node_name == "__interrupt__":
            continue

        # Structured output from generate_structured_response node
        if node_name == "generate_structured_response":
            sr = node_data.get("structured_response")
            if sr is not None:
                console.print("\n[bold green][Structured Output][/]")
                if hasattr(sr, "model_dump"):
                    output_str = json.dumps(sr.model_dump(), indent=2, ensure_ascii=False)
                else:
                    output_str = json.dumps(sr, indent=2, ensure_ascii=False)
                console.print(output_str)
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


def _resolve_prompt_paths(scenario_config: Any, scenario_dir: Path) -> None:
    """Resolve relative prompt paths in scenario config against scenario_dir."""
    if scenario_config.orchestrator.prompts:
        scenario_config.orchestrator.prompts = {
            k: str(scenario_dir / v)
            for k, v in scenario_config.orchestrator.prompts.items()
        }

    if scenario_config.orchestrator.output is not None:
        scenario_config.orchestrator.output.prompt = str(
            scenario_dir / scenario_config.orchestrator.output.prompt
        )

    for agent_config in scenario_config.agents.values():
        if agent_config.prompt is not None:
            agent_config.prompt = str(scenario_dir / agent_config.prompt)
        if agent_config.task_type_prompts:
            agent_config.task_type_prompts = {
                k: str(scenario_dir / v)
                for k, v in agent_config.task_type_prompts.items()
            }
        if agent_config.compression and agent_config.compression.prompt:
            agent_config.compression.prompt = str(
                scenario_dir / agent_config.compression.prompt
            )
