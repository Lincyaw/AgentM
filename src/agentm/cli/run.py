"""CLI run subcommand — execute an RCA investigation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import uvicorn
from langchain_core.messages import HumanMessage
from rich.console import Console

import agentm.tools.observability as obs_tools
from agentm.builder import AgentSystemBuilder
from agentm.config.loader import load_scenario_config, load_system_config
from agentm.core.debug_console import DebugConsole
from agentm.core.trajectory import TrajectoryCollector
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
            app,
            host=dashboard_host,
            port=dashboard_port,
            log_level="warning",
        )
        server = uvicorn.Server(uvi_config)
        dashboard_server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)
        console.print(
            f"Dashboard: [link=http://localhost:{dashboard_port}]http://localhost:{dashboard_port}[/link]"
        )

    # Register trajectory listeners — all events flow through TrajectoryCollector
    if system.trajectory is not None:
        if debug_console is not None:
            system.trajectory.add_listener(debug_console.on_trajectory_event)
        if dashboard_broadcast is not None:

            async def _traj_to_ws(event: dict) -> None:
                await dashboard_broadcast(
                    {
                        "event_type": event.get("event_type", ""),
                        "agent_path": event.get("agent_path", []),
                        "data": event.get("data", {}),
                        "timestamp": event.get("timestamp", ""),
                        "mode": "trajectory",
                    }
                )

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
                console.print(
                    f"\n[yellow][!] Reached step limit ({max_steps}), stopping.[/]"
                )
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
        if not isinstance(node_data, dict):
            continue

        # Structured output from generate_structured_response node (react mode)
        # or synthesize node (node mode)
        if node_name in ("generate_structured_response", "synthesize"):
            sr = node_data.get("structured_response")
            if sr is not None:
                console.print("\n[bold green][Structured Output][/]")
                if hasattr(sr, "model_dump"):
                    output_str = json.dumps(
                        sr.model_dump(), indent=2, ensure_ascii=False
                    )
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


async def run_memory_extraction(
    trajectories: list[str],
    task: str,
    scenario_dir: str,
    config_path: str,
    debug_mode: bool = False,
    verbose: bool = False,
    dashboard: bool = False,
    dashboard_port: int = 8765,
    dashboard_host: str = "127.0.0.1",
    max_steps: int = 60,
) -> None:
    """Run a memory extraction pass over one or more completed RCA trajectories."""

    project_root = Path(config_path).resolve().parent.parent
    scenario_path = Path(scenario_dir)
    if not scenario_path.is_absolute():
        scenario_path = project_root / scenario_path

    system_config = load_system_config(config_path)
    scenario_config = load_scenario_config(scenario_path / "scenario.yaml")

    if model := os.environ.get("AGENTM_ORCHESTRATOR_MODEL"):
        scenario_config.orchestrator.model = model
    if model := os.environ.get("AGENTM_WORKER_MODEL"):
        scenario_config.agents["worker"].model = model

    if debug_mode:
        system_config.debug.console_live = True
    if verbose:
        system_config.debug.verbose = True

    console.rule("AgentM — Memory Extraction")
    console.print(f"Source trajectories: [cyan]{len(trajectories)}[/]")
    for t in trajectories:
        console.print(f"  • [dim]{t}[/]")
    console.print(f"Orchestrator: [cyan]{scenario_config.orchestrator.model}[/]")
    console.print(f"Worker: [cyan]{scenario_config.agents['worker'].model}[/]")

    _resolve_prompt_paths(scenario_config, scenario_path)

    # Resolve thread_ids BEFORE build so checkpointer points at the right DB
    thread_ids: list[str] = []
    for entry in trajectories:
        p = Path(entry)
        if p.exists() and p.suffix == ".jsonl":
            meta = TrajectoryCollector.read_metadata(p)
            tid = meta.get("thread_id", "")
            checkpoint_db = meta.get("checkpoint_db", "")
            if not tid:
                console.print(f"[red]ERROR: {entry} has no thread_id metadata.[/]")
                sys.exit(1)
            # Point checkpointer at the recorded DB before build so
            # set_checkpointer() gets the right instance
            if checkpoint_db:
                system_config.storage.checkpointer.backend = "sqlite"
                system_config.storage.checkpointer.url = checkpoint_db
            thread_ids.append(tid)
            console.print(
                f"  Resolved [dim]{p.name}[/] → thread_id [cyan]{tid[:16]}…[/]"
            )
        else:
            thread_ids.append(entry)  # assume it's already a thread_id UUID

    system = AgentSystemBuilder.build(
        system_type="memory_extraction",
        scenario_config=scenario_config,
        system_config=system_config,
    )

    # Debug console
    debug_console = None
    if system_config.debug.console_live:
        debug_console = DebugConsole(verbose=verbose)
        debug_console.start()

    # Dashboard
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
            app, host=dashboard_host, port=dashboard_port, log_level="warning"
        )
        server = uvicorn.Server(uvi_config)
        dashboard_server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)
        console.print(
            f"Dashboard: [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link]"
        )

    if system.trajectory is not None:
        if debug_console is not None:
            system.trajectory.add_listener(debug_console.on_trajectory_event)
        if dashboard_broadcast is not None:

            async def _traj_to_ws(event: dict) -> None:
                await dashboard_broadcast(
                    {
                        "event_type": event.get("event_type", ""),
                        "agent_path": event.get("agent_path", []),
                        "data": event.get("data", {}),
                        "timestamp": event.get("timestamp", ""),
                        "mode": "trajectory",
                    }
                )

            system.trajectory.add_listener(_traj_to_ws)

    # Build instruction for the orchestrator
    if not task:
        traj_list = "\n".join(f"- {tid}" for tid in thread_ids)
        task = (
            f"Extract reusable knowledge from the following completed RCA "
            f"trajectories:\n\n{traj_list}\n\n"
            "Follow the four-phase workflow: collect → analyze → extract → refine."
        )

    from agentm.models.enums import Phase

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task_id": system.thread_id,
        "task_description": task,
        "current_phase": Phase.EXPLORATION,
        "source_trajectories": thread_ids,
        "extracted_patterns": [],
        "knowledge_entries": [],
        "existing_knowledge": [],
    }

    console.rule("Starting extraction")
    console.print(f"Task: {task[:200]}{'...' if len(task) > 200 else ''}")
    console.print()

    step = 0
    try:
        async for event in system.stream(initial_state):
            step += 1
            if not system_config.debug.console_live:
                _print_event(event, step, verbose)
            if step > max_steps:
                console.print(
                    f"\n[yellow][!] Reached step limit ({max_steps}), stopping.[/]"
                )
                break

    except KeyboardInterrupt:
        console.print("\n[yellow][!] Extraction interrupted by user.[/]")
    except Exception as e:
        console.print(f"\n[red][ERROR] {e}[/]")
        if verbose:
            traceback.print_exc()
    finally:
        if debug_console is not None:
            debug_console.stop()
        await system._close_checkpointer()

        # Dump knowledge store to JSON so it survives process exit
        _dump_knowledge_store(system)

        if system.trajectory is not None:
            path = await system.trajectory.close()
            console.print()
            console.rule("Extraction complete")
            console.print(f"Steps: {step}")
            if path:
                console.print(f"Trajectory saved: [green]{path}[/]")
            console.print(f"Thread ID: [dim]{system.thread_id}[/]")

    if dashboard_server_task is not None:
        console.print(
            f"\nDashboard running at [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link] — press Ctrl+C to stop."
        )
        try:
            await dashboard_server_task
        except asyncio.CancelledError:
            pass


def _dump_knowledge_store(system: Any) -> None:
    """Dump all knowledge store entries to knowledge/knowledge.json.

    Merges with any existing entries from previous runs so knowledge
    accumulates across multiple extract invocations.
    """
    try:
        from agentm.tools import knowledge as km

        store = km._store_var.get()
        if store is None:
            return

        items = store.search(("knowledge",), query=None, limit=500)
        if not items:
            return

        new_entries: dict[str, Any] = {}
        for item in items:
            path = "/" + "/".join(item.namespace[1:]) + "/" + item.key
            new_entries[path] = {"path": path, **item.value}

        out_path = Path("./knowledge/knowledge.json")
        out_path.parent.mkdir(exist_ok=True)

        # Merge with existing entries
        existing: dict[str, Any] = {}
        if out_path.exists():
            try:
                for entry in json.loads(out_path.read_text(encoding="utf-8")):
                    existing[entry["path"]] = entry
            except Exception:
                pass

        merged = {**existing, **new_entries}
        out_path.write_text(
            json.dumps(list(merged.values()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(
            f"Knowledge saved: [green]{out_path}[/] "
            f"({len(new_entries)} new, {len(merged)} total)"
        )

    except Exception as e:
        console.print(f"[yellow]Could not dump knowledge store: {e}[/]")


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


def _print_checkpoints(snapshots: list) -> None:
    """Print checkpoint list in a readable table."""
    console.print("\n[bold]Available checkpoints:[/]")
    console.print(
        f"{'#':>3}  {'checkpoint_id':<40}  {'step':>5}  {'time':<19}  next_node"
    )
    console.print("-" * 90)
    for i, snap in enumerate(snapshots):
        cid = snap.config["configurable"].get("checkpoint_id", "")
        step = snap.metadata.get("step", "?")
        ts = snap.created_at[:19] if getattr(snap, "created_at", None) else "?"
        next_node = snap.next[0] if getattr(snap, "next", None) else "END"
        marker = " ← latest" if i == 0 else ""
        console.print(f"{i:>3}  {cid:<40}  {step:>5}  {ts}  {next_node}{marker}")


async def resume_investigation(
    trajectory_file: str,
    data_dir: str,
    scenario_dir: str,
    config_path: str,
    checkpoint_id: str | None = None,
    list_only: bool = False,
    dashboard: bool = False,
    dashboard_port: int = 8765,
    dashboard_host: str = "127.0.0.1",
    verbose: bool = False,
) -> None:
    """Resume an interrupted investigation from a trajectory file."""

    # 1. Read metadata from trajectory file
    traj_path = Path(trajectory_file)
    if not traj_path.exists():
        console.print(f"[red]ERROR: Trajectory file not found: {trajectory_file}[/]")
        sys.exit(1)

    meta = TrajectoryCollector.read_metadata(traj_path)
    thread_id = meta.get("thread_id", "")
    checkpoint_db = meta.get("checkpoint_db", "")

    if not thread_id:
        console.print("[red]ERROR: Trajectory file has no thread_id metadata.[/]")
        console.print(
            "[dim]This trajectory was created before resume support was added.[/]"
        )
        sys.exit(1)

    console.rule("AgentM — Resume Investigation")
    console.print(f"Trajectory: [cyan]{trajectory_file}[/]")
    console.print(f"Thread ID:  [cyan]{thread_id}[/]")
    console.print(f"Checkpoint DB: [cyan]{checkpoint_db or '(none)'}[/]")

    # 2. Load configs
    project_root = Path(config_path).resolve().parent.parent
    scenario_path = Path(scenario_dir)
    if not scenario_path.is_absolute():
        scenario_path = project_root / scenario_path

    system_config = load_system_config(config_path)
    scenario_config = load_scenario_config(scenario_path / "scenario.yaml")

    # Override storage to point at the saved checkpoint db
    if checkpoint_db:
        system_config.storage.checkpointer.backend = "sqlite"
        system_config.storage.checkpointer.url = checkpoint_db

    # Override from env
    if model := os.environ.get("AGENTM_ORCHESTRATOR_MODEL"):
        scenario_config.orchestrator.model = model
    if model := os.environ.get("AGENTM_WORKER_MODEL"):
        scenario_config.agents["worker"].model = model

    # 3. Initialize data
    if data_dir:
        result = obs_tools.set_data_directory(data_dir)
        init_info = json.loads(result)
        if "error" in init_info:
            console.print(f"[red]ERROR: {init_info['error']}[/]")
            sys.exit(1)
        console.print(f"Data initialized: {len(init_info['files'])} parquet files")

    # Resolve prompt paths
    _resolve_prompt_paths(scenario_config, scenario_path)

    # 4. Build AgentSystem (reuse existing thread_id → picks up saved checkpoints)
    system = AgentSystemBuilder.build(
        system_type="hypothesis_driven",
        scenario_config=scenario_config,
        system_config=system_config,
        existing_thread_id=thread_id,
    )

    # Ensure async checkpointer is initialized before listing snapshots
    await system._ensure_checkpointer()

    # 5. List checkpoints (use async API for AsyncSqliteSaver compatibility)
    langgraph_config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshots = [s async for s in system.graph.aget_state_history(langgraph_config)]
    except Exception as e:
        console.print(f"[red]ERROR reading checkpoint history: {e}[/]")
        sys.exit(1)

    if not snapshots:
        console.print("[yellow]No checkpoints found for this thread.[/]")
        console.print(
            "[dim]The checkpoint DB may have been moved or the thread_id is incorrect.[/]"
        )
        sys.exit(1)

    if list_only:
        _print_checkpoints(snapshots)
        await system._close_checkpointer()
        return

    # 6. Select checkpoint
    if checkpoint_id is None:
        _print_checkpoints(snapshots)
        console.print()
        raw = console.input(
            "[bold]Enter checkpoint number (0 = latest, Enter = latest): [/]"
        ).strip()
        idx = int(raw) if raw else 0
        if idx < 0 or idx >= len(snapshots):
            console.print(
                f"[red]Invalid index {idx}. Valid range: 0–{len(snapshots) - 1}[/]"
            )
            await system._close_checkpointer()
            sys.exit(1)
        checkpoint_id = snapshots[idx].config["configurable"].get("checkpoint_id", "")

    resume_config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }
    console.print(f"\nResuming from checkpoint: [cyan]{checkpoint_id[:16]}…[/]")

    # 7. Setup dashboard if requested
    dashboard_server_task = None
    if dashboard:
        app = create_dashboard_app(
            graph=system.graph,
            scenario_config=system.scenario_config,
            task_manager=system.task_manager,
            trajectory=system.trajectory,
            thread_id=thread_id,
        )
        if system.task_manager is not None:
            system.task_manager.set_broadcast_callback(broadcast_event)

        # Wire trajectory → WebSocket (same as run_investigation)
        if system.trajectory is not None:

            async def _traj_to_ws(event: dict) -> None:
                await broadcast_event(
                    {
                        "event_type": event.get("event_type", ""),
                        "agent_path": event.get("agent_path", []),
                        "data": event.get("data", {}),
                        "timestamp": event.get("timestamp", ""),
                        "mode": "trajectory",
                    }
                )

            system.trajectory.add_listener(_traj_to_ws)

        uvi_config = uvicorn.Config(
            app,
            host=dashboard_host,
            port=dashboard_port,
            log_level="warning",
        )
        server = uvicorn.Server(uvi_config)
        dashboard_server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)
        console.print(
            f"Dashboard: [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link]"
        )

    # 8. Resume — astream with None input + resume_config
    console.rule("Resuming")
    step = 0
    try:
        async for event in system.graph.astream(None, config=resume_config):
            step += 1
            _print_event(event, step, verbose)
            # Record node events to trajectory so dashboard gets live updates
            if system.trajectory is not None:
                for node_name, node_data in event.items():
                    if node_name == "__interrupt__" or not isinstance(node_data, dict):
                        continue
                    await system._record_node_event(node_name, node_data, step)
    except KeyboardInterrupt:
        console.print("\n[yellow][!] Interrupted by user.[/]")
    except Exception as e:
        console.print(f"\n[red][ERROR] {e}[/]")
        if verbose:
            traceback.print_exc()
    finally:
        await system._close_checkpointer()
        console.rule("Resume complete")
        console.print(f"Steps executed: {step}")

    if dashboard_server_task is not None:
        console.print(
            f"\nDashboard running at [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link] — press Ctrl+C to stop."
        )
        try:
            await dashboard_server_task
        except asyncio.CancelledError:
            pass
