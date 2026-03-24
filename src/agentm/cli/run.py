"""CLI run subcommand — execute agent scenarios."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Callable

import uvicorn
from langchain_core.messages import HumanMessage
from rich.console import Console

import agentm.tools.observability as obs_tools
from agentm.exceptions import CheckpointError, DataInitError
from agentm.tools.duckdb_sql import register_tables as duckdb_register_tables
from agentm.builder import AgentSystemBuilder
from agentm.config.loader import load_scenario_config, load_system_config
from agentm.core.debug_console import DebugConsole
from agentm.core.trajectory import TrajectoryCollector
from agentm.server.app import Broadcaster, create_dashboard_app


console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------


def _load_and_override(
    scenario_dir: str,
    config_path: str,
    debug_mode: bool,
    verbose: bool,
) -> tuple[Any, Any, Path]:
    """Load configs, apply env overrides and CLI flags.

    Returns (system_config, scenario_config, scenario_path).
    """
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

    _resolve_prompt_paths(scenario_config, scenario_path)
    return system_config, scenario_config, scenario_path


async def _setup_debug_and_dashboard(
    system: Any,
    system_config: Any,
    verbose: bool,
    dashboard: bool,
    dashboard_host: str,
    dashboard_port: int,
    *,
    data_dir: str = "",
) -> tuple[Any | None, Any | None, Any | None, str | None]:
    """Wire up DebugConsole and optional dashboard with EvalTracker.

    Returns (debug_console, dashboard_server_task, eval_tracker, sample_id).
    """
    from rcabench_platform.v3.sdk.llm_eval.eval.tracker import EvalTracker

    debug_console = None
    if system_config.debug.console_live:
        debug_console = DebugConsole(verbose=verbose)
        debug_console.start()

    dashboard_server_task = None
    eval_tracker = None
    sample_id = None
    bc: Broadcaster | None = None
    if dashboard:
        bc = Broadcaster()
        tracker = EvalTracker()
        sample_id = system.thread_id

        tracker.register_sample(sample_id, dataset_index=0, data_dir=data_dir)
        tracker.mark_running(sample_id, run_id=sample_id)
        if system.trajectory is not None:
            tracker.update_trajectory_path(sample_id, str(system.trajectory.file_path))

        # Wire tracker → WebSocket
        _loop = asyncio.get_running_loop()

        def _tracker_to_ws(event: dict) -> None:
            try:
                asyncio.run_coroutine_threadsafe(bc.broadcast(event), _loop)
            except RuntimeError:
                pass

        tracker.add_listener(_tracker_to_ws)
        eval_tracker = tracker

        app = create_dashboard_app(
            graph=system.graph,
            scenario_config=system.scenario_config,
            task_manager=system.task_manager,
            trajectory=system.trajectory,
            thread_id=system.thread_id,
            broadcaster=bc,
            eval_tracker=tracker,
        )
        if system.task_manager is not None:
            system.task_manager.set_broadcast_callback(bc.broadcast)

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

    # Register trajectory listeners
    if system.trajectory is not None:
        if debug_console is not None:
            system.trajectory.add_listener(debug_console.on_trajectory_event)
        if dashboard and bc is not None:

            async def _traj_to_ws(event: dict) -> None:
                await bc.broadcast(
                    {
                        "event_type": event.get("event_type", ""),
                        "agent_path": event.get("agent_path", []),
                        "data": event.get("data", {}),
                        "timestamp": event.get("timestamp", ""),
                        "mode": "trajectory",
                    }
                )

            system.trajectory.add_listener(_traj_to_ws)

    return debug_console, dashboard_server_task, eval_tracker, sample_id


async def _stream_and_finalize(
    system: Any,
    initial_state: dict[str, Any],
    system_config: Any,
    debug_console: Any | None,
    dashboard_server_task: Any | None,
    dashboard_port: int,
    verbose: bool,
    max_steps: int,
    label: str,
    eval_tracker: Any | None = None,
    sample_id: str | None = None,
) -> None:
    """Stream execution, handle shutdown, and optionally keep dashboard alive."""
    console.rule(f"Starting {label}")
    step = 0
    stream_error: Exception | None = None
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
        console.print(f"\n[yellow][!] {label} interrupted by user.[/]")
    except Exception as e:
        stream_error = e
        console.print(f"\n[ERROR] {e}", style="red", markup=False)
        if verbose:
            traceback.print_exc()
    finally:
        if debug_console is not None:
            debug_console.stop()
        await system._close_checkpointer()
        if system.trajectory is not None:
            path = await system.trajectory.close()
            console.print()
            console.rule(f"{label} complete")
            console.print(f"Steps: {step}")
            if path:
                console.print(f"Trajectory saved: [green]{path}[/]")
            console.print(f"Thread ID: [dim]{system.thread_id}[/]")
        # Update eval tracker status
        if eval_tracker is not None and sample_id is not None:
            if stream_error is not None:
                eval_tracker.mark_failed(sample_id, str(stream_error))
            else:
                eval_tracker.mark_completed(sample_id)

    if dashboard_server_task is not None:
        console.print(
            f"\nDashboard running at [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link] — press Ctrl+C to stop."
        )
        try:
            await dashboard_server_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------


async def run_trajectory_analysis(
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
    """Run a trajectory analysis pass over one or more completed RCA trajectories."""
    system_config, scenario_config, _ = _load_and_override(
        scenario_dir, config_path, debug_mode, verbose
    )

    console.rule("AgentM — Trajectory Analysis")
    console.print(f"Source trajectories: [cyan]{len(trajectories)}[/]")
    for t in trajectories:
        console.print(f"  • [dim]{t}[/]")
    console.print(f"Orchestrator: [cyan]{scenario_config.orchestrator.model}[/]")
    console.print(f"Worker: [cyan]{scenario_config.agents['worker'].model}[/]")

    # Resolve thread_ids BEFORE build so checkpointer points at the right DB
    thread_ids: list[str] = []
    for entry in trajectories:
        p = Path(entry)
        if p.exists() and p.suffix == ".jsonl":
            meta = TrajectoryCollector.read_metadata(p)
            tid = meta.get("thread_id", "")
            checkpoint_db = meta.get("checkpoint_db", "")
            if not tid:
                raise CheckpointError(f"{entry} has no thread_id metadata.")
            if checkpoint_db:
                system_config.storage.checkpointer.backend = "sqlite"
                system_config.storage.checkpointer.url = checkpoint_db
            thread_ids.append(tid)
            console.print(
                f"  Resolved [dim]{p.name}[/] → thread_id [cyan]{tid[:16]}…[/]"
            )
        else:
            thread_ids.append(entry)

    system = AgentSystemBuilder.build(
        system_type="trajectory_analysis",
        scenario_config=scenario_config,
        system_config=system_config,
    )

    (
        debug_console,
        dashboard_task,
        eval_tracker,
        sample_id,
    ) = await _setup_debug_and_dashboard(
        system,
        system_config,
        verbose,
        dashboard,
        dashboard_host,
        dashboard_port,
    )

    if not task:
        traj_list = "\n".join(f"- {tid}" for tid in thread_ids)
        task = (
            f"Extract reusable knowledge from the following completed RCA "
            f"trajectories:\n\n{traj_list}"
        )

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task_id": system.thread_id,
        "task_description": task,
        "current_phase": "exploration",
        "source_trajectories": thread_ids,
        "extracted_patterns": [],
        "knowledge_entries": [],
        "existing_knowledge": [],
    }

    console.print(f"Task: {task[:200]}{'...' if len(task) > 200 else ''}")
    console.print()

    await _stream_and_finalize(
        system,
        initial_state,
        system_config,
        debug_console,
        dashboard_task,
        dashboard_port,
        verbose,
        max_steps,
        "trajectory_analysis",
        eval_tracker=eval_tracker,
        sample_id=sample_id,
    )


# ---------------------------------------------------------------------------
# Resume investigation
# ---------------------------------------------------------------------------


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
    project_root: str | Path | None = None,
) -> None:
    """Resume an interrupted investigation from a trajectory file."""
    traj_path = Path(trajectory_file)
    if not traj_path.exists():
        raise CheckpointError(f"Trajectory file not found: {trajectory_file}")

    meta = TrajectoryCollector.read_metadata(traj_path)
    thread_id = meta.get("thread_id", "")
    checkpoint_db = meta.get("checkpoint_db", "")

    if not thread_id:
        raise CheckpointError(
            "Trajectory file has no thread_id metadata. "
            "This trajectory was created before resume support was added."
        )

    console.rule("AgentM — Resume Investigation")
    console.print(f"Trajectory: [cyan]{trajectory_file}[/]")
    console.print(f"Thread ID:  [cyan]{thread_id}[/]")
    console.print(f"Checkpoint DB: [cyan]{checkpoint_db or '(none)'}[/]")

    system_config, scenario_config, _ = _load_and_override(
        scenario_dir, config_path, False, verbose
    )

    if checkpoint_db:
        system_config.storage.checkpointer.backend = "sqlite"
        system_config.storage.checkpointer.url = checkpoint_db

    # Initialize data
    if data_dir:
        result = obs_tools.set_data_directory(data_dir)
        init_info = json.loads(result)
        if "error" in init_info:
            raise DataInitError(init_info["error"])
        console.print(f"Data initialized: {len(init_info['files'])} parquet files")
        duckdb_register_tables(
            {
                Path(f).stem: str(Path(data_dir) / f)
                for f in init_info["files"]
                if Path(f).parent == Path(".")
                and Path(f).name in obs_tools.ALLOWED_TABLE_FILES
            }
        )

    system = AgentSystemBuilder.build(
        system_type="hypothesis_driven",
        scenario_config=scenario_config,
        system_config=system_config,
        existing_thread_id=thread_id,
    )

    await system._ensure_checkpointer()

    langgraph_config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshots = [s async for s in system.graph.aget_state_history(langgraph_config)]
    except Exception as e:
        raise CheckpointError(f"Error reading checkpoint history: {e}") from e

    if not snapshots:
        raise CheckpointError(
            "No checkpoints found for this thread. "
            "The checkpoint DB may have been moved or the thread_id is incorrect."
        )

    if list_only:
        _print_checkpoints(snapshots)
        await system._close_checkpointer()
        return

    if checkpoint_id is None:
        _print_checkpoints(snapshots)
        console.print()
        raw = console.input(
            "[bold]Enter checkpoint number (0 = latest, Enter = latest): [/]"
        ).strip()
        idx = int(raw) if raw else 0
        if idx < 0 or idx >= len(snapshots):
            await system._close_checkpointer()
            raise CheckpointError(
                f"Invalid index {idx}. Valid range: 0\u2013{len(snapshots) - 1}"
            )
        checkpoint_id = snapshots[idx].config["configurable"].get("checkpoint_id", "")

    resume_config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }
    console.print(f"\nResuming from checkpoint: [cyan]{checkpoint_id[:16]}…[/]")

    # Dashboard
    dashboard_server_task = None
    eval_tracker = None
    sample_id = None
    if dashboard:
        from rcabench_platform.v3.sdk.llm_eval.eval.tracker import EvalTracker

        bc = Broadcaster()
        tracker = EvalTracker()
        sample_id = thread_id

        tracker.register_sample(sample_id, dataset_index=0, data_dir=data_dir)
        tracker.mark_running(sample_id, run_id=sample_id)
        if system.trajectory is not None:
            tracker.update_trajectory_path(sample_id, str(system.trajectory.file_path))

        _loop = asyncio.get_running_loop()

        def _tracker_to_ws(event: dict) -> None:
            try:
                asyncio.run_coroutine_threadsafe(bc.broadcast(event), _loop)
            except RuntimeError:
                pass

        tracker.add_listener(_tracker_to_ws)
        eval_tracker = tracker

        app = create_dashboard_app(
            graph=system.graph,
            scenario_config=system.scenario_config,
            task_manager=system.task_manager,
            trajectory=system.trajectory,
            thread_id=thread_id,
            broadcaster=bc,
            eval_tracker=tracker,
        )
        if system.task_manager is not None:
            system.task_manager.set_broadcast_callback(bc.broadcast)

        if system.trajectory is not None:

            async def _traj_to_ws(event: dict) -> None:
                await bc.broadcast(
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
            app, host=dashboard_host, port=dashboard_port, log_level="warning"
        )
        server = uvicorn.Server(uvi_config)
        dashboard_server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)
        console.print(
            f"Dashboard: [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link]"
        )

    console.rule("Resuming")
    step = 0
    stream_error: Exception | None = None
    try:
        async for event in system.graph.astream(None, config=resume_config):
            step += 1
            _print_event(event, step, verbose)
            if system.trajectory is not None:
                for node_name, node_data in event.items():
                    if node_name == "__interrupt__" or not isinstance(node_data, dict):
                        continue
                    await system._record_node_event(node_name, node_data, step)
    except KeyboardInterrupt:
        console.print("\n[yellow][!] Interrupted by user.[/]")
    except Exception as e:
        stream_error = e
        console.print(f"\n[red][ERROR] {e}[/red]")
        if verbose:
            traceback.print_exc()
    finally:
        await system._close_checkpointer()
        console.rule("Resume complete")
        console.print(f"Steps executed: {step}")
        if eval_tracker is not None and sample_id is not None:
            if stream_error is not None:
                eval_tracker.mark_failed(sample_id, str(stream_error))
            else:
                eval_tracker.mark_completed(sample_id)

    if dashboard_server_task is not None:
        console.print(
            f"\nDashboard running at [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link] — press Ctrl+C to stop."
        )
        try:
            await dashboard_server_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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


def _print_event(event: dict, step: int, verbose: bool) -> None:
    """Print a streaming event in non-debug mode."""
    for node_name, node_data in event.items():
        if node_name == "__interrupt__":
            continue
        if not isinstance(node_data, dict):
            continue

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
                    console.print(content[:limit], markup=False)
                    if len(content) > limit:
                        console.print(f"  [dim]... ({len(content)} chars total)[/]")

                tool_calls = getattr(msg, "tool_calls", [])
                for tc in tool_calls:
                    args_str = json.dumps(tc["args"], ensure_ascii=False)
                    console.print(
                        f"\n  -> {tc['name']}({args_str[:300]})", markup=False
                    )

            elif role == "tool":
                tool_name = getattr(msg, "name", "?")
                if content:
                    preview = content
                    console.print(f"\n  <- {tool_name}: {preview}", markup=False)
                    if len(content) > len(preview):
                        console.print(f"     [dim]... ({len(content)} chars total)[/]")


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


# ---------------------------------------------------------------------------
# Headless runner (for batch eval)
# ---------------------------------------------------------------------------


def _langchain_msg_to_openai(msg: Any) -> dict[str, Any]:
    """Convert a LangChain message to OpenAI chat-completion message format.

    Role mapping: ai -> assistant, human -> user, system -> system, tool -> tool.
    AI messages with tool_calls include the ``tool_calls`` array in OpenAI format.
    Tool messages include ``tool_call_id``.
    """
    role = getattr(msg, "type", "unknown")
    content = getattr(msg, "content", "") or ""

    openai_role = {
        "ai": "assistant",
        "human": "user",
        "system": "system",
        "tool": "tool",
    }.get(role, role)

    entry: dict[str, Any] = {"role": openai_role, "content": content}

    if role == "ai":
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ]
    elif role == "tool":
        entry["name"] = getattr(msg, "name", "")
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id

    return entry


def _build_trajectory_json(run_id: str, messages: list[dict[str, Any]]) -> str:
    """Pack collected OpenAI-format messages into rcabench Span JSON."""
    span = {
        "trajectories": [
            {
                "trajectory_id": run_id,
                "agent_name": "agentm-orchestrator",
                "messages": messages,
            }
        ]
    }
    return json.dumps(span, ensure_ascii=False)


def _normalize_structured_response(data: dict[str, Any]) -> dict[str, Any]:
    """Convert agent-internal schema to eval-compatible format.

    Handles two transformations:

    1. **raw_text unwrap**: If the synthesize step fell back to plain LLM
       and produced ``{"raw_text": "..."}`` where the inner text is valid
       JSON with graph fields, unwrap it so the judge can parse it.

    2. **component_to_service**: The agent uses
       ``list[{component_name, service_name}]`` (required by Pydantic /
       ``with_structured_output``), but the eval judge expects
       ``dict[str, str]``.  Convert before serialization.
    """
    # 1. Unwrap raw_text fallback if inner content is a valid graph JSON
    if "raw_text" in data and len(data) == 1:
        raw = data["raw_text"]
        if isinstance(raw, str):
            try:
                inner = json.loads(raw)
                if isinstance(inner, dict) and (
                    "nodes" in inner or "root_causes" in inner
                ):
                    logger.info(
                        "raw_text unwrapped to graph JSON (keys=%s)",
                        list(inner.keys()),
                    )
                    data = inner
            except (json.JSONDecodeError, TypeError):
                pass

    # 2. Convert component_to_service list -> dict
    c2s = data.get("component_to_service")
    if isinstance(c2s, list):
        mapping: dict[str, str] = {}
        for item in c2s:
            if isinstance(item, dict):
                cname = item.get("component_name", "")
                sname = item.get("service_name", "")
                if cname:
                    mapping[cname] = sname
        logger.info(
            "component_to_service converted: list[%d] -> dict[%d]",
            len(c2s),
            len(mapping),
        )
        return {**data, "component_to_service": mapping}
    return data


async def run_investigation_headless(
    data_dir: str,
    incident: str,
    scenario_dir: str,
    config_path: str,
    max_steps: int = 100,
    timeout: float = 600.0,
    on_start: Callable[[str, str | None], None] | None = None,
    exp_id: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Run an RCA investigation without any console output or dashboard.

    Returns ``(structured_response_json, trajectory_json, run_id,
    trajectory_file_path)`` where any value may be ``None`` if not produced.
    Raises on configuration errors; runtime errors during streaming are caught
    and result in ``(None, None, run_id, trajectory_file_path)``.

    Args:
        on_start: Optional callback ``(run_id, trajectory_file_path)`` invoked
            right after the system is built but before streaming starts.  Use
            to update external trackers with the real trajectory path.

    Safe for concurrent use: both ``set_data_directory`` and
    ``register_tables`` store state in ``ContextVar``s, and each DuckDB query
    opens a fresh in-memory connection.
    """
    import uuid

    from dotenv import load_dotenv

    load_dotenv()

    system_config, scenario_config, _ = _load_and_override(
        scenario_dir, config_path, debug_mode=False, verbose=False
    )
    # Disable console live so no Rich output is produced
    system_config.debug.console_live = False

    # Route trajectory files into an exp_id sub-directory
    if exp_id:
        base_dir = system_config.debug.trajectory.output_dir
        system_config.debug.trajectory.output_dir = str(Path(base_dir) / exp_id)

    result = obs_tools.set_data_directory(data_dir)
    init_info = json.loads(result)
    if "error" in init_info:
        from agentm.exceptions import DataInitError

        raise DataInitError(init_info["error"])

    duckdb_register_tables(
        {
            Path(f).stem: str(Path(data_dir) / f)
            for f in init_info["files"]
            if Path(f).parent == Path(".")
            and Path(f).name in obs_tools.ALLOWED_TABLE_FILES
        }
    )

    system = AgentSystemBuilder.build(
        system_type="hypothesis_driven",
        scenario_config=scenario_config,
        system_config=system_config,
    )

    run_id = f"headless-{uuid.uuid4().hex[:12]}"
    initial_state: dict[str, Any] = {"messages": [HumanMessage(content=incident)]}

    # Resolve the real trajectory file path (set by builder)
    trajectory_file_path: str | None = None
    if system.trajectory is not None:
        trajectory_file_path = str(system.trajectory.file_path)

    if on_start is not None:
        on_start(run_id, trajectory_file_path)

    structured_response_json: str | None = None
    collected_messages: list[dict[str, Any]] = []

    try:

        async def _stream() -> None:
            nonlocal structured_response_json
            step = 0
            async for event in system.stream(initial_state):
                step += 1
                for node_name, node_data in event.items():
                    if node_name == "__interrupt__" or not isinstance(node_data, dict):
                        continue
                    # Capture structured response from synthesis nodes
                    if node_name in ("generate_structured_response", "synthesize"):
                        sr = node_data.get("structured_response")
                        if sr is not None:
                            if hasattr(sr, "model_dump"):
                                sr_dict = _normalize_structured_response(
                                    sr.model_dump()
                                )
                                structured_response_json = json.dumps(
                                    sr_dict, ensure_ascii=False
                                )
                            elif isinstance(sr, str):
                                structured_response_json = sr
                            elif isinstance(sr, dict):
                                sr_dict = _normalize_structured_response(sr)
                                structured_response_json = json.dumps(
                                    sr_dict, ensure_ascii=False
                                )
                            else:
                                structured_response_json = json.dumps(
                                    sr, ensure_ascii=False
                                )
                            # Log what we captured
                            is_fallback = isinstance(sr, dict) and "raw_text" in sr
                            logger.info(
                                "headless captured structured_response "
                                "(node=%s, is_fallback=%s, len=%d)",
                                node_name,
                                is_fallback,
                                len(structured_response_json)
                                if structured_response_json
                                else 0,
                            )
                    # Collect messages for trajectory
                    for msg in node_data.get("messages", []):
                        collected_messages.append(_langchain_msg_to_openai(msg))
                if step > max_steps and structured_response_json is not None:
                    break

        if timeout > 0:
            await asyncio.wait_for(_stream(), timeout=timeout)
        else:
            await _stream()

    except asyncio.TimeoutError:
        logger.warning("headless stream timed out after %.0fs", timeout)
    except Exception:
        logger.error("headless stream failed", exc_info=True)
    finally:
        await system._close_checkpointer()
        if system.trajectory is not None:
            await system.trajectory.close()

    trajectory_json: str | None = None
    if system.trajectory is not None and system.trajectory.events:
        from agentm.core.trajectory_converter import build_trajectory_from_events

        trajectory_json = build_trajectory_from_events(run_id, system.trajectory.events)
    elif collected_messages:
        trajectory_json = _build_trajectory_json(run_id, collected_messages)

    return structured_response_json, trajectory_json, run_id, trajectory_file_path
