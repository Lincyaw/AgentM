"""Trajectory judging runner — single-pass classification.

Orchestrates batch execution of the trajectory_judger scenario.
Reuses AgentSystem.execute() and shared infrastructure from batch.py / run.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pydantic
from rich.console import Console
from rich.table import Table

from agentm.builder import build_agent_system
from agentm.cli.batch import CaseInfo, parse_ground_truth
from agentm.config.schema import ScenarioConfig, SkeletonConfig, SystemConfig
from agentm.exceptions import AgentMError
from agentm.scenarios.trajectory_judger.data import TrajectoryLabel
from agentm.tools.trajectory_reader import get_reader

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int, *, oneline: bool = False) -> str:
    if oneline:
        text = text.replace("\n", " ").replace("\r", "")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _should_include_tool(tool_name: str, config: SkeletonConfig) -> bool:
    if config.include_tools:
        return tool_name in config.include_tools
    if config.exclude_tools:
        return tool_name not in config.exclude_tools
    return True


def _extract_skeleton_from_json(
    data: dict, config: SkeletonConfig,
) -> list[dict]:
    """Extract skeleton from DB-exported JSON (OpenAI message format).

    Walks through trajectories[*].messages looking for assistant tool_calls
    and their paired tool responses.
    """
    # Support both formats:
    # - Multi-agent wrapper: {"trajectories": [{"messages": [...]}, ...]}
    # - Single trajectory:  {"messages": [...]}
    trajectories = data.get("trajectories")
    if isinstance(trajectories, list):
        all_messages = [msg for traj in trajectories for msg in traj.get("messages", [])]
    elif isinstance(data.get("messages"), list):
        all_messages = data["messages"]
    else:
        return []

    # Build tool_call_id → response content map
    response_map: dict[str, str] = {}
    for msg in all_messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                response_map[tc_id] = str(msg.get("content", ""))

    steps: list[dict] = []
    step_num = 0
    for msg in all_messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            if not _should_include_tool(tool_name, config):
                continue

            step_num += 1
            raw_args = func.get("arguments", "")
            if isinstance(raw_args, str):
                try:
                    args_obj = json.loads(raw_args)
                    args_str = json.dumps(args_obj, ensure_ascii=False)
                except json.JSONDecodeError:
                    args_str = raw_args
            else:
                args_str = json.dumps(raw_args, ensure_ascii=False)

            tc_id = tc.get("id", "")
            response_content = response_map.get(tc_id, "")

            steps.append({
                "step": step_num,
                "tool": tool_name,
                "args": _truncate(args_str, config.max_args_length),
                "response_preview": _truncate(response_content, config.response_preview_length, oneline=True),
                "response_chars": len(response_content),
            })
    return steps


def _extract_skeleton_from_jsonl(
    path: Path, config: SkeletonConfig,
) -> list[dict]:
    """Extract skeleton from JSONL trajectory (TrajectoryCollector events).

    Pairs tool_call events with subsequent tool_result events by sequence.
    """
    tool_calls: list[dict] = []
    tool_results: dict[int, dict] = {}  # seq → result event data

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            et = event.get("event_type", "")
            if et == "tool_call":
                tool_calls.append(event)
            elif et == "tool_result":
                tool_results[event.get("seq", -1)] = event.get("data", {})

    steps: list[dict] = []
    # tool_result seq is typically tool_call seq + 1
    for i, tc_event in enumerate(tool_calls):
        data = tc_event.get("data", {})
        tool_name = data.get("tool_name", "")
        if not _should_include_tool(tool_name, config):
            continue

        args = data.get("args", {})
        args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)

        # Find matching tool_result (seq = tc_seq + 1)
        tc_seq = tc_event.get("seq", -1)
        result_data = tool_results.get(tc_seq + 1, {})
        response_content = str(result_data.get("content", ""))

        steps.append({
            "step": i + 1,
            "tool": tool_name,
            "args": _truncate(args_str, config.max_args_length),
            "response_preview": _truncate(response_content, config.response_preview_length, oneline=True),
            "response_chars": len(response_content),
        })
    return steps


def extract_skeleton(
    file_path: Path, config: SkeletonConfig,
) -> list[dict]:
    """Extract tool-call skeleton from a trajectory file.

    Supports JSON (DB export with _eval_meta/trajectories) and JSONL
    (TrajectoryCollector event stream) formats.

    Returns a list of step dicts:
        {"step": 1, "tool": "duckdb_sql", "args": "...", "response_preview": "...", "response_chars": 2847}
    """
    if not file_path.exists():
        return []

    if file_path.suffix == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return _extract_skeleton_from_json(data, config)

    return _extract_skeleton_from_jsonl(file_path, config)


def _load_injection_context(data_dir: str) -> str:
    """Load injection.json from data_dir and format key fault details.

    Searches data_dir itself and its parent (handles both
    ``{base}/{source}/`` and ``{base}/{source}/converted/`` layouts).

    Returns a compact summary string, or empty string if unavailable.
    """
    if not data_dir:
        return ""
    injection_path = Path(data_dir) / "injection.json"
    if not injection_path.exists():
        return ""
    try:
        injection = json.loads(injection_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    parts: list[str] = []

    fault_type = injection.get("fault_type")
    if fault_type is not None:
        parts.append(f"- **Fault Type**: {fault_type}")

    # Parse display_config for injection point details
    display_config_raw = injection.get("display_config")
    if display_config_raw:
        try:
            dc = json.loads(display_config_raw) if isinstance(display_config_raw, str) else display_config_raw
            ip = dc.get("injection_point", {})
            if ip:
                target_parts = []
                if ip.get("app_name"):
                    target_parts.append(ip["app_name"])
                if ip.get("class_name"):
                    target_parts.append(ip["class_name"])
                if ip.get("method_name"):
                    target_parts.append(ip["method_name"])
                if target_parts:
                    parts.append(f"- **Injection Target**: {' / '.join(target_parts)}")
            duration = dc.get("duration")
            if duration:
                parts.append(f"- **Duration**: {duration} min")
            mem_type = dc.get("mem_type")
            if mem_type is not None:
                label = "Heap" if mem_type == 1 else "Stack" if mem_type == 2 else str(mem_type)
                parts.append(f"- **Memory Type**: {label}")
        except (json.JSONDecodeError, TypeError):
            pass

    gt = injection.get("ground_truth", {})
    if gt:
        if gt.get("service"):
            parts.append(f"- **GT Service**: {', '.join(gt['service'])}")
        if gt.get("function"):
            parts.append(f"- **GT Function**: {', '.join(gt['function'])}")
        if gt.get("metric"):
            parts.append(f"- **GT Metric**: {', '.join(gt['metric'])}")

    start = injection.get("start_time")
    end = injection.get("end_time")
    if start and end:
        parts.append(f"- **Time Window**: {start} ~ {end}")

    return "\n".join(parts)


def format_skeleton(steps: list[dict]) -> str:
    """Format skeleton steps as compact text for LLM consumption."""
    if not steps:
        return "(no tool calls found)"
    lines = []
    for s in steps:
        preview = f" → {s['response_preview']}" if s["response_preview"] else ""
        chars = f" ({s['response_chars']} chars)" if s["response_chars"] else ""
        lines.append(f"Step {s['step']}: {s['tool']}({s['args']}){preview}{chars}")
    return "\n".join(lines)

# Category → Rich color for consistent display
_CATEGORY_COLORS: dict[str, str] = {
    "success": "green",
    "lucky_hit": "yellow",
    "exploration_fail": "red",
    "confirmation_fail": "red",
    "judgment_fail": "red",
}

# Ordered list for summary table
_CATEGORIES = ["success", "lucky_hit", "exploration_fail", "confirmation_fail", "judgment_fail"]


def _wire_trajectory_to_broadcaster(
    trajectory: object, broadcaster: object, case_id: str,
) -> None:
    """Add an async listener that forwards trajectory events to the WebSocket broadcaster."""

    async def _traj_to_ws(event: dict) -> None:
        traj_event = {
            "event_type": event.get("event_type", ""),
            "agent_path": event.get("agent_path", []),
            "data": event.get("data", {}),
            "timestamp": event.get("timestamp", ""),
        }
        # Push through the unified eval channel so the frontend receives
        # trajectory events via the same WebSocket path as status events.
        await broadcaster.broadcast(  # type: ignore[attr-defined]
            {
                "channel": "eval",
                "event_type": "sample_trajectory_event",
                "sample_id": case_id,
                "data": traj_event,
            }
        )

    trajectory.add_listener(_traj_to_ws)  # type: ignore[attr-defined]


def _parse_label(output: object, case_id: str, ground_truth: list[str]) -> TrajectoryLabel | None:
    """Coerce AgentSystem output into a TrajectoryLabel.

    The harness returns a dict (from Pydantic model_dump) when output_schema
    is set. Handles the raw_text fallback gracefully.
    """
    if output is None:
        return None

    if isinstance(output, dict):
        # raw_text fallback means structured output failed entirely
        if "raw_text" in output and len(output) == 1:
            logger.warning("Case %s: structured output failed, got raw_text fallback", case_id)
            return None
        # Fill identity fields that the LLM may have omitted
        output.setdefault("trajectory_id", case_id)
        output.setdefault("case_id", case_id)
        output.setdefault("ground_truth", ground_truth)
        try:
            return TrajectoryLabel(**output)
        except pydantic.ValidationError as exc:
            logger.warning("Case %s: TrajectoryLabel validation failed: %s", case_id, exc)
            return None

    if isinstance(output, TrajectoryLabel):
        return output

    logger.warning("Case %s: unexpected output type %s", case_id, type(output).__name__)
    return None


async def _judge_single_case(
    case: CaseInfo,
    system_config: SystemConfig,
    scenario_config: ScenarioConfig,
    broadcaster: object | None = None,
    eval_tracker: object | None = None,
) -> TrajectoryLabel | None:
    """Run trajectory_judger on a single case via AgentSystem.execute()."""
    case_id = case.case_id
    file_path = case.file_path
    ground_truth = parse_ground_truth(case.correct_answer)

    # Register trajectory file for jq_query access
    if file_path.exists():
        get_reader().register(str(file_path))

    system = build_agent_system(
        "trajectory_judger",
        scenario_config,
        system_config,
    )

    # Update tracker with the real trajectory file path so the REST events
    # endpoint (/api/eval/samples/{id}/events) can read from the correct file.
    if eval_tracker is not None and system.trajectory is not None:
        eval_tracker.update_trajectory_path(case_id, str(system.trajectory.file_path))  # type: ignore[attr-defined]

    # Wire trajectory events → dashboard WebSocket
    if broadcaster is not None and system.trajectory is not None:
        _wire_trajectory_to_broadcaster(system.trajectory, broadcaster, case_id)

    # Build trajectory skeleton for context injection
    skeleton_config = scenario_config.orchestrator.skeleton or SkeletonConfig()
    skeleton_steps = extract_skeleton(file_path, skeleton_config)
    skeleton_text = format_skeleton(skeleton_steps)

    gt_str = ", ".join(ground_truth)
    parts = [
        f"## Case {case_id}",
        "",
        f"- **Trajectory ID**: {case_id}",
        f"- **Ground Truth**: {gt_str}",
        f"- **jq_query thread_id**: `{case_id}`",
    ]

    # Fault context: injection details from data_dir + eval reasoning
    logger.info(
        "Case %s: data_dir=%r, reasoning=%r",
        case_id, case.data_dir or "(empty)", (case.reasoning or "")[:80],
    )
    injection_ctx = _load_injection_context(case.data_dir)
    if injection_ctx:
        logger.info("Case %s: loaded injection context from %s", case_id, case.data_dir)
    elif case.data_dir:
        logger.warning("Case %s: no injection.json found at %s", case_id, case.data_dir)
    if injection_ctx or case.reasoning:
        parts.extend(["", "## Fault Context", ""])
        if injection_ctx:
            parts.append(injection_ctx)
        if case.reasoning:
            if injection_ctx:
                parts.append("")
            parts.append(f"**Eval Reasoning**: {case.reasoning}")

    if skeleton_steps:
        parts.extend([
            "",
            "## Trajectory Skeleton",
            "",
            skeleton_text,
        ])

    task_content = "\n".join(parts)

    try:
        async with system:
            result = await system.execute({"task_description": task_content})
        return _parse_label(result.get("output"), case_id, ground_truth)
    except (AgentMError, asyncio.TimeoutError) as e:
        logger.error("Error judging case %s: %s", case_id, e)
        return None


def _print_summary(results: list[TrajectoryLabel], total: int, failed_count: int) -> None:
    """Print classification summary table."""
    console.print()
    console.rule("Judgment Complete")
    console.print(f"Total cases: [cyan]{total}[/]")
    console.print(f"Successful: [green]{len(results)}[/]")
    if failed_count:
        console.print(f"Failed: [red]{failed_count}[/]")

    if not results:
        return

    table = Table(title="Classification Results")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="yellow", justify="right")

    category_counts: dict[str, int] = {}
    for r in results:
        category_counts[r.category] = category_counts.get(r.category, 0) + 1

    for category in _CATEGORIES:
        count = category_counts.get(category, 0)
        if count > 0:
            pct = count / len(results) * 100
            table.add_row(category, str(count), f"{pct:.1f}%")

    console.print()
    console.print(table)


def _save_results(results: list[TrajectoryLabel], total: int, failed_count: int, output_path: str) -> None:
    """Write results to a JSON file."""
    output_data = {
        "total": total,
        "successful": len(results),
        "failed": failed_count,
        "results": [r.model_dump(mode="json") for r in results],
    }
    Path(output_path).write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\nResults saved to: [green]{output_path}[/]")


async def run_judging(
    cases: list[CaseInfo],
    system_config: SystemConfig,
    scenario_config: ScenarioConfig,
    output_path: str | None = None,
    dashboard: bool = False,
    dashboard_port: int = 8765,
    dashboard_host: str = "0.0.0.0",
) -> list[TrajectoryLabel]:
    """Run trajectory_judger scenario on each case.

    Args:
        cases: List of CaseInfo objects.
        system_config: Loaded system configuration.
        scenario_config: Loaded scenario configuration.
        output_path: Optional JSON file to write results.
        dashboard: Start web dashboard for real-time monitoring.
        dashboard_port: Dashboard server port.
        dashboard_host: Dashboard server bind address.

    Returns:
        List of TrajectoryLabel results.
    """
    if not cases:
        console.print("[yellow]No cases to judge.[/]")
        return []

    # Disable trajectory collection for judging unless dashboard needs events
    if not dashboard:
        system_config.debug.trajectory.enabled = False

    console.print(f"Judging {len(cases)} cases with model: {scenario_config.orchestrator.model}")
    console.print()

    # Optional dashboard setup
    tracker, broadcaster, dashboard_task = None, None, None
    if dashboard:
        dashboard_tuples = [
            (c.case_id, c.file_path, parse_ground_truth(c.correct_answer))
            for c in cases
        ]
        tracker, broadcaster, dashboard_task = await _start_dashboard(
            dashboard_tuples, scenario_config, dashboard_host, dashboard_port,
        )

    results: list[TrajectoryLabel] = []
    failed_cases: list[str] = []

    for i, case in enumerate(cases, 1):
        console.print(f"[{i}/{len(cases)}] Judging case {case.case_id}...")

        if tracker is not None:
            tracker.mark_running(case.case_id, run_id=case.case_id)

        label = await _judge_single_case(
            case=case,
            system_config=system_config,
            scenario_config=scenario_config,
            broadcaster=broadcaster,
            eval_tracker=tracker,
        )

        if label is not None:
            results.append(label)
            color = _CATEGORY_COLORS.get(label.category, "white")
            console.print(f"    [{color}]{label.category}[/] — {label.reasoning[:80]}...")
            if tracker is not None:
                tracker.mark_completed(case.case_id)
        else:
            failed_cases.append(case.case_id)
            console.print("    [red]FAILED[/]")
            if tracker is not None:
                tracker.mark_failed(case.case_id, "classification failed")

    _print_summary(results, total=len(cases), failed_count=len(failed_cases))

    if output_path and results:
        _save_results(results, total=len(cases), failed_count=len(failed_cases), output_path=output_path)

    # Keep dashboard alive after judging
    if dashboard_task is not None:
        console.print(
            f"\nDashboard running at [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link] — press Ctrl+C to stop."
        )
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass

    return results


async def _start_dashboard(
    cases: list[tuple[str, Path, list[str]]],
    scenario_config: ScenarioConfig,
    host: str,
    port: int,
) -> tuple:
    """Start the dashboard server and register all cases with the tracker.

    Returns (tracker, broadcaster, server_task).
    """
    import uvicorn

    from rcabench_platform.v3.sdk.llm_eval.eval.tracker import EvalTracker

    from agentm.server.app import Broadcaster, create_dashboard_app

    tracker = EvalTracker()
    broadcaster = Broadcaster()

    # Register all cases as pending
    for i, (case_id, file_path, _gt) in enumerate(cases):
        tracker.register_sample(case_id, dataset_index=i, data_dir=str(file_path.parent))

    # Wire tracker events → WebSocket
    loop = asyncio.get_running_loop()

    def _tracker_to_ws(event: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(broadcaster.broadcast(event), loop)
        except RuntimeError:
            logger.debug("Dropping dashboard event: event loop closed")

    tracker.add_listener(_tracker_to_ws)

    app = create_dashboard_app(
        scenario_config=scenario_config,
        runtime=None,
        trajectory=None,
        thread_id=f"judge-{cases[0][0] if cases else 'batch'}",
        broadcaster=broadcaster,
        eval_tracker=tracker,
    )

    uvi_config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(uvi_config)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)
    console.print(
        f"Dashboard: [link=http://localhost:{port}]http://localhost:{port}[/link]"
    )

    return tracker, broadcaster, task
