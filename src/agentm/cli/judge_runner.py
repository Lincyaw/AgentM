"""Trajectory judging runner — single-pass classification."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import uvicorn
from rich.console import Console
from rich.table import Table

from agentm.builder import build_agent_system
from agentm.scenarios.trajectory_judger.data import TrajectoryLabel
from agentm.server.app import Broadcaster, create_dashboard_app
from agentm.tools.trajectory_reader import get_reader

logger = logging.getLogger(__name__)
console = Console()


async def _judge_single_case(
    case_info: Any,
    ground_truth: list[str],
    system_config: Any,
    scenario_config: Any,
    verbose: bool = False,
) -> TrajectoryLabel | None:
    """Run trajectory_judger on a single case.

    Args:
        case_info: Case metadata (with file_path pointing to trajectory file)
        ground_truth: List of actual root-cause service names
        system_config: System configuration
        scenario_config: Scenario configuration
        verbose: Print detailed progress

    Returns:
        TrajectoryLabel or None if analysis failed
    """
    # Register trajectory file with TrajectoryReader for jq_query access
    # File was already exported by collect_from_db
    if case_info.file_path.exists():
        reader = get_reader()
        reader.register(str(case_info.file_path))

    # Build the agent system
    system = build_agent_system(
        "trajectory_judger",
        scenario_config,
        system_config,
    )

    # Construct the task input - include case_id for jq_query reference
    task_content = json.dumps(
        {
            "trajectory_id": case_info.case_id,
            "case_id": case_info.case_id,
            "ground_truth": ground_truth,
            "note": f"Use jq_query tool with thread_id='{case_info.case_id}' to query trajectory data",
        },
        ensure_ascii=False,
        indent=2,
    )

    initial_state = {
        "messages": [{"role": "human", "content": task_content}],
        "task_id": case_info.case_id,
        "task_description": f"Analyze trajectory {case_info.case_id}",
        "current_phase": "analyze",
        "ground_truth": ground_truth,
        "structured_output": None,
    }

    if verbose:
        console.print(f"  Judging case {case_info.case_id}...")

    try:
        # Stream and capture the result
        result_output = None
        async for event in system.stream(initial_state):
            agent_event = event.get("event")
            if agent_event is None:
                continue

            etype = getattr(agent_event, "type", "")
            data = getattr(agent_event, "data", {})

            if etype == "complete":
                result = data.get("result")
                if result is not None:
                    result_output = getattr(result, "output", None)

        # Parse the output into TrajectoryLabel
        if result_output is None:
            if verbose:
                console.print(f"    [red]No output for case {case_info.case_id}[/]")
            return None

        # Handle dict output from structured response
        if isinstance(result_output, dict):
            # Ensure required fields are present
            output_dict = dict(result_output)
            output_dict["trajectory_id"] = output_dict.get("trajectory_id", case_info.case_id)
            output_dict["case_id"] = output_dict.get("case_id", case_info.case_id)
            output_dict["agent_conclusion"] = output_dict.get("agent_conclusion", [])
            output_dict["ground_truth"] = output_dict.get("ground_truth", ground_truth)
            output_dict["is_correct"] = output_dict.get("is_correct", False)
            output_dict["is_partial"] = output_dict.get("is_partial", False)
            output_dict["category"] = output_dict.get("category", "judgment_fail")
            output_dict["reasoning"] = output_dict.get("reasoning", "No reasoning provided")

            return TrajectoryLabel(**output_dict)

        # Handle Pydantic model output
        if isinstance(result_output, TrajectoryLabel):
            return result_output

        if verbose:
            console.print(f"    [red]Unexpected output type: {type(result_output)}[/]")
        return None

    except Exception as e:
        if verbose:
            console.print(f"    [red]Error judging case {case_info.case_id}: {e}[/]")
        logger.error("Error judging case %s: %s", case_info.case_id, e)
        return None
    finally:
        if system.trajectory is not None:
            await system.trajectory.close()


async def run_judging(
    cases: list[tuple[Any, list[str]]],
    output_path: str | None = None,
    verbose: bool = False,
    dashboard: bool = False,
    dashboard_port: int = 8765,
    dashboard_host: str = "127.0.0.1",
) -> list[TrajectoryLabel]:
    """Run trajectory_judger scenario on each case.

    Args:
        cases: List of (case_info, ground_truth) tuples. case_info.file_path
               points to the exported trajectory JSON file.
        output_path: Optional JSON file to write results
        verbose: Print detailed progress

    Returns:
        List of TrajectoryLabel results
    """
    if not cases:
        console.print("[yellow]No cases to judge.[/]")
        return []

    # Import here to avoid circular dependencies
    from agentm.cli.run import _load_and_override

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    config_path = project_root / "config" / "system.yaml"
    scenario_path = project_root / "config" / "scenarios" / "trajectory_judger"

    console.print(f"Loading configuration for {len(cases)} cases...")

    # Use the same config loading pattern as run.py
    system_config, scenario_config, _ = _load_and_override(
        str(scenario_path),
        str(config_path),
        debug_mode=False,
        verbose=verbose,
    )

    # Disable trajectory collection for judging
    system_config.debug.trajectory.enabled = False

    console.print(f"Starting judgment with orchestrator model: {scenario_config.orchestrator.model}")
    console.print()

    results: list[TrajectoryLabel] = []
    failed_cases: list[str] = []

    for i, (case_info, ground_truth) in enumerate(cases, 1):
        console.print(f"[{i}/{len(cases)}] Judging case {case_info.case_id}...")

        result = await _judge_single_case(
            case_info=case_info,
            ground_truth=ground_truth,
            system_config=system_config,
            scenario_config=scenario_config,
            verbose=verbose,
        )

        if result is not None:
            results.append(result)
            label_color = {
                "success": "green",
                "lucky_hit": "yellow",
                "exploration_fail": "red",
                "confirmation_fail": "red",
                "judgment_fail": "red",
            }.get(result.category, "white")
            console.print(f"    [{label_color}]{result.category}[/] — {result.reasoning[:80]}...")
        else:
            failed_cases.append(case_info.case_id)
            console.print("    [red]FAILED[/]")

    console.print()
    console.rule("Judgment Complete")
    console.print(f"Total cases: [cyan]{len(cases)}[/]")
    console.print(f"Successful: [green]{len(results)}[/]")
    if failed_cases:
        console.print(f"Failed: [red]{len(failed_cases)}[/]")

    # Print summary table
    if results:
        table = Table(title="Classification Results")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_column("Percentage", style="yellow", justify="right")

        category_counts: dict[str, int] = {}
        for r in results:
            category_counts[r.category] = category_counts.get(r.category, 0) + 1

        for category in ["success", "lucky_hit", "exploration_fail", "confirmation_fail", "judgment_fail"]:
            count = category_counts.get(category, 0)
            if count > 0:
                pct = count / len(results) * 100
                table.add_row(category, str(count), f"{pct:.1f}%")

        console.print()
        console.print(table)

    # Write output file if specified
    if output_path and results:
        output_data = {
            "total": len(cases),
            "successful": len(results),
            "failed": len(failed_cases),
            "results": [r.model_dump(mode="json") for r in results],
        }
        Path(output_path).write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print()
        console.print(f"Results saved to: [green]{output_path}[/]")

    # Start dashboard after judging completes if requested
    if dashboard:
        broadcaster = Broadcaster()

        # Create dashboard app
        app = create_dashboard_app(
            scenario_config=scenario_config,
            runtime=None,
            trajectory=None,
            thread_id=f"judge-{cases[0][0].case_id if cases else 'batch'}",
            broadcaster=broadcaster,
            eval_tracker=None,
        )
        uvi_config = uvicorn.Config(
            app, host=dashboard_host, port=dashboard_port, log_level="warning"
        )
        server = uvicorn.Server(uvi_config)

        # Send all results to dashboard
        for i, (case_info, _) in enumerate(cases):
            result = results[i] if i < len(results) else None
            status = "completed" if result else "failed"
            await broadcaster.broadcast({
                "channel": "eval",
                "event_type": "sample_status",
                "data": {
                    "sample_id": case_info.case_id,
                    "dataset_index": case_info.dataset_index or 0,
                    "data_dir": case_info.data_dir or "",
                    "status": status,
                    "trajectory_path": str(case_info.file_path) if case_info.file_path else None,
                },
                "summary": {
                    "total": len(cases),
                    "completed": len(results),
                    "failed": len(failed_cases),
                    "pending": 0,
                    "running": 0,
                    "skipped": 0,
                },
            })

        console.print()
        console.print(
            f"Dashboard: [link=http://localhost:{dashboard_port}]"
            f"http://localhost:{dashboard_port}[/link]"
        )
        console.print("Press Ctrl+C to stop.")
        await server.serve()

    return results
