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
from agentm.exceptions import AgentMError
from agentm.config.schema import ScenarioConfig, SystemConfig
from agentm.scenarios.trajectory_judger.data import TrajectoryLabel
from agentm.tools.trajectory_reader import get_reader

logger = logging.getLogger(__name__)
console = Console()

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
    case_id: str,
    file_path: Path,
    ground_truth: list[str],
    system_config: SystemConfig,
    scenario_config: ScenarioConfig,
) -> TrajectoryLabel | None:
    """Run trajectory_judger on a single case via AgentSystem.execute()."""
    # Register trajectory file for jq_query access
    if file_path.exists():
        get_reader().register(str(file_path))

    system = build_agent_system(
        "trajectory_judger",
        scenario_config,
        system_config,
    )

    task_content = json.dumps(
        {
            "trajectory_id": case_id,
            "case_id": case_id,
            "ground_truth": ground_truth,
            "note": f"Use jq_query tool with thread_id='{case_id}' to query trajectory data",
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        async with system:
            result = await system.execute(
                {"task_description": f"Analyze trajectory {case_id}", "messages": [{"role": "human", "content": task_content}]}
            )
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
    cases: list[tuple[str, Path, list[str]]],
    system_config: SystemConfig,
    scenario_config: ScenarioConfig,
    output_path: str | None = None,
    verbose: bool = False,
) -> list[TrajectoryLabel]:
    """Run trajectory_judger scenario on each case.

    Args:
        cases: List of (case_id, file_path, ground_truth) tuples.
        system_config: Loaded system configuration.
        scenario_config: Loaded scenario configuration.
        output_path: Optional JSON file to write results.
        verbose: Print detailed progress.

    Returns:
        List of TrajectoryLabel results.
    """
    if not cases:
        console.print("[yellow]No cases to judge.[/]")
        return []

    # Disable trajectory collection for judging (we're analyzing, not producing)
    system_config.debug.trajectory.enabled = False

    console.print(f"Judging {len(cases)} cases with model: {scenario_config.orchestrator.model}")
    console.print()

    results: list[TrajectoryLabel] = []
    failed_cases: list[str] = []

    for i, (case_id, file_path, ground_truth) in enumerate(cases, 1):
        console.print(f"[{i}/{len(cases)}] Judging case {case_id}...")

        label = await _judge_single_case(
            case_id=case_id,
            file_path=file_path,
            ground_truth=ground_truth,
            system_config=system_config,
            scenario_config=scenario_config,
        )

        if label is not None:
            results.append(label)
            color = _CATEGORY_COLORS.get(label.category, "white")
            console.print(f"    [{color}]{label.category}[/] — {label.reasoning[:80]}...")
        else:
            failed_cases.append(case_id)
            console.print("    [red]FAILED[/]")

    _print_summary(results, total=len(cases), failed_count=len(failed_cases))

    if output_path and results:
        _save_results(results, total=len(cases), failed_count=len(failed_cases), output_path=output_path)

    return results
