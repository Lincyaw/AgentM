"""AgentM CLI — typer application with run, debug, resume, and extract commands."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from agentm.exceptions import AgentMError

from agentm.cli.batch import (
    collect_cases,
    load_batch_config,
    parse_ground_truth,
    run_batch_analysis,
)
from agentm.cli.debug import analyze_trajectory
from agentm.cli.export_eval import export_eval_batch, export_eval_result
from agentm.cli.judge_runner import run_judging
from agentm.cli.run import (
    resume_investigation,
    run_trajectory_analysis,
)

app = typer.Typer(
    name="agentm",
    help="AgentM — hypothesis-driven multi-agent orchestration framework.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command()
def debug(
    trajectory_file: str = typer.Argument(help="Path to .jsonl trajectory file"),
    summary: bool = typer.Option(False, "--summary", help="Print summary statistics"),
    timeline: bool = typer.Option(False, "--timeline", help="Show tool call timeline"),
    filter_agent: str | None = typer.Option(
        None, "--filter-agent", help="Filter by agent path prefix"
    ),
    filter_type: str | None = typer.Option(
        None, "--filter-type", help="Filter by event_type"
    ),
) -> None:
    """Analyze a trajectory JSONL file."""
    analyze_trajectory(
        trajectory_file=trajectory_file,
        show_summary=summary,
        show_timeline=timeline,
        filter_agent=filter_agent,
        filter_type=filter_type,
    )


def main() -> None:
    load_dotenv()
    log_level = os.environ.get("AGENTM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Apply custom log level only to agentm loggers
    if log_level != "INFO":
        logging.getLogger("agentm").setLevel(getattr(logging, log_level, logging.INFO))
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    try:
        app()
    except AgentMError as e:
        from rich.console import Console

        Console().print(f"[red]ERROR: {e}[/]")
        import sys

        sys.exit(1)


@app.command()
def analyze(
    trajectories: list[str] = typer.Argument(
        help=(
            "One or more source trajectories to analyze. "
            "Each can be a .jsonl file (event format), a .json file "
            "(message format from eval DB export), or a raw thread_id UUID."
        ),
        default=None,
    ),
    task: str = typer.Option(
        ...,
        "--task",
        help=(
            "Analysis task with evaluation feedback. Describe what to analyze "
            "and whether the trajectory succeeded or failed. "
            "e.g. 'success: correctly identified mysql as root cause, "
            "extract the reasoning patterns that led to this' or "
            "'failure: missed ts-order-service, anchored on ts-preserve-service'"
        ),
    ),
    scenario: str = typer.Option(
        "config/scenarios/trajectory_analysis",
        "--scenario",
        help="Scenario directory (default: config/scenarios/trajectory_analysis)",
    ),
    config: str = typer.Option(
        "config/system.yaml", "--config", help="System config YAML"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable rich debug terminal UI"),
    verbose: bool = typer.Option(False, "--verbose", help="Extra detail in output"),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Start web dashboard for real-time monitoring"
    ),
    port: int = typer.Option(
        8765, "--port", help="Dashboard server port (requires --dashboard)"
    ),
    dashboard_host: str = typer.Option(
        "0.0.0.0", "--dashboard-host", help="Dashboard server bind address"
    ),
    max_steps: int = typer.Option(
        60, "--max-steps", help="Maximum orchestrator steps (default: 60)"
    ),
) -> None:
    """Analyze completed RCA trajectories and extract reusable knowledge.

    Each TRAJECTORY argument can be a .jsonl file (event format), a .json file
    (message format from eval DB export), or a raw thread_id UUID.
    --task is required and should include evaluation feedback.

    Examples:

      # Analyze an event-format trajectory (JSONL)
      agentm analyze trajectories/rca-20260311-162834.jsonl \\
          --task "failure: missed ts-order-service"

      # Analyze an exported eval case (JSON)
      agentm analyze eval-trajectories/agentm-v11_7901_incorrect.json \\
          --task "failure: ground truth is ts-basic-service,ts-price-service"

      # Multiple files
      agentm analyze trajectories/rca-*.jsonl \\
          --task "2/3 succeeded, 1 failed on cascade identification"
    """
    if not trajectories:
        typer.echo(
            "ERROR: At least one trajectory file or thread_id is required.",
            err=True,
        )
        raise typer.Exit(code=1)

    asyncio.run(
        run_trajectory_analysis(
            trajectories=trajectories,
            task=task,
            scenario_dir=scenario,
            config_path=config,
            debug_mode=debug,
            verbose=verbose,
            dashboard=dashboard,
            dashboard_port=port,
            dashboard_host=dashboard_host,
            max_steps=max_steps,
        )
    )


@app.command("analyze-batch")
def analyze_batch(
    config_file: str = typer.Argument(
        ...,
        help="Path to batch config YAML (e.g. config/batch/default.yaml)",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Override source.limit in config"
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Override batch.size in config"
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", help="Override batch.concurrency in config"
    ),
    exp_id: str | None = typer.Option(
        None, "--exp-id", help="Override source.exp_id in config"
    ),
    filter_correctness: str | None = typer.Option(
        None,
        "--filter",
        help="Override source.filter (incorrect|correct|all)",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Extra detail in output"),
    dashboard: bool = typer.Option(False, "--dashboard", help="Start web dashboard"),
    port: int = typer.Option(8765, "--port", help="Dashboard server port"),
) -> None:
    """Batch analyze evaluation trajectories from config file.

    Load a batch config YAML that specifies data source (directory or DB),
    batch grouping strategy, and analysis goals. Each batch groups N
    trajectories into a single analysis run for cross-case pattern detection.

    Examples:

      # Error analysis on failed cases
      agentm analyze-batch config/batch/default.yaml

      # Include correct cases too
      agentm analyze-batch config/batch/default.yaml --filter all

      # Override experiment and limit
      agentm analyze-batch config/batch/default.yaml --exp-id agentm-v12 --limit 30

      # Quick ad-hoc override
      agentm analyze-batch config/batch/default.yaml --batch-size 5 --verbose
    """
    cfg = load_batch_config(config_file)

    # Apply CLI overrides
    if limit is not None:
        cfg.source.limit = limit
    if batch_size is not None:
        cfg.batch.size = batch_size
    if concurrency is not None:
        cfg.batch.concurrency = concurrency
    if exp_id is not None:
        cfg.source.exp_id = exp_id
    if filter_correctness is not None:
        cfg.source.filter = filter_correctness  # type: ignore[assignment]
    if verbose:
        cfg.output.verbose = True
    if dashboard:
        cfg.output.dashboard = True
        cfg.output.dashboard_port = port

    cases = collect_cases(cfg)
    if not cases:
        typer.echo("No cases matched the filter criteria.", err=True)
        raise typer.Exit(code=1)

    asyncio.run(run_batch_analysis(cfg, cases))


@app.command()
def resume(  # noqa: ARG001  — CLI params reserved for future checkpoint resume
    trajectory_file: str = typer.Argument(help="Path to trajectory .jsonl file"),
    data_dir: str = typer.Option("", "--data-dir", help="Observability data directory"),
    scenario: str = typer.Option(
        "config/scenarios/rca_hypothesis",
        "--scenario",
        help="Scenario directory",
    ),
    config: str = typer.Option(
        "config/system.yaml", "--config", help="System config YAML"
    ),
    checkpoint: str | None = typer.Option(  # noqa: ARG001
        None,
        "--checkpoint",
        help="Checkpoint ID to restore (skips interactive selection)",
    ),
    list_checkpoints: bool = typer.Option(  # noqa: ARG001
        False, "--list", help="List available checkpoints without executing"
    ),
    dashboard: bool = typer.Option(  # noqa: ARG001
        False, "--dashboard", help="Start web dashboard after resuming"
    ),
    port: int = typer.Option(  # noqa: ARG001
        8765, "--port", help="Dashboard server port (requires --dashboard)"
    ),
    dashboard_host: str = typer.Option(  # noqa: ARG001
        "0.0.0.0", "--dashboard-host", help="Dashboard server bind address"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Extra detail in output"),
) -> None:
    """Resume an interrupted investigation from a trajectory file.

    Without --checkpoint: shows an interactive list to pick a restore point.
    With --list: only lists available checkpoints, does not execute.
    With --checkpoint <id>: resumes directly from the given checkpoint ID.
    """
    asyncio.run(
        resume_investigation(
            trajectory_file=trajectory_file,
            data_dir=data_dir,
            scenario_dir=scenario,
            config_path=config,
            verbose=verbose,
        )
    )


@app.command("export-result")
def export_result(
    trajectory_file: str = typer.Argument(help="Path to trajectory .jsonl file"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output JSON path. "
            "Default: same directory as trajectory with suffix .export.json"
        ),
    ),
) -> None:
    """Export case_dir + ground_truth + final outputs for one trajectory."""
    from rich.console import Console

    console = Console()
    out_path = export_eval_result(trajectory_file=trajectory_file, output_file=output)
    console.print(f"Exported: [green]{out_path}[/]")


@app.command("export-batch")
def export_batch(
    trajectory_dir: str = typer.Argument(help="Directory containing trajectory .jsonl files"),
    pattern: str = typer.Option(
        "*.jsonl",
        "--pattern",
        "-p",
        help="Glob pattern under trajectory_dir (default: *.jsonl)",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help=(
            "Directory for exported JSON files. "
            "Default: write next to each trajectory file."
        ),
    ),
) -> None:
    """Batch export case_dir + ground_truth + final outputs from trajectories."""
    from rich.console import Console

    console = Console()
    success, failed = export_eval_batch(
        trajectory_dir=trajectory_dir,
        pattern=pattern,
        output_dir=output_dir,
    )

    console.print(f"Exported: [green]{len(success)}[/] files")
    if success:
        preview = success[:5]
        for item in preview:
            console.print(f"  - [green]{item}[/]")
        if len(success) > len(preview):
            console.print(f"  ... and {len(success) - len(preview)} more")

    if failed:
        console.print(f"Failed: [red]{len(failed)}[/] files")
        for traj, err in failed[:10]:
            console.print(f"  - [red]{traj}[/]: {err}")


@app.command()
def judge(
    exp_id: str | None = typer.Option(None, "--exp-id", help="Experiment ID filter"),
    limit: int | None = typer.Option(None, "--limit", help="Max cases to process"),
    filter_correctness: str = typer.Option("all", "--filter", help="Filter: incorrect/correct/all"),
    agent_type: str | None = typer.Option(None, "--agent-type", help="Agent type filter"),
    source: str = typer.Option("database", "--source", help="Source: database or directory"),
    directory: str | None = typer.Option(None, "--dir", help="Directory path (if source=directory)"),
    export_dir: str = typer.Option(
        "./eval-trajectories", "--export-dir",
        help="Directory for DB-exported JSON files",
    ),
    output: str | None = typer.Option(None, "-o", "--output", help="Output JSON file for results"),
    verbose: bool = typer.Option(False, "--verbose", help="Extra detail"),
    scenario: str = typer.Option(
        "config/scenarios/trajectory_judger",
        "--scenario",
        help="Scenario directory",
    ),
    config: str = typer.Option(
        "config/system.yaml", "--config", help="System config YAML",
    ),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Start web dashboard for real-time monitoring",
    ),
    port: int = typer.Option(8765, "--port", help="Dashboard server port"),
    dashboard_host: str = typer.Option(
        "0.0.0.0", "--dashboard-host", help="Dashboard server bind address",
    ),
) -> None:
    """Judge trajectories using decision-tree classification.

    Classifies each trajectory as: success, lucky_hit, exploration_fail,
    confirmation_fail, or judgment_fail with detailed reasoning.

    Examples:

      # Judge one case from database
      agentm judge --limit 1

      # Judge all incorrect cases from an experiment
      agentm judge --exp-id agentm-v12 --filter incorrect

      # Judge cases from a directory
      agentm judge --source directory --dir ./eval-trajectories --limit 10

      # Save results to JSON
      agentm judge --limit 50 -o judgment_results.json
    """
    if filter_correctness not in ("incorrect", "correct", "all"):
        typer.echo("ERROR: --filter must be one of: incorrect, correct, all", err=True)
        raise typer.Exit(code=1)

    if source not in ("database", "directory"):
        typer.echo("ERROR: --source must be one of: database, directory", err=True)
        raise typer.Exit(code=1)

    if source == "directory" and not directory:
        typer.echo("ERROR: --dir is required when --source=directory", err=True)
        raise typer.Exit(code=1)

    from agentm.cli.batch import collect_from_db, collect_from_directory
    from agentm.cli.run import _load_and_override

    # Collect cases — use return value directly (no double-collect)
    try:
        if source == "database":
            case_infos = collect_from_db(
                exp_id=exp_id,
                filter_correctness=filter_correctness,
                agent_type=agent_type,
                limit=limit,
                output_dir=export_dir,
            )
        else:
            case_infos = collect_from_directory(
                directory=directory,  # type: ignore[arg-type]  # validated above
                filter_correctness=filter_correctness,
                exp_id=exp_id,
                agent_type=agent_type,
                limit=limit,
            )
    except Exception as e:
        typer.echo(f"ERROR: Failed to collect cases: {e}", err=True)
        raise typer.Exit(code=1)

    if not case_infos:
        typer.echo("No cases matched the filter criteria.", err=True)
        raise typer.Exit(code=1)

    # Build (case_id, file_path, ground_truth) tuples for run_judging
    cases = [
        (ci.case_id, ci.file_path, parse_ground_truth(ci.correct_answer))
        for ci in case_infos
    ]

    # Load configs via the shared loader (same pattern as analyze/run)
    system_config, scenario_config, _ = _load_and_override(
        scenario, config, debug_mode=False, verbose=verbose,
    )

    asyncio.run(run_judging(
        cases,
        system_config=system_config,
        scenario_config=scenario_config,
        output_path=output,
        dashboard=dashboard,
        dashboard_port=port,
        dashboard_host=dashboard_host,
    ))
