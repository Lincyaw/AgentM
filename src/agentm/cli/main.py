"""AgentM CLI — typer application with run, debug, and extract commands."""

from __future__ import annotations

import asyncio
import logging
import os

import typer
from dotenv import load_dotenv

from agentm.exceptions import AgentMError

from agentm.cli.debug import analyze_trajectory
from agentm.cli.export_eval import export_eval_batch, export_eval_result
from agentm.cli.judge_runner import collect_cases, load_judge_config, run_judging
from agentm.cli.run import run_trajectory_analysis
from agentm.server.app import DashboardOpts

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
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
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
            "and whether the trajectory succeeded or failed."
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

      agentm analyze trajectories/rca-20260311-162834.jsonl \\
          --task "failure: missed ts-order-service"

      agentm analyze eval-trajectories/agentm-v11_7901_incorrect.json \\
          --task "failure: ground truth is ts-basic-service,ts-price-service"
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
            dashboard_opts=DashboardOpts(
                enabled=dashboard, port=port, host=dashboard_host,
            ),
            max_steps=max_steps,
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
    config_file: str = typer.Argument(
        ...,
        help="Path to batch config YAML (e.g. config/batch/default.yaml)",
    ),
    exp_id: str | None = typer.Option(None, "--exp-id", help="Override source.exp_id"),
    limit: int | None = typer.Option(None, "--limit", help="Override source.limit"),
    filter_correctness: str | None = typer.Option(
        None, "--filter", help="Override source.filter (incorrect|correct|all)",
    ),
    agent_type: str | None = typer.Option(None, "--agent-type", help="Override source.agent_type"),
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
    data_base_dir: str | None = typer.Option(
        None, "--data-base-dir", help="Override source.data_base_dir",
    ),
    source_path_pattern: str | None = typer.Option(
        None, "--source-path-pattern", help="Override source.source_path_pattern",
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", help="Number of cases to judge in parallel (overrides config file)",
    ),
) -> None:
    """Judge trajectories using decision-tree classification.

    Reads source settings from a batch config YAML. CLI options override
    config file values.

    Examples:

      agentm judge config/batch/default.yaml
      agentm judge config/batch/default.yaml --limit 5 --filter incorrect
      agentm judge config/batch/default.yaml -o judgment_results.json
      agentm judge config/batch/default.yaml --concurrency 4
    """
    from agentm.cli.run import load_and_override

    cfg = load_judge_config(config_file)

    # Apply CLI overrides to source config
    overrides = {
        "exp_id": exp_id, "limit": limit, "agent_type": agent_type,
        "filter": filter_correctness, "data_base_dir": data_base_dir,
        "source_path_pattern": source_path_pattern,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(cfg.source, key, val)

    # Auto-generate output path for caching when not specified
    effective_output = output
    if effective_output is None and cfg.source.exp_id:
        effective_output = f"judge_results_{cfg.source.exp_id}.json"

    try:
        case_infos = collect_cases(cfg)
    except Exception as e:
        typer.echo(f"ERROR: Failed to collect cases: {e}", err=True)
        raise typer.Exit(code=1)

    if not case_infos:
        typer.echo("No cases matched the filter criteria.", err=True)
        raise typer.Exit(code=1)

    system_config, scenario_config, _ = load_and_override(
        scenario, config, debug_mode=False, verbose=verbose,
    )

    asyncio.run(run_judging(
        case_infos,
        system_config=system_config,
        scenario_config=scenario_config,
        output_path=effective_output,
        dashboard_opts=DashboardOpts(
            enabled=dashboard, port=port, host=dashboard_host,
        ),
        concurrency=concurrency if concurrency is not None else cfg.concurrency,
    ))
