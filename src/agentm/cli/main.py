"""AgentM CLI — typer application with run, debug, resume, and extract commands."""

from __future__ import annotations

import asyncio

import typer
from dotenv import load_dotenv

from agentm.cli.debug import analyze_trajectory
from agentm.cli.run import (
    resume_investigation,
    run_investigation,
    run_memory_extraction,
)

app = typer.Typer(
    name="agentm",
    help="AgentM — hypothesis-driven multi-agent orchestration framework.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command()
def run(
    data_dir: str = typer.Option(
        ..., "--data-dir", help="Observability data directory"
    ),
    incident: str = typer.Option(..., "--incident", help="Incident description"),
    scenario: str = typer.Option(
        "config/scenarios/rca_hypothesis",
        "--scenario",
        help="Scenario directory",
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
        "0.0.0.0",
        "--dashboard-host",
        help="Dashboard server bind address (default: 0.0.0.0)",
    ),
    max_steps: int = typer.Option(
        100,
        "--max-steps",
        help="Maximum orchestrator steps (default: 100)",
    ),
) -> None:
    """Run an RCA investigation."""
    asyncio.run(
        run_investigation(
            data_dir=data_dir,
            incident=incident,
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
    app()


@app.command()
def extract(
    trajectories: list[str] = typer.Argument(
        help=(
            "One or more source trajectories to process. "
            "Each can be a path to a .jsonl file (thread_id read from metadata) "
            "or a raw thread_id UUID."
        ),
        default=None,
    ),
    task: str = typer.Option(
        "",
        "--task",
        help=(
            "Custom extraction task description. "
            "If omitted, a default four-phase task is generated automatically."
        ),
    ),
    scenario: str = typer.Option(
        "config/scenarios/memory_extraction",
        "--scenario",
        help="Scenario directory (default: config/scenarios/memory_extraction)",
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
        "127.0.0.1", "--dashboard-host", help="Dashboard server bind address"
    ),
    max_steps: int = typer.Option(
        60, "--max-steps", help="Maximum orchestrator steps (default: 60)"
    ),
) -> None:
    """Extract reusable knowledge from completed RCA trajectories.

    Each TRAJECTORY argument is either a path to a .jsonl trajectory file
    or a raw thread_id UUID. Multiple trajectories can be passed at once.

    Examples:

      # Single trajectory file
      agentm extract trajectories/rca-20260311-162834.jsonl

      # Multiple files
      agentm extract trajectories/rca-*.jsonl

      # Raw thread_id with custom task description
      agentm extract 41fcf339-4d23-4b89-b7b3-b59602becd40 \\
          --task "Focus on database failure patterns only"

      # With live dashboard
      agentm extract trajectories/rca-20260311-162834.jsonl --dashboard
    """
    if not trajectories:
        typer.echo(
            "ERROR: At least one trajectory file or thread_id is required.",
            err=True,
        )
        raise typer.Exit(code=1)

    asyncio.run(
        run_memory_extraction(
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


@app.command()
def resume(
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
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint ID to restore (skips interactive selection)",
    ),
    list_checkpoints: bool = typer.Option(
        False, "--list", help="List available checkpoints without executing"
    ),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Start web dashboard after resuming"
    ),
    port: int = typer.Option(
        8765, "--port", help="Dashboard server port (requires --dashboard)"
    ),
    dashboard_host: str = typer.Option(
        "127.0.0.1", "--dashboard-host", help="Dashboard server bind address"
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
            checkpoint_id=checkpoint,
            list_only=list_checkpoints,
            dashboard=dashboard,
            dashboard_port=port,
            dashboard_host=dashboard_host,
            verbose=verbose,
        )
    )
