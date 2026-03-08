"""AgentM CLI — typer application with run and debug commands."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="agentm",
    help="AgentM — hypothesis-driven multi-agent orchestration framework.",
    no_args_is_help=True,
)


@app.command()
def run(
    data_dir: str = typer.Option(..., "--data-dir", help="Observability data directory"),
    incident: str = typer.Option(..., "--incident", help="Incident description"),
    scenario: str = typer.Option(
        "config/scenarios/rca_hypothesis",
        "--scenario",
        help="Scenario directory",
    ),
    config: str = typer.Option("config/system.yaml", "--config", help="System config YAML"),
    debug: bool = typer.Option(False, "--debug", help="Enable rich debug terminal UI"),
    verbose: bool = typer.Option(False, "--verbose", help="Extra detail in output"),
) -> None:
    """Run an RCA investigation."""
    from agentm.cli.run import run_investigation

    import asyncio
    asyncio.run(run_investigation(
        data_dir=data_dir,
        incident=incident,
        scenario_dir=scenario,
        config_path=config,
        debug_mode=debug,
        verbose=verbose,
    ))


@app.command()
def debug(
    trajectory_file: str = typer.Argument(help="Path to .jsonl trajectory file"),
    summary: bool = typer.Option(False, "--summary", help="Print summary statistics"),
    timeline: bool = typer.Option(False, "--timeline", help="Show tool call timeline"),
    filter_agent: str | None = typer.Option(None, "--filter-agent", help="Filter by agent path prefix"),
    filter_type: str | None = typer.Option(None, "--filter-type", help="Filter by event_type"),
) -> None:
    """Analyze a trajectory JSONL file."""
    from agentm.cli.debug import analyze_trajectory

    analyze_trajectory(
        trajectory_file=trajectory_file,
        show_summary=summary,
        show_timeline=timeline,
        filter_agent=filter_agent,
        filter_type=filter_type,
    )


def main() -> None:
    app()
