"""Unified eval CLI: agentm eval <benchmark> <command> [options]."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from agentm_eval.experiment import Experiment, _default_output_root
from agentm_eval.registry import discover, list_benchmarks

app = typer.Typer(
    name="eval",
    help="Run and manage benchmark evaluations.",
    add_completion=False,
)

# Register adapter sub-apps at import time so typer knows about them
# before command lookup.
discover()

from agentm_eval import registry as _reg  # noqa: E402

for _name, _desc in _reg.list_benchmarks():
    try:
        _sub = _reg.get_cli(_name)
        app.add_typer(_sub, name=_name)
    except Exception as _e:
        logger.debug("Failed to register benchmark CLI {}: {}", _name, _e)


@app.command("list")
def list_bench() -> None:
    """List available benchmarks."""
    benches = list_benchmarks()
    if not benches:
        typer.echo("No benchmarks available.")
        return
    typer.echo(f"{'Benchmark':<25} Description")
    typer.echo("-" * 60)
    for name, desc in benches:
        typer.echo(f"{name:<25} {desc}")


@app.command("runs")
def list_runs(
    benchmark: Annotated[str | None, typer.Option("--benchmark", "-b")] = None,
    limit: Annotated[int, typer.Option("-n")] = 20,
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
) -> None:
    """List past experiment runs."""
    root = output_root or _default_output_root()
    if not root.is_dir():
        typer.echo("No experiments found.")
        return

    runs: list[dict] = []
    for d in sorted(root.iterdir(), reverse=True):
        meta_file = d / "meta.json"
        if not meta_file.is_file():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if benchmark and meta.get("benchmark") != benchmark:
            continue
        runs.append(meta)
        if len(runs) >= limit:
            break

    if not runs:
        typer.echo("No experiments found.")
        return

    typer.echo(f"{'Exp ID':<45} {'Benchmark':<20} {'Model':<16} {'Status':<10}")
    typer.echo("-" * 91)
    for m in runs:
        typer.echo(
            f"{m['exp_id']:<45} {m['benchmark']:<20} "
            f"{(m.get('model') or '-'):<16} {m.get('status', '?'):<10}"
        )


@app.command("report")
def show_report(
    exp_id: Annotated[str, typer.Argument(help="Experiment ID")],
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
) -> None:
    """Show results for an experiment."""
    exp = Experiment.load(exp_id, output_root)
    meta_file = exp.output_dir / "meta.json"
    meta = json.loads(meta_file.read_text())

    typer.echo(f"Experiment: {exp.exp_id}")
    typer.echo(f"Benchmark:  {exp.benchmark}")
    typer.echo(f"Model:      {exp.model or '-'}")
    typer.echo(f"Status:     {meta.get('status', '?')}")
    typer.echo(f"Started:    {meta.get('start_time', '?')}")
    if meta.get("end_time"):
        typer.echo(f"Ended:      {meta['end_time']}")
    typer.echo(f"Output:     {exp.output_dir}")
    typer.echo("")

    sessions = exp.load_sessions()
    if sessions:
        typer.echo(f"Sessions: {len(sessions)} registered")

    results = exp.load_results()
    if not results:
        typer.echo("No results recorded.")
        return

    n_pass = sum(1 for r in results if r.get("status") == "pass")
    n_fail = sum(1 for r in results if r.get("status") == "fail")
    n_error = sum(1 for r in results if r.get("status") == "error")
    typer.echo(f"Tasks: {len(results)} total, {n_pass} pass, {n_fail} fail, {n_error} error")

    if meta.get("summary"):
        typer.echo(f"\nSummary: {json.dumps(meta['summary'], indent=2)}")

    report_file = exp.output_dir / "report.txt"
    if report_file.is_file():
        typer.echo(f"\n{report_file.read_text()}")


@app.command("sessions")
def list_sessions(
    exp_id: Annotated[str, typer.Argument(help="Experiment ID")],
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
) -> None:
    """List registered sessions for an experiment."""
    exp = Experiment.load(exp_id, output_root)
    sessions = exp.load_sessions()
    if not sessions:
        typer.echo("No sessions registered.")
        return

    typer.echo(f"{'Session ID':<36} {'Case ID':<40} {'Registered':<24}")
    typer.echo("-" * 100)
    for s in sessions:
        sid = s.get("session_id", "?")
        cid = s.get("case_id", "-")
        ts = s.get("registered_at", "?")[:19]
        typer.echo(f"{sid:<36} {cid:<40} {ts:<24}")
    typer.echo(f"\n{len(sessions)} sessions total")


@app.command("export")
def export_traces(
    exp_id: Annotated[str, typer.Argument(help="Experiment ID")],
    format: Annotated[str, typer.Option("--format", "-f")] = "ndjson",
    role: Annotated[str | None, typer.Option("--role", help="Filter by role (index, auditor, ...)")] = None,
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
) -> None:
    """Export trajectories for all trace sessions from ClickHouse."""
    exp = Experiment.load(exp_id, output_root)
    trace_sids = exp.trace_session_ids(role=role)
    if not trace_sids:
        typer.echo("No trace sessions recorded. Run an eval first or check `agentm-eval sessions`.")
        return

    typer.echo(f"Exporting {len(trace_sids)} trace sessions for {exp_id}...")
    exported = exp.export_all_trajectories(fmt=format, role=role)
    typer.echo(f"Exported {exported}/{len(trace_sids)} to {exp.output_dir / 'trajectories'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
