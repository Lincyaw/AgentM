"""CLI for generic rescue-window branch experiments."""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from .llmharness_policy import LlmHarnessPolicy
from .policy_runner import PolicyRunConfig, run_policy_over_sessions
from .provider_profiles import build_profile_provider
from .runner import run_specs
from .schema import load_experiment_spec
from .session_inputs import (
    read_existing_source_ids,
    read_session_file,
    unique_preserve_order,
)


app = typer.Typer(
    name="rescue-window",
    help=__doc__,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main() -> None:
    """Load AgentM's shared environment before any command runs."""

    from agentm.cli import autoload_dotenv

    autoload_dotenv(Path.cwd())


@app.command()
def plan(
    spec: Annotated[Path, typer.Argument(help="Experiment spec YAML/JSON.")],
) -> None:
    """Validate a spec and print the planned branches."""

    experiment = load_experiment_spec(spec)
    rows = [
        {
            "branch_id": branch.branch_id,
            "source_session_id": branch.source_session_id,
            "fork_point": branch.fork_point.to_dict(),
            "policy_id": branch.policy_id,
            "condition_id": branch.intervention.condition_id,
            "action": branch.intervention.action.value,
            "content_level": branch.intervention.content_level,
        }
        for branch in experiment.branches
    ]
    typer.echo(
        json.dumps(
            {
                "experiment_id": experiment.experiment_id,
                "schema_version": experiment.schema_version,
                "branches": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("render-message")
def render_message(
    spec: Annotated[Path, typer.Argument(help="Experiment spec YAML/JSON.")],
    branch_id: Annotated[str, typer.Argument(help="Branch id.")],
) -> None:
    """Print the actor-visible message for one branch."""

    experiment = load_experiment_spec(spec)
    for branch in experiment.branches:
        if branch.branch_id == branch_id:
            typer.echo(branch.intervention.message)
            return
    raise typer.BadParameter(f"unknown branch_id: {branch_id}")


@app.command()
def run(
    spec: Annotated[Path, typer.Argument(help="Experiment spec YAML/JSON.")],
    out: Annotated[Path, typer.Option("--out", help="Output branch JSONL path.")],
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Run only this branch id. Repeatable."),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", "-j", min=1)] = 1,
) -> None:
    """Run explicit-intervention branches from a spec."""

    experiment = load_experiment_spec(spec)
    selected = experiment.branches
    if only:
        allowed = set(only)
        selected = [branch for branch in selected if branch.branch_id in allowed]
    if limit is not None:
        selected = selected[:limit]
    results = asyncio.run(run_specs(selected, out_jsonl=out, concurrency=concurrency))
    typer.echo(
        json.dumps(
            {
                "experiment_id": experiment.experiment_id,
                "branches": len(results),
                "out": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("llmharness")
def run_llmharness(
    session: Annotated[
        list[str] | None,
        typer.Option("--session", help="Source session id(s) (repeatable)."),
    ] = None,
    session_file: Annotated[
        list[Path] | None,
        typer.Option(
            "--session-file",
            help="Plain, CSV, or TSV file containing source session ids.",
        ),
    ] = None,
    session_column: Annotated[
        str,
        typer.Option(
            "--session-column",
            help="Column to read from tabular --session-file input.",
        ),
    ] = "baseline_session_id",
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing",
            help="Skip source sessions already present in --out and append new rows.",
        ),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Run at most this many remaining sessions."),
    ] = None,
    harness_model: Annotated[
        str,
        typer.Option(
            "--harness-model",
            help="config.toml profile for llmharness extractor/auditor sessions.",
        ),
    ] = "doubao",
    auditor_prompt: Annotated[
        str,
        typer.Option("--auditor-prompt", help="llmharness auditor prompt variant."),
    ] = "rca_fork",
    audit_interval: Annotated[
        int,
        typer.Option("--audit-interval", min=1),
    ] = 5,
    max_turns: Annotated[
        int,
        typer.Option("--max-turns", min=1),
    ] = 60,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-j", min=1),
    ] = 1,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output branch result JSONL path."),
    ] = Path("runs/rescue-window/llmharness-branches.jsonl"),
    obs_dir: Annotated[
        Path | None,
        typer.Option(
            "--obs-dir",
            help=(
                "Force a JSONL observability directory. Omit to use the default "
                "session store, which may query ClickHouse."
            ),
        ),
    ] = None,
) -> None:
    """Run the llmharness policy over recorded sessions and fork surfaced cases."""

    from agentm.core.runtime.session_bootstrap import make_default_session_store
    from agentm.core.runtime.session_manager import JsonlSessionStore

    sessions = list(session or [])
    for path in session_file or []:
        try:
            sessions.extend(read_session_file(path, column=session_column))
        except Exception as exc:
            raise typer.BadParameter(str(exc), param_hint="--session-file") from exc
    sessions = unique_preserve_order(sessions)
    skipped = 0
    if skip_existing:
        completed = read_existing_source_ids(out)
        before_skip = len(sessions)
        sessions = [sid for sid in sessions if sid not in completed]
        skipped = before_skip - len(sessions)
    if limit is not None:
        sessions = sessions[:limit]
    if not sessions:
        typer.echo(
            f"# sessions: 0  skipped_existing: {skipped}\n"
            "# no sessions to process"
        )
        return

    provider = build_profile_provider(harness_model)
    store = (
        JsonlSessionStore(session_dir=obs_dir.resolve())
        if obs_dir is not None
        else make_default_session_store(str(Path.cwd()))
    )
    policy = LlmHarnessPolicy(
        provider=provider,
        auditor_prompt=auditor_prompt,
        audit_interval=audit_interval,
    )
    typer.echo(
        f"# policy: {policy.policy_id}\n"
        f"# harness: {provider[1].get('model')}\n"
        f"# auditor_prompt: {auditor_prompt}\n"
        f"# sessions: {len(sessions)}  skipped_existing: {skipped}  "
        f"concurrency: {concurrency}"
    )

    def _on_result(result, done, total):  # type: ignore[no-untyped-def]
        tag = result.status.upper()
        fork = result.fork_session_id or "-"
        surface = result.intervention.get("metadata", {}).get("surface_turn_index")
        typer.echo(
            f"  [{done}/{total}] {tag} source={result.source_session_id} "
            f"surface={surface if surface is not None else '-'} fork={fork}"
        )
        if result.error:
            typer.echo(f"    error: {result.error}")

    results = asyncio.run(
        run_policy_over_sessions(
            sessions,
            policy=policy,
            out_jsonl=out,
            store=store,
            config=PolicyRunConfig(max_turns=max_turns),
            concurrency=concurrency,
            append=skip_existing,
            on_result=_on_result,
        )
    )
    counts = {
        "succeeded": sum(1 for row in results if row.status == "succeeded"),
        "skipped": sum(1 for row in results if row.status == "skipped"),
        "failed": sum(1 for row in results if row.status == "failed"),
    }
    typer.echo(
        json.dumps(
            {"branches": len(results), "counts": counts, "out": str(out)},
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def export(
    results: Annotated[Path, typer.Argument(help="Branch result JSONL.")],
    adapter: Annotated[str, typer.Option("--adapter", help="Export adapter name.")],
    out_prefix: Annotated[
        Path | None,
        typer.Option("--out-prefix", help="Adapter-specific output prefix."),
    ] = None,
) -> None:
    """Dispatch adapter-specific exports.

    The generic package does not score tasks. Adapters live with their scenario.
    """

    if adapter != "rca":
        raise typer.BadParameter("only --adapter rca is currently registered")
    from rca_eval.rescue_window_adapter import build_export_commands

    commands = build_export_commands(results, out_prefix=out_prefix)
    for command in commands:
        typer.echo(command)


@app.command("export-run")
def export_run(
    results: Annotated[Path, typer.Argument(help="Branch result JSONL.")],
    adapter: Annotated[str, typer.Option("--adapter", help="Export adapter name.")],
    out_prefix: Annotated[Path, typer.Option("--out-prefix")],
) -> None:
    """Run an adapter export command set."""

    if adapter != "rca":
        raise typer.BadParameter("only --adapter rca is currently registered")
    from rca_eval.rescue_window_adapter import build_export_commands

    for command in build_export_commands(results, out_prefix=out_prefix):
        subprocess.run(shlex.split(command), check=True)


if __name__ == "__main__":
    app()
