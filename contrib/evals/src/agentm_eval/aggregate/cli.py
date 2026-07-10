"""``llmharness-aggregate`` CLI: AgentM trace sessions -> case directories."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .fork_collector import export_forks
from .session_collector import collect_session_case, trace_ids_for_sessions
from .writer import write_case

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fold AgentM trace sessions into canonical case directories.",
)


def _session_ids_from_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped.split()[0])
    return out


def _unique_session_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _emit_session(  # type: ignore[no-untyped-def]
    session_id: str,
    case_dir: Path,
    case_meta,
    main_agent_messages: int,
) -> None:
    typer.echo(
        f"clickhouse:{session_id} -> {case_dir}/ "
        f"(messages={main_agent_messages} "
        f"ext={case_meta.extractor_firings} aud={case_meta.auditor_firings})"
    )


@app.command()
def sessions(
    out: Annotated[
        Path,
        typer.Option("--out", help="Output cases/ root.", resolve_path=True),
    ],
    session_id: Annotated[
        list[str] | None,
        typer.Option(
            "--session-id",
            help="AgentM session id to fetch from ClickHouse. Repeatable.",
        ),
    ] = None,
    session_file: Annotated[
        Path | None,
        typer.Option(
            "--session-file",
            help="Text file containing one session id per line.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    case_prefix: Annotated[
        str,
        typer.Option(
            "--case-prefix",
            help="Prefix to prepend to generated case ids when sample_id is not set.",
        ),
    ] = "",
    sample_id: Annotated[
        str | None,
        typer.Option(
            "--sample-id",
            help="Override case sample_id. Only valid with exactly one session.",
        ),
    ] = None,
    dataset_name: Annotated[
        str | None, typer.Option("--dataset-name", help="Override case dataset_name.")
    ] = None,
    dataset_path: Annotated[
        str | None, typer.Option("--dataset-path", help="Override case dataset_path.")
    ] = None,
) -> None:
    """Aggregate AgentM ClickHouse session trajectories into case directories."""

    requested = list(session_id or [])
    if session_file is not None:
        requested.extend(_session_ids_from_file(session_file))
    sessions_to_export = _unique_session_ids(requested)
    if not sessions_to_export:
        typer.echo("no session ids provided", err=True)
        raise typer.Exit(2)
    if sample_id is not None and len(sessions_to_export) != 1:
        typer.echo("--sample-id is only valid with exactly one session", err=True)
        raise typer.Exit(2)

    out.mkdir(parents=True, exist_ok=True)
    trace_ids = trace_ids_for_sessions(set(sessions_to_export))
    for sid in sessions_to_export:
        case_id = sample_id or f"{case_prefix}{sid}"
        case = collect_session_case(
            session_id=sid,
            trace_id=trace_ids.get(sid),
            case_id=case_id,
            sample_id_override=sample_id,
            dataset_name_override=dataset_name,
            dataset_path_override=dataset_path,
        )
        case_dir = write_case(case, out)
        _emit_session(sid, case_dir, case.meta, len(case.main_agent_messages))


@app.command()
def forks(
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run-dir",
            help="RCA rescue-window run directory containing score CSVs.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output fork-review root.", resolve_path=True),
    ],
    forks_tsv: Annotated[
        Path | None,
        typer.Option(
            "--forks-tsv",
            help="Fork index TSV. Defaults to <run-dir>/forks.tsv.",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    include_full: Annotated[
        bool,
        typer.Option(
            "--include-full/--no-include-full",
            help="Also write source_full.jsonl and fork_full.jsonl for each fork.",
        ),
    ] = True,
) -> None:
    """Export reminder-fork before/after trajectories from ClickHouse."""

    resolved_forks_tsv = forks_tsv or (run_dir / "forks.tsv")
    if not resolved_forks_tsv.is_file():
        typer.echo(f"fork TSV not found: {resolved_forks_tsv}", err=True)
        raise typer.Exit(2)
    export_forks(
        run_dir=run_dir,
        forks_tsv=resolved_forks_tsv,
        out_dir=out,
        include_full=include_full,
    )
    typer.echo(f"{resolved_forks_tsv} -> {out}/")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
