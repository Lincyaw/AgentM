"""``llmharness-aggregate`` CLI: replay sidecar(s) → per-case directories.

Two input layouts, both writing the same canonical case-directory shape
documented in ``docs/06-case-aggregation.md``:

    replay   — walk ``<cwd>/.agentm/audit_replay/*.jsonl`` (live runs)
    one      — aggregate a single replay-format JSONL file

Each takes ``--out`` for the destination ``cases/`` root, plus optional
``--sample-id`` / ``--dataset-name`` / ``--dataset-path`` overrides for
runs that did not mount ``llmharness.distill.binding`` (so no meta
sidecar is present).

Examples::

    # Live-run layout: every session under <cwd>/.agentm/audit_replay/
    llmharness-aggregate replay --cwd /run --out ./cases

    # Single session from a live run
    llmharness-aggregate replay --cwd /run --session-id abc123 --out ./cases

    # Single replay JSONL file (e.g. inspect a hand-edited record set)
    llmharness-aggregate one --replay-path /tmp/abc.jsonl --out ./cases \\
        --sample-id rca-mysql-001 --dataset-name rca-openrca2-lite
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .collector import collect_case
from .session_collector import collect_session_case, trace_ids_for_sessions
from .writer import write_case

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fold replay sidecar(s) into canonical case directories.",
)


def _emit(replay_path: Path, case_dir: Path, case_meta) -> None:  # type: ignore[no-untyped-def]
    typer.echo(
        f"{replay_path} → {case_dir}/ "
        f"(ext={case_meta.extractor_firings} aud={case_meta.auditor_firings} "
        f"surfaced={case_meta.surfaced_reminders})"
    )


def _aggregate_one(
    replay_path: Path,
    out_dir: Path,
    *,
    meta_path: Path | None = None,
    sample_id: str | None = None,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
) -> None:
    case = collect_case(
        replay_path=replay_path,
        meta_path=meta_path if (meta_path and meta_path.is_file()) else None,
        sample_id_override=sample_id,
        dataset_name_override=dataset_name,
        dataset_path_override=dataset_path,
    )
    case_dir = write_case(case, out_dir)
    _emit(replay_path, case_dir, case.meta)


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
def replay(
    cwd: Annotated[
        Path,
        typer.Option(
            "--cwd",
            help="Run directory containing .agentm/audit_replay/",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output cases/ root.", resolve_path=True),
    ],
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Aggregate only this session; default = every sidecar in --cwd.",
        ),
    ] = None,
    sample_id: Annotated[
        str | None,
        typer.Option(
            "--sample-id",
            help=(
                "Override case sample_id (e.g. for rca llm-eval runs without "
                "distill.binding). Applies to every aggregated session — pair "
                "with --session-id for per-sample runs."
            ),
        ),
    ] = None,
    dataset_name: Annotated[
        str | None,
        typer.Option("--dataset-name", help="Override case dataset_name."),
    ] = None,
    dataset_path: Annotated[
        str | None,
        typer.Option("--dataset-path", help="Override case dataset_path."),
    ] = None,
) -> None:
    """Walk ``<cwd>/.agentm/audit_replay/`` and aggregate every (or one) session."""
    replay_dir = cwd / ".agentm" / "audit_replay"
    if not replay_dir.is_dir():
        typer.echo(f"no replay dir at {replay_dir}; nothing to aggregate", err=True)
        raise typer.Exit(2)

    if session_id:
        candidate = replay_dir / f"{session_id}.jsonl"
        sessions = [candidate] if candidate.is_file() else []
    else:
        sessions = sorted(replay_dir.glob("*.jsonl"))
    if not sessions:
        typer.echo(f"no replay sidecars matched in {replay_dir}", err=True)
        raise typer.Exit(2)

    out.mkdir(parents=True, exist_ok=True)
    for path in sessions:
        _aggregate_one(
            path,
            out,
            meta_path=path.with_suffix(".meta.json"),
            sample_id=sample_id,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )


@app.command()
def one(
    replay_path: Annotated[
        Path,
        typer.Option(
            "--replay-path",
            help="One replay-format JSONL file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output cases/ root.", resolve_path=True),
    ],
    meta_path: Annotated[
        Path | None,
        typer.Option(
            "--meta-path",
            help="Optional .meta.json sidecar (distill.binding shape).",
        ),
    ] = None,
    sample_id: Annotated[
        str | None, typer.Option("--sample-id", help="Override case sample_id.")
    ] = None,
    dataset_name: Annotated[
        str | None, typer.Option("--dataset-name", help="Override case dataset_name.")
    ] = None,
    dataset_path: Annotated[
        str | None, typer.Option("--dataset-path", help="Override case dataset_path.")
    ] = None,
) -> None:
    """Aggregate a single replay-format JSONL into one case directory."""
    out.mkdir(parents=True, exist_ok=True)
    _aggregate_one(
        replay_path,
        out,
        meta_path=meta_path,
        sample_id=sample_id,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
