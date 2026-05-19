"""``llmharness-aggregate`` CLI: replay sidecar(s) → per-case directories.

Three input layouts, all writing the same canonical case-directory shape
documented in ``docs/06-case-aggregation.md``:

    replay   — walk ``<cwd>/.agentm/audit_replay/*.jsonl`` (live runs)
    eval-db  — walk ``<src>/<row_id>/{records.jsonl,meta.json}`` produced
               by ``llmharness adapter eval-db extract``
    one      — aggregate a single replay-format JSONL file

Each takes ``--out`` for the destination ``cases/`` root, plus optional
``--sample-id`` / ``--dataset-name`` / ``--dataset-path`` overrides for
runs that did not mount ``llmharness.distill.binding`` (so no meta
sidecar is present).

Examples::

    # Legacy live-run layout
    llmharness-aggregate replay --cwd /run --out ./cases

    # Single session from a live run
    llmharness-aggregate replay --cwd /run --root-session-id abc123 --out ./cases

    # eval-db extract output (every row → one case)
    llmharness-aggregate eval-db --src runs/eval_db/openrca-2-lite-n500-t20 \\
        --out ./cases

    # Single replay JSONL file (e.g. inspect a hand-edited record set)
    llmharness-aggregate one --replay-path /tmp/abc.jsonl --out ./cases \\
        --sample-id rca-mysql-001 --dataset-name rca-openrca2-lite
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .collector import collect_case
from .writer import write_case

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fold replay sidecar(s) into canonical case directories.",
)


def _resolve_dataset_name(
    cli_override: str | None, row_meta: dict[str, object]
) -> str | None:
    """CLI flag wins; otherwise fall back to row_meta['dataset'] if it's a string."""
    if cli_override:
        return cli_override
    ds = row_meta.get("dataset")
    return ds if isinstance(ds, str) else None


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
    root_session_id: Annotated[
        str | None,
        typer.Option(
            "--root-session-id",
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
                "with --root-session-id for per-sample runs."
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

    if root_session_id:
        candidate = replay_dir / f"{root_session_id}.jsonl"
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


@app.command(name="eval-db")
def eval_db(
    src: Annotated[
        Path,
        typer.Option(
            "--src",
            help=(
                "Root produced by `llmharness adapter eval-db extract`, "
                "typically <out-dir>/<exp_id>/. Contains <row_id>/records.jsonl."
            ),
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
    sample_id_template: Annotated[
        str,
        typer.Option(
            "--sample-id-template",
            help=(
                "Format string for sample_id; placeholders: {exp_id}, {row_id}. "
                "Defaults to a compact, sortable form."
            ),
        ),
    ] = "{exp_id}-row{row_id}",
    dataset_name: Annotated[
        str | None,
        typer.Option(
            "--dataset-name",
            help="Override dataset_name on every case. Default = meta.json `dataset`.",
        ),
    ] = None,
    dataset_path: Annotated[
        str | None,
        typer.Option("--dataset-path", help="Override dataset_path on every case."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Cap row count after row-id filtering; 0 = no cap."),
    ] = 0,
    row_ids: Annotated[
        list[int] | None,
        typer.Option(
            "--id",
            help="Specific row_id (repeatable). Default = every row under --src.",
        ),
    ] = None,
) -> None:
    """Aggregate every row produced by ``eval-db extract``.

    Each row's ``records.jsonl`` is a ReplayRecord stream, so the collector
    consumes it directly. Per-row ``meta.json`` supplies exp_id / row_id /
    dataset for sample-id derivation.
    """
    rows: list[Path] = []
    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "records.jsonl").is_file():
            continue
        row_int: int | None
        try:
            row_int = int(child.name)
        except ValueError:
            row_int = None
        if row_ids and (row_int is None or row_int not in row_ids):
            continue
        rows.append(child)
    if limit and limit > 0:
        rows = rows[:limit]
    if not rows:
        typer.echo(f"no eval-db rows matched in {src}", err=True)
        raise typer.Exit(2)

    out.mkdir(parents=True, exist_ok=True)
    for row_dir in rows:
        records = row_dir / "records.jsonl"
        meta_file = row_dir / "meta.json"
        row_meta: dict[str, object] = {}
        if meta_file.is_file():
            try:
                row_meta = json.loads(meta_file.read_text())
            except json.JSONDecodeError:
                typer.echo(
                    f"warning: {meta_file} is not valid JSON; aggregating without it",
                    err=True,
                )

        sid = sample_id_template.format(
            exp_id=row_meta.get("exp_id", "unknown"),
            row_id=row_meta.get("row_id", row_dir.name),
        )
        _aggregate_one(
            records,
            out,
            meta_path=None,  # eval-db meta.json is run-level, not the distill.binding shape
            sample_id=sid,
            dataset_name=_resolve_dataset_name(dataset_name, row_meta),
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
