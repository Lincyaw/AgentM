"""Dataset export support for ``agentm trace``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from agentm.cli.trace.backend import trace_clickhouse_url, trace_cwd
from agentm.cli.trace.format import _fail, _info
from agentm.env import autoload_dotenv

CwdOpt = Annotated[
    Path | None,
    typer.Option(
        "--cwd",
        help=(
            "Working directory for .env loading and ClickHouse filtering. "
            "Defaults to AGENTM_CWD, then the process cwd."
        ),
    ),
]


def export_dataset_cmd(
    output: Annotated[
        Path,
        typer.Argument(help="Output file path (.parquet or .jsonl)."),
    ],
    cwd: CwdOpt = None,
    scenario: Annotated[
        list[str] | None,
        typer.Option("--scenario", help="Filter by scenario (repeatable)."),
    ] = None,
    purpose: Annotated[
        list[str] | None,
        typer.Option("--purpose", help="Filter by purpose (repeatable)."),
    ] = None,
    session: Annotated[
        list[str] | None,
        typer.Option("--session", help="Export specific session IDs (repeatable)."),
    ] = None,
    roots_only: Annotated[
        bool,
        typer.Option("--roots-only", help="Only root sessions (no children)."),
    ] = False,
    include_system_prompt: Annotated[
        bool,
        typer.Option("--system-prompt", help="Include the system prompt."),
    ] = False,
    include_thinking: Annotated[
        bool,
        typer.Option("--thinking", help="Include assistant thinking blocks."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Max sessions to export."),
    ] = None,
    compression: Annotated[
        str,
        typer.Option(
            "--compression",
            help=(
                "Parquet compression "
                "(zstd, snappy, gzip, uncompressed, brotli, lz4)."
            ),
        ),
    ] = "zstd",
    expand_children: Annotated[
        bool,
        typer.Option(
            "--expand-children",
            help="For workflow sessions, auto-expand to child agent sessions.",
        ),
    ] = False,
) -> None:
    """Export traces to a HuggingFace-compatible Parquet or JSONL dataset."""
    from agentm.dataset_export import DatasetExporter

    autoload_dotenv(trace_cwd(cwd))

    scenarios = set(scenario) if scenario else None
    purposes = set(purpose) if purpose else None
    session_ids = list(session) if session else None

    clickhouse_url = trace_clickhouse_url(cwd)
    if clickhouse_url is not None:
        exporter = DatasetExporter.from_clickhouse(clickhouse_url)
        _info("Using ClickHouse backend")
    else:
        exporter = DatasetExporter.from_local()
        _info("Using local JSONL backend")

    suffix = output.suffix.lower()
    if suffix not in {".jsonl", ".parquet"}:
        _fail(
            2,
            "bad-output-suffix",
            f"unsupported output suffix {suffix!r}",
            fix="use a .parquet or .jsonl output path",
        )
    if suffix == ".jsonl":
        count = exporter.export_jsonl(
            output,
            session_ids=session_ids,
            scenarios=scenarios,
            purposes=purposes,
            roots_only=roots_only,
            include_system_prompt=include_system_prompt,
            include_thinking=include_thinking,
            expand_children=expand_children,
            limit=limit,
        )
    else:
        count = exporter.export_parquet(
            output,
            session_ids=session_ids,
            scenarios=scenarios,
            purposes=purposes,
            roots_only=roots_only,
            include_system_prompt=include_system_prompt,
            include_thinking=include_thinking,
            expand_children=expand_children,
            compression=compression,
            limit=limit,
        )

    _info(f"Exported {count} conversation(s) to {output}")


__all__ = ["export_dataset_cmd"]
