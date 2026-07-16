"""Directory-granular session topology support for ``agentm trace``."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

from agentm.cli.trace.backend import clickhouse, trace_clickhouse_url, trace_cwd
from agentm.cli.trace.format import (
    _emit,
    _fail,
    _info,
    _open_output,
    _resolve_format,
)
from agentm.core.abi import TraceReader
from agentm.env import autoload_dotenv

_CACHE_FILE = ".trace_index_cache.json"

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
FormatOpt = Annotated[
    str | None,
    typer.Option(
        "--format",
        help="Output format: ndjson, json, text. Defaults by TTY.",
    ),
]
OutputOpt = Annotated[
    Path | None,
    typer.Option("--output", help="Write output to this path instead of stdout."),
]
LimitOpt = Annotated[
    int | None,
    typer.Option("--limit", help="Stop after N records."),
]


def _scan_file(path: Path) -> dict[str, Any] | None:
    """Scan one trace file for its identity and record count."""
    identity, line_count = TraceReader(path).scan_identity_and_line_count()
    if identity is None:
        return None
    return {
        "path": str(path),
        "trace_id": identity.trace_id,
        "session_id": identity.session_id,
        "parent_session_id": identity.parent_session_id,
        "purpose": identity.purpose,
        "scenario": identity.scenario,
        "records": line_count or None,
    }


def _load_index_cache(obs_dir: Path) -> dict[str, Any]:
    cache_path = obs_dir / _CACHE_FILE
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.debug(
            "trace: index cache unreadable at {}, rebuilding: {}",
            cache_path,
            exc,
        )
    return {}


def _save_index_cache(obs_dir: Path, cache: dict[str, Any]) -> None:
    cache_path = obs_dir / _CACHE_FILE
    temporary = cache_path.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, separators=(",", ":"))
        temporary.replace(cache_path)
    except OSError:
        temporary.unlink(missing_ok=True)


def _index_filter(
    rows: Iterable[dict[str, Any]],
    *,
    trace_id: str | None,
    purposes: set[str],
    scenarios: set[str],
    roots_only: bool,
    children_of: str | None,
    min_records: int | None,
) -> Iterator[dict[str, Any]]:
    for row in rows:
        if trace_id is not None and row.get("trace_id") != trace_id:
            continue
        if purposes and row.get("purpose") not in purposes:
            continue
        if scenarios and row.get("scenario") not in scenarios:
            continue
        if roots_only and row.get("parent_session_id") is not None:
            continue
        if children_of is not None and row.get("parent_session_id") != children_of:
            continue
        if min_records is not None:
            record_count = row.get("records")
            if record_count is not None and record_count < min_records:
                continue
        yield row


def index_cmd(
    cwd: CwdOpt = None,
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="Observability directory to scan (default $AGENTM_HOME/observability/).",
        ),
    ] = None,
    trace_id: Annotated[
        str | None,
        typer.Option("--trace", help="Filter by trace_id (exact match)."),
    ] = None,
    purpose: Annotated[
        list[str] | None,
        typer.Option("--purpose", help="Filter by purpose (repeatable)."),
    ] = None,
    scenario: Annotated[
        list[str] | None,
        typer.Option("--scenario", help="Filter by scenario name (repeatable)."),
    ] = None,
    roots_only: Annotated[
        bool,
        typer.Option("--roots-only", help="Only show root sessions (no parent)."),
    ] = False,
    children_of: Annotated[
        str | None,
        typer.Option(
            "--children-of",
            help="Only show sessions whose parent_session_id matches this value.",
        ),
    ] = None,
    min_records: Annotated[
        int | None,
        typer.Option(
            "--min-records",
            help="Only show sessions with at least N records.",
        ),
    ] = None,
    limit: LimitOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
    jobs: Annotated[
        int,
        typer.Option(
            "--jobs",
            "-j",
            help="Parallel workers (default: min(cpu_count, 8)).",
        ),
    ] = 0,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Bypass the on-disk index cache and re-scan every file.",
        ),
    ] = False,
) -> None:
    """Map every session file to its trace-tree identity."""
    if roots_only and children_of is not None:
        _fail(
            2,
            "argument",
            "--roots-only and --children-of are mutually exclusive",
        )

    purposes = set(purpose or [])
    scenarios = set(scenario or [])
    sink, close = _open_output(out)
    try:
        chosen_format = _resolve_format(fmt, sink)
        resolved_cwd = trace_cwd(cwd)
        autoload_dotenv(resolved_cwd)

        def render(row: dict[str, Any]) -> str:
            return (
                f"[session {row.get('session_id') or '?'}] "
                f"trace={row.get('trace_id') or '?'} "
                f"parent={row.get('parent_session_id') or '-'} "
                f"purpose={row.get('purpose') or '-'} "
                f"records={row.get('records') if row.get('records') is not None else '?'} "
                f"{row.get('path', '-')}"
            )

        if directory is None:
            clickhouse_url = trace_clickhouse_url(cwd)
            if clickhouse_url is not None:
                rows = clickhouse().index(
                    clickhouse_url,
                    trace_id=trace_id,
                    purposes=purposes or None,
                    scenarios=scenarios or None,
                    roots_only=roots_only,
                    children_of=children_of,
                )
                filtered = _index_filter(
                    rows,
                    trace_id=None,
                    purposes=set(),
                    scenarios=set(),
                    roots_only=False,
                    children_of=None,
                    min_records=min_records,
                )
                count = _emit(filtered, chosen_format, render, sink, limit)
                _info(f"{count} session(s) (clickhouse)")
                return

        if directory is not None:
            observability_dir = directory
        else:
            from agentm.core.observability.otel_export import (
                resolve_observability_dir,
            )

            observability_dir = resolve_observability_dir(resolved_cwd)
        if not observability_dir.is_dir():
            _fail(
                3,
                "not_found",
                f"observability directory not found: {observability_dir}",
                "pass --dir, run `agentm` once, or set AGENTM_OBSERVABILITY_DIR",
            )
        files = sorted(
            path
            for path in observability_dir.glob("*.jsonl")
            if path.is_file()
        )

        cache = {} if no_cache else _load_index_cache(observability_dir)
        cached_rows: list[dict[str, Any]] = []
        stale_files: list[Path] = []
        for path in files:
            entry = cache.get(path.name)
            if entry is not None:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if (
                    entry.get("_mtime") == stat.st_mtime
                    and entry.get("_size") == stat.st_size
                ):
                    row = entry.get("row")
                    if row is not None:
                        row["path"] = str(path)
                        cached_rows.append(row)
                    continue
            stale_files.append(path)

        scanned_rows: list[dict[str, Any]] = []
        if stale_files:
            max_workers = jobs if jobs > 0 else min(os.cpu_count() or 4, 8)
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                scanned = pool.map(_scan_file, stale_files)
                for path, row in zip(stale_files, scanned, strict=True):
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    cache[path.name] = {
                        "_mtime": stat.st_mtime,
                        "_size": stat.st_size,
                        "row": row,
                    }
                    if row is not None:
                        scanned_rows.append(row)

        live_names = {path.name for path in files}
        for stale_name in [name for name in cache if name not in live_names]:
            del cache[stale_name]

        if stale_files or len(cache) != len(files):
            _save_index_cache(observability_dir, cache)

        _info(
            f"{len(cached_rows)} cached, {len(scanned_rows)} scanned"
            f" ({len(stale_files)} file(s) re-scanned)"
        )
        all_rows = sorted(
            cached_rows + scanned_rows,
            key=lambda row: row["path"],
        )
        filtered = _index_filter(
            all_rows,
            trace_id=trace_id,
            purposes=purposes,
            scenarios=scenarios,
            roots_only=roots_only,
            children_of=children_of,
            min_records=min_records,
        )
        count = _emit(filtered, chosen_format, render, sink, limit)
        _info(f"{count} session file(s)")
    finally:
        if close:
            sink.close()


__all__ = ["index_cmd"]
