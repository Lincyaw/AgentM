"""Replay-fork CLI.

Session config (scenario, provider, data_dir) is auto-restored from
the source session. Only the harness model needs to be specified.

Example::

    uv run python -m rca_eval.replay_fork.cli \
        --session abc123 --session def456 \
        --harness-model doubao \
        --out runs/replay-fork/results.jsonl
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer

app = typer.Typer(
    name="replay-fork",
    help=__doc__,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def _read_session_file(path: Path, *, column: str) -> list[str]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return []

    first = lines[0]
    delimiter = "\t" if "\t" in first else "," if "," in first else None
    if delimiter is not None:
        header = next(csv.reader([first], delimiter=delimiter))
        if column in header:
            reader = csv.DictReader(lines, delimiter=delimiter)
            return [
                value
                for row in reader
                if (value := (row.get(column) or "").strip())
            ]
        if len(header) > 1:
            raise ValueError(
                f"{path} looks tabular but has no column {column!r}; "
                f"available columns: {', '.join(header)}"
            )

    ids: list[str] = []
    header_names = {column, "session_id", "baseline_session_id"}
    for line in lines:
        value = line.split()[0].strip()
        if value and value not in header_names:
            ids.append(value)
    return ids


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _read_existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("case_id"), str):
            ids.add(payload["case_id"])
    return ids


@app.callback(invoke_without_command=True)
def run(
    session: Annotated[
        list[str] | None,
        typer.Option("--session", help="Baseline session id(s) (repeatable)"),
    ] = None,
    session_file: Annotated[
        list[Path] | None,
        typer.Option(
            "--session-file",
            help=(
                "File containing baseline sessions. Plain one-id-per-line files "
                "and TSV/CSV files are supported."
            ),
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
            help="Skip case_id values already present in --out and append new results.",
        ),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Run at most this many remaining sessions."),
    ] = None,
    harness_model: Annotated[
        str, typer.Option("--harness-model", help="config.toml profile for extractor+auditor"),
    ] = "doubao",
    auditor_prompt: Annotated[
        str, typer.Option("--auditor-prompt", help="auditor prompt variant"),
    ] = "minimal_index",
    audit_interval: Annotated[
        int, typer.Option("--audit-interval"),
    ] = 5,
    max_turns: Annotated[
        int, typer.Option("--max-turns"),
    ] = 60,
    max_forks: Annotated[
        int,
        typer.Option(
            "--max-forks",
            help="Maximum sequential reminder forks per baseline session.",
        ),
    ] = 1,
    fork_live_harness: Annotated[
        bool,
        typer.Option(
            "--fork-live-harness/--no-fork-live-harness",
            help=(
                "After the first offline surface, mount llmharness into the "
                "fork continuation as well so the fork can receive additional "
                "per-turn reminders."
            ),
        ),
    ] = False,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-j"),
    ] = 1,
    out: Annotated[
        Path, typer.Option("--out", help="results JSONL path"),
    ] = Path("runs/replay-fork/results.jsonl"),
    obs_dir: Annotated[
        Path | None,
        typer.Option(
            "--obs-dir",
            help=(
                "Force a JSONL observability directory. Omit to use the "
                "default session store, which prefers ClickHouse when "
                "available and falls back to local JSONL."
            ),
        ),
    ] = None,
) -> None:
    """Run replay-fork over baseline sessions."""
    import logging as _stdlib_logging

    from loguru import logger as _logger

    class _InterceptHandler(_stdlib_logging.Handler):
        def emit(self, record: _stdlib_logging.LogRecord) -> None:
            level: str | int
            try:
                level = _logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame: FrameType | None
            frame, depth = _stdlib_logging.currentframe(), 2
            while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
                frame = frame.f_back
                depth += 1
            _logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    _logger.remove()
    _logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level: <7}</level> <cyan>{file}:{line}</cyan> <level>{message}</level>")
    _stdlib_logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    _stdlib_logging.getLogger("httpx").setLevel(_stdlib_logging.WARNING)
    _stdlib_logging.getLogger("opentelemetry").setLevel(_stdlib_logging.WARNING)
    _stdlib_logging.getLogger("agentm.core.runtime.catalog").setLevel(_stdlib_logging.WARNING)

    from agentm.core.runtime.session_bootstrap import make_default_session_store
    from agentm.core.runtime.session_manager import JsonlSessionStore

    from .api import replay_batch
    from .providers import build_profile_provider

    sessions = list(session or [])
    for path in session_file or []:
        try:
            sessions.extend(_read_session_file(path, column=session_column))
        except Exception as exc:
            raise typer.BadParameter(str(exc), param_hint="--session-file") from exc

    sessions = _unique_preserve_order(sessions)
    if skip_existing:
        completed = _read_existing_case_ids(out)
        before_skip = len(sessions)
        sessions = [sid for sid in sessions if sid not in completed]
        skipped = before_skip - len(sessions)
    else:
        skipped = 0
    if limit is not None:
        sessions = sessions[:limit]
    if not sessions:
        typer.echo(
            f"# sessions: 0  skipped_existing: {skipped}\n"
            "# no sessions to process"
        )
        return

    harness_prov = build_profile_provider(harness_model)
    if obs_dir is None:
        store = make_default_session_store(str(Path.cwd()))
    else:
        store = JsonlSessionStore(session_dir=obs_dir.resolve())

    typer.echo(
        f"# harness: {harness_prov[1].get('model')}\n"
        f"# auditor_prompt: {auditor_prompt}\n"
        f"# fork_live_harness: {fork_live_harness}\n"
        f"# max_forks: {max_forks}\n"
        f"# sessions: {len(sessions)}  skipped_existing: {skipped}  concurrency: {concurrency}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a" if skip_existing else "w", encoding="utf-8")

    def _on_result(result, done, total):  # type: ignore[no-untyped-def]
        fh.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + "\n")
        fh.flush()
        tag = "FIRE" if result.fired else "----"
        ctrl = "Y" if result.control_correct else "N"
        iv = "Y" if result.intervene_correct else ("N" if result.intervene_correct is not None else "-")
        flip = ""
        if result.helped:
            flip = " HELPED"
        elif result.harmed:
            flip = " HARMED"
        typer.echo(f"  [{done}/{total}] {tag} ctrl={ctrl} iv={iv}{flip} {result.case_id}")
        ctrl_summary = result.control_submission_summary or {}
        iv_summary = result.intervene_submission_summary or {}
        if ctrl_summary:
            typer.echo(f"    control roots:   {ctrl_summary.get('root_causes')}")
        if iv_summary and iv_summary != ctrl_summary:
            typer.echo(f"    intervene roots: {iv_summary.get('root_causes')}")
        for attempt in getattr(result, "fork_attempts", []):
            roots = None
            if attempt.fork_submission_summary:
                roots = attempt.fork_submission_summary.get("root_causes")
            ok = "Y" if attempt.fork_correct else ("N" if attempt.fork_correct is not None else "-")
            fork_sid = attempt.forked_session_id or "-"
            typer.echo(
                f"    gen {attempt.generation}: src={attempt.source_session_id} "
                f"turn={attempt.surface_turn} fork={fork_sid} ok={ok} roots={roots}"
            )
        typer.echo(f"    baseline: agentm trace messages --session {result.case_id} --format text")
        if result.forked_session_id:
            typer.echo(f"    fork:     agentm trace messages --session {result.forked_session_id} --format text")
        for f in getattr(result, "audit_firings", []):
            ext = f.extractor_session_id or "-"
            aud = f.auditor_session_id or "-"
            sfx = " ★" if f.surfaced else ""
            typer.echo(f"    turn {f.turn_number:>3}: ext={ext} aud={aud}{sfx}")

    try:
        summary = asyncio.run(
            replay_batch(
                sessions,
                store=store,
                harness_provider=harness_prov,
                audit_interval=audit_interval,
                auditor_prompt=auditor_prompt,
                max_turns=max_turns,
                fork_live_harness=fork_live_harness,
                max_forks=max_forks,
                concurrency=concurrency,
                on_result=_on_result,
            )
        )
    finally:
        fh.close()

    typer.echo(f"\n=== replay-fork ===\n{summary.format()}\n# results: {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
