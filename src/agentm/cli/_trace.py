# code-health: ignore-file[AM025] -- CLI renders typed-union trace records from query/store boundaries
"""``agentm trace`` -- query trajectories via the SDK store layer."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, TypeVar, TypedDict

import typer

from agentm.cli._display import EXIT_NOT_FOUND, EXIT_TIMEOUT, is_tty, stderr_console
from agentm.cli._store import resolve_trajectory_store
from agentm.core.abi.messages import (
    ImageContent,
    OpaqueThinkingBlock,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
)
from agentm.core.abi.query import (
    SessionFilter,
    TrajectoryQueryStore,
)
from agentm.core.abi.store import TrajectoryDiagnostic
from agentm.core.abi.termination import SignalAborted
from agentm.core.abi.trajectory import Turn, TurnCheckpoint
from agentm.core.abi.trigger import UserInput
from agentm.core.lib.trajectory_query import TrajectoryStoreQueryAdapter
from agentm.presenter.trajectory.model import (
    TraceRow,
    TraceSnapshot,
    TraceTurnSummary,
    build_trace_snapshot,
)

TraceFormat = Literal["text", "ndjson"]
RecordT = TypeVar("RecordT")

_ROLE_ANSI: dict[str, str] = {
    "system": "\033[1;35m",
    "user": "\033[1;32m",
    "assistant": "\033[1;34m",
    "tool_call": "\033[1;33m",
    "tool_result": "\033[36m",
    "error": "\033[1;31m",
}
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"


class _TurnSummary(TypedDict):
    status: Literal["committed", "incomplete"]
    turn_index: int
    turn_id: str
    run_id: str
    run_step: int
    trigger_source: str
    tool_calls: list[str]
    tool_call_count: int
    tool_error_count: int
    input_tokens: int
    output_tokens: int
    cache_read: int
    model: str | None
    cause: str | None
    error_type: NotRequired[str]
    error: NotRequired[str]


class _MessageRecord(TypedDict):
    turn_index: int
    run_id: str
    run_step: int
    role: str
    content: str
    tool: NotRequired[str]
    is_error: NotRequired[bool]


class _ToolRecord(TypedDict):
    turn_index: int
    run_id: str
    run_step: int
    tool: str
    args: dict[str, object]
    is_error: bool
    result: str
    result_chars: int
    result_truncated: bool


class _StatusRecord(TypedDict):
    session_id: str
    state: Literal["active", "idle", "empty"]
    committed_turns: int
    incomplete_turns: int
    active_checkpoint: bool
    last_activity_at: float | None
    last_turn_index: int | None
    last_turn_id: str | None
    last_turn_cause: str | None
    checkpoint_turn_index: int | None
    checkpoint_turn_id: str | None
    checkpoint_run_id: str | None
    checkpoint_run_step: int | None
    checkpoint_updated_at: float | None
    diagnostic_count: int
    last_diagnostic_id: str | None


class _DiagnosticRecord(TypedDict):
    id: str
    session_id: str
    timestamp: float
    level: str
    source: str
    phase: str
    message: str
    error_type: str | None
    error_detail: str | None
    turn_id: str | None
    turn_index: int | None
    checkpoint_id: str | None


class _WatchEvent(TypedDict):
    event_id: str
    type: Literal["checkpoint", "commit", "abort", "diagnostic"]
    session_id: str
    timestamp: float
    turn_index: NotRequired[int]
    turn_id: NotRequired[str]
    run_id: NotRequired[str]
    run_step: NotRequired[int]
    cause: NotRequired[str]
    checkpoint_updated_at: NotRequired[float]
    diagnostic: NotRequired[_DiagnosticRecord]


@dataclass(frozen=True, slots=True)
class _TraceContext:
    query: TrajectoryQueryStore


def _load_trace_snapshot(
    query: TrajectoryQueryStore,
    session_id: str,
) -> TraceSnapshot:
    turns = list(query.turns(session_id))
    checkpoints = list(query.checkpoints(session_id))
    return build_trace_snapshot(session_id, turns, checkpoints)


def _load_status_record(
    query: TrajectoryQueryStore,
    session_id: str,
) -> _StatusRecord:
    turns = list(query.turns(session_id))
    checkpoints = list(query.checkpoints(session_id))
    diagnostics = list(query.diagnostics(session_id))
    checkpoint = checkpoints[0] if checkpoints else None
    last_turn = turns[-1] if turns else None
    last_diagnostic = diagnostics[-1] if diagnostics else None
    activity = [turn.timestamp for turn in turns]
    activity.extend(item.updated_at for item in checkpoints)
    activity.extend(item.timestamp for item in diagnostics)
    state: Literal["active", "idle", "empty"]
    if checkpoint is not None:
        state = "active"
    elif turns:
        state = "idle"
    else:
        state = "empty"
    return {
        "session_id": session_id,
        "state": state,
        "committed_turns": len(turns),
        "incomplete_turns": len(checkpoints),
        "active_checkpoint": checkpoint is not None,
        "last_activity_at": max(activity) if activity else None,
        "last_turn_index": last_turn.index if last_turn is not None else None,
        "last_turn_id": last_turn.id if last_turn is not None else None,
        "last_turn_cause": (
            type(last_turn.outcome.cause).__name__ if last_turn is not None else None
        ),
        "checkpoint_turn_index": (checkpoint.index if checkpoint is not None else None),
        "checkpoint_turn_id": checkpoint.id if checkpoint is not None else None,
        "checkpoint_run_id": checkpoint.run_id if checkpoint is not None else None,
        "checkpoint_run_step": (
            checkpoint.run_step if checkpoint is not None else None
        ),
        "checkpoint_updated_at": (
            checkpoint.updated_at if checkpoint is not None else None
        ),
        "diagnostic_count": len(diagnostics),
        "last_diagnostic_id": (
            last_diagnostic.id if last_diagnostic is not None else None
        ),
    }


def _diagnostic_record(diagnostic: TrajectoryDiagnostic) -> _DiagnosticRecord:
    return {
        "id": diagnostic.id,
        "session_id": diagnostic.session_id,
        "timestamp": diagnostic.timestamp,
        "level": diagnostic.level,
        "source": diagnostic.source,
        "phase": diagnostic.phase,
        "message": diagnostic.message,
        "error_type": diagnostic.error_type,
        "error_detail": diagnostic.error_detail,
        "turn_id": diagnostic.turn_id,
        "turn_index": diagnostic.turn_index,
        "checkpoint_id": diagnostic.checkpoint_id,
    }


def _watch_events(
    query: TrajectoryQueryStore,
    session_id: str,
) -> list[_WatchEvent]:
    turns = list(query.turns(session_id))
    checkpoints = list(query.checkpoints(session_id))
    diagnostics = list(query.diagnostics(session_id))
    events: list[_WatchEvent] = []
    for turn in turns:
        aborted = isinstance(turn.outcome.cause, SignalAborted)
        events.append(
            {
                "event_id": f"turn:{turn.id}",
                "type": "abort" if aborted else "commit",
                "session_id": session_id,
                "timestamp": turn.timestamp,
                "turn_index": turn.index,
                "turn_id": turn.id,
                "run_id": turn.run_id,
                "run_step": turn.run_step,
                "cause": type(turn.outcome.cause).__name__,
            }
        )
    for checkpoint in checkpoints:
        events.append(
            {
                "event_id": f"checkpoint:{checkpoint.id}:{checkpoint.updated_at}",
                "type": "checkpoint",
                "session_id": session_id,
                "timestamp": checkpoint.updated_at,
                "turn_index": checkpoint.index,
                "turn_id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "run_step": checkpoint.run_step,
                "checkpoint_updated_at": checkpoint.updated_at,
            }
        )
    for diagnostic in diagnostics:
        events.append(
            {
                "event_id": f"diagnostic:{diagnostic.id}",
                "type": "diagnostic",
                "session_id": session_id,
                "timestamp": diagnostic.timestamp,
                "diagnostic": _diagnostic_record(diagnostic),
            }
        )
    events.sort(key=lambda event: (event["timestamp"], event["event_id"]))
    return events


def _message_record_from_row(
    row: TraceRow,
    *,
    hide_thinking: bool,
) -> _MessageRecord | None:
    if row.turn_index is None:
        return None
    if row.kind == "thinking":
        if hide_thinking:
            return None
        return {
            "turn_index": row.turn_index,
            "run_id": row.run_id or "",
            "run_step": row.run_step or 0,
            "role": "assistant",
            "content": f"[thinking] {row.content}",
        }
    if row.kind == "tool_call":
        arguments = json.dumps(
            _metadata_mapping(row, "arguments"),
            ensure_ascii=False,
        )[:200]
        return {
            "turn_index": row.turn_index,
            "run_id": row.run_id or "",
            "run_step": row.run_step or 0,
            "role": "assistant",
            "content": f"[tool_call: {row.tool_name or row.title}({arguments})]",
        }
    if row.kind == "tool_result":
        return {
            "turn_index": row.turn_index,
            "run_id": row.run_id or "",
            "run_step": row.run_step or 0,
            "role": "tool_result",
            "tool": row.tool_name or "?",
            "is_error": row.is_error,
            "content": row.content[:500],
        }
    if row.kind == "control":
        if not row.is_error:
            return None
        return {
            "turn_index": row.turn_index,
            "run_id": row.run_id or "",
            "run_step": row.run_step or 0,
            "role": "error",
            "content": row.content,
        }
    if row.kind in {"system", "user", "trigger", "assistant", "error"}:
        role = row.kind
        if role == "trigger":
            role = "user"
        return {
            "turn_index": row.turn_index,
            "run_id": row.run_id or "",
            "run_step": row.run_step or 0,
            "role": role,
            "content": row.content,
        }
    return None


def _message_records_from_snapshot(
    snapshot: TraceSnapshot,
    *,
    hide_thinking: bool,
) -> list[_MessageRecord]:
    records: list[_MessageRecord] = []
    for row in snapshot.rows:
        record = _message_record_from_row(row, hide_thinking=hide_thinking)
        if record is not None:
            records.append(record)
    return records


trace_app = typer.Typer(
    name="trace",
    help="Query session trajectories.",
    invoke_without_command=True,
    add_completion=False,
)


@trace_app.callback()
def _trace_default(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    fmt: str | None = typer.Option(None, "--output", "-o", help="json for ndjson dump"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new content"),
) -> None:
    """Open the interactive viewer for the latest session when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    if follow and not sys.stdout.isatty():
        _follow_session(ctx, session)
        return
    if not follow and (fmt == "json" or not sys.stdout.isatty()):
        messages_cmd(
            ctx,
            session=session,
            role=None,
            hide_thinking=False,
            limit=None,
            fmt="ndjson",
        )
        return
    view_cmd(ctx, session=session, follow=follow)


def _get_query_store(ctx: typer.Context) -> TrajectoryQueryStore:
    state = ctx.obj
    if isinstance(state, _TraceContext):
        return state.query
    if state is not None:
        raise TypeError("agentm trace received an unexpected command context")

    resolved = resolve_trajectory_store()
    if resolved is not None:
        query = TrajectoryStoreQueryAdapter(resolved.store)
        ctx.obj = _TraceContext(query=query)
        ctx.call_on_close(resolved.close)
        return query
    stderr_console.print(
        "[red]error: no trajectory store found[/red]\n"
        "[dim]Set AGENTM_TRAJECTORY_DSN for Postgres, "
        "AGENTM_TRAJECTORY_DIR for JSONL, or run from a project "
        "with .agentm/trajectory.[/dim]"
    )
    raise typer.Exit(EXIT_NOT_FOUND)


def _resolve_session_id(
    query: TrajectoryQueryStore,
    session: str | None,
) -> str:
    if session:
        return session
    metas = list(query.sessions())
    if not metas:
        stderr_console.print("[red]error: no sessions in store[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    return max(metas, key=lambda item: item.created_at).id


# -- follow ----------------------------------------------------------------


def _follow_print(
    role: str,
    label: str,
    content: str,
    *,
    dim: bool = False,
) -> None:
    color = _ROLE_ANSI.get(role, "")
    sys.stdout.write(f"{color}── {label} ──{_ANSI_RESET}\n")
    if dim:
        sys.stdout.write(f"{_ANSI_DIM}{content}{_ANSI_RESET}\n\n")
    else:
        sys.stdout.write(f"{content}\n\n")
    sys.stdout.flush()


def _follow_print_trigger(
    record: Turn | TurnCheckpoint,
) -> None:
    if not isinstance(record.trigger, UserInput):
        return
    parts: list[str] = []
    for block in record.trigger.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append(f"[image {block.mime_type}]")
        else:
            parts.append(f"[{type(block).__name__}]")
    _follow_print("user", f"USER  turn={record.index}", "\n".join(parts))


def _follow_print_turn_payload(record: Turn | TurnCheckpoint) -> None:
    # assistant response
    text_parts: list[str] = []
    if record.response is not None:
        for block in record.response.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingBlock):
                text_parts.append(f"[thinking] {block.text[:200]}")
            elif isinstance(block, OpaqueThinkingBlock):
                text_parts.append(f"[thinking: {block.provider}]")
            elif isinstance(block, ToolCallBlock):
                args = json.dumps(dict(block.arguments), ensure_ascii=False)[:120]
                text_parts.append(f"[call: {block.name}({args})]")
    if text_parts:
        _follow_print(
            "assistant",
            f"ASSISTANT  turn={record.index}",
            "\n".join(text_parts),
        )

    # tool results
    for rec in record.tool_results:
        txt = "".join(b.text for b in rec.result.content if isinstance(b, TextContent))
        label = f"RESULT: {rec.call.name}  turn={record.index}"
        if rec.result.is_error:
            label += " [ERROR]"
        preview = txt[:500]
        if len(txt) > 500:
            preview += f"\n... ({len(txt) - 500} chars truncated)"
        _follow_print(
            "tool_result" if not rec.result.is_error else "error",
            label,
            preview,
            dim=not rec.result.is_error,
        )


def _follow_print_commit(turn: Turn) -> None:
    cause = type(turn.outcome.cause).__name__
    inp = turn.meta.total_input_tokens
    out = turn.meta.total_output_tokens
    sys.stdout.write(
        f"{_ANSI_DIM}── committed: {cause}  in:{inp:,} out:{out:,} ──{_ANSI_RESET}\n\n"
    )
    sys.stdout.flush()


def _follow_session(
    ctx: typer.Context,
    session: str | None,
) -> None:
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)

    shown_turn_ids: set[str] = set()
    checkpoint_id: str | None = None
    checkpoint_payload_shown = False

    stderr_console.print(f"[dim]following {sid} (Ctrl+C to stop)[/dim]")

    try:
        while True:
            try:
                turns = list(query.turns(sid))
            except KeyError:
                stderr_console.print(f"[red]error: session not found: {sid}[/red]")
                raise typer.Exit(EXIT_NOT_FOUND)

            for turn in turns:
                if turn.id in shown_turn_ids:
                    continue
                shown_turn_ids.add(turn.id)

                if turn.id == checkpoint_id:
                    if not checkpoint_payload_shown:
                        _follow_print_turn_payload(turn)
                else:
                    _follow_print_trigger(turn)
                    _follow_print_turn_payload(turn)

                _follow_print_commit(turn)
                checkpoint_id = None
                checkpoint_payload_shown = False

            checkpoints = list(query.checkpoints(sid))
            if checkpoints:
                cp = checkpoints[0]
                if cp.id != checkpoint_id:
                    checkpoint_id = cp.id
                    checkpoint_payload_shown = False
                    _follow_print_trigger(cp)

                if cp.response is not None and not checkpoint_payload_shown:
                    _follow_print_turn_payload(cp)
                    checkpoint_payload_shown = True

            time.sleep(1)
    except KeyboardInterrupt:
        stderr_console.print("\n[dim]stopped[/dim]")


def _resolve_format(fmt: str | None) -> TraceFormat:
    if fmt == "text":
        return "text"
    if fmt == "ndjson":
        return "ndjson"
    if fmt is not None:
        raise typer.BadParameter(
            "format must be 'text' or 'ndjson'",
            param_hint="--format",
        )
    return "text" if is_tty() else "ndjson"


def _emit_json(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _emit_records(
    records: Iterable[RecordT],
    fmt: TraceFormat,
    render_fn: Callable[[RecordT], str],
    limit: int | None,
) -> int:
    count = 0
    for record in records:
        if limit is not None and count >= limit:
            break
        if fmt == "ndjson":
            _emit_json(record)
        else:
            sys.stdout.write(render_fn(record) + "\n")
        count += 1
    return count


# -- live status -------------------------------------------------------------


def _render_status(status: _StatusRecord) -> str:
    checkpoint = (
        f"turn={status['checkpoint_turn_index']} id={status['checkpoint_turn_id']}"
        if status["active_checkpoint"]
        else "---"
    )
    return (
        f"session:             {status['session_id']}\n"
        f"state:               {status['state']}\n"
        f"committed turns:     {status['committed_turns']}\n"
        f"incomplete turns:    {status['incomplete_turns']}\n"
        f"active checkpoint:   {checkpoint}\n"
        f"diagnostics:         {status['diagnostic_count']}"
    )


def _emit_status(status: _StatusRecord, fmt: TraceFormat) -> None:
    if fmt == "ndjson":
        _emit_json(status)
    else:
        sys.stdout.write(_render_status(status) + "\n")


def _validate_poll_options(
    *,
    timeout: float | None,
    poll_interval: float,
) -> None:
    if timeout is not None and timeout < 0:
        raise typer.BadParameter("timeout must be non-negative", param_hint="--timeout")
    if poll_interval <= 0:
        raise typer.BadParameter(
            "poll interval must be positive",
            param_hint="--poll-interval",
        )


@trace_app.command("status")
def status_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print one scriptable snapshot of session trajectory progress."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)
    try:
        status = _load_status_record(query, sid)
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    _emit_status(status, _resolve_format(fmt))


@trace_app.command("wait")
def wait_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    min_committed_turns: int | None = typer.Option(
        None,
        "--min-committed-turns",
    ),
    require_active_checkpoint: bool = typer.Option(
        False,
        "--require-active-checkpoint",
    ),
    timeout: float = typer.Option(120.0, "--timeout"),
    poll_interval: float = typer.Option(0.25, "--poll-interval"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Wait until typed trajectory progress conditions are satisfied."""
    _validate_poll_options(timeout=timeout, poll_interval=poll_interval)
    if min_committed_turns is not None and min_committed_turns < 0:
        raise typer.BadParameter(
            "minimum committed turns must be non-negative",
            param_hint="--min-committed-turns",
        )
    if min_committed_turns is None and not require_active_checkpoint:
        raise typer.BadParameter(
            "provide --min-committed-turns and/or --require-active-checkpoint"
        )
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)
    deadline = time.monotonic() + timeout
    chosen_fmt = _resolve_format(fmt)
    while True:
        try:
            status = _load_status_record(query, sid)
        except KeyError:
            stderr_console.print(f"[red]error: session not found: {sid}[/red]")
            raise typer.Exit(EXIT_NOT_FOUND)
        enough_turns = (
            min_committed_turns is None
            or status["committed_turns"] >= min_committed_turns
        )
        checkpoint_ready = not require_active_checkpoint or status["active_checkpoint"]
        if enough_turns and checkpoint_ready:
            _emit_status(status, chosen_fmt)
            return
        if time.monotonic() >= deadline:
            _emit_status(status, chosen_fmt)
            stderr_console.print(
                f"[red]error: trace wait timed out after {timeout:g}s[/red]"
            )
            raise typer.Exit(EXIT_TIMEOUT)
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


@trace_app.command("watch")
def watch_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    include_existing: bool = typer.Option(
        False,
        "--include-existing",
        help="Emit current trajectory records before watching for new deltas.",
    ),
    timeout: float | None = typer.Option(None, "--timeout"),
    poll_interval: float = typer.Option(0.25, "--poll-interval"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Stream checkpoint, commit, abort, and diagnostic deltas."""
    _validate_poll_options(timeout=timeout, poll_interval=poll_interval)
    if limit is not None and limit < 0:
        raise typer.BadParameter("limit must be non-negative", param_hint="--limit")
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)
    chosen_fmt = _resolve_format(fmt)
    seen: set[str] = set()
    try:
        initial = _watch_events(query, sid)
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    if not include_existing:
        seen.update(event["event_id"] for event in initial)
    deadline = time.monotonic() + timeout if timeout is not None else None
    emitted = 0
    try:
        while limit is None or emitted < limit:
            events = _watch_events(query, sid)
            for event in events:
                if event["event_id"] in seen:
                    continue
                seen.add(event["event_id"])
                if chosen_fmt == "ndjson":
                    _emit_json(event)
                else:
                    sys.stdout.write(
                        f"{event['timestamp']:.3f} {event['type']:<10} "
                        f"turn={event.get('turn_index', '-')} "
                        f"id={event.get('turn_id', event['event_id'])}\n"
                    )
                    sys.stdout.flush()
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
            if deadline is not None and time.monotonic() >= deadline:
                return
            sleep_for = poll_interval
            if deadline is not None:
                sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        stderr_console.print("\n[dim]stopped[/dim]")


# -- diagnostics ------------------------------------------------------------


@trace_app.command("diagnostics")
def diagnostics_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    level: str | None = typer.Option(None, "--level"),
    phase: str | None = typer.Option(None, "--phase"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print durable structured session diagnostics."""
    if level is not None and level not in {"info", "warning", "error"}:
        raise typer.BadParameter(
            "level must be 'info', 'warning', or 'error'",
            param_hint="--level",
        )
    if limit is not None and limit < 0:
        raise typer.BadParameter("limit must be non-negative", param_hint="--limit")
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)
    try:
        records = [
            _diagnostic_record(diagnostic)
            for diagnostic in query.diagnostics(sid)
            if (level is None or diagnostic.level == level)
            and (phase is None or diagnostic.phase == phase)
        ]
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)

    def _render(record: _DiagnosticRecord) -> str:
        location = f"turn={record['turn_index']} checkpoint={record['checkpoint_id']}"
        error = ""
        if record["error_type"] is not None:
            error = f"\n  {record['error_type']}: {record['error_detail']}"
        return (
            f"{record['timestamp']:.3f} {record['level']} "
            f"phase={record['phase']} {location}\n"
            f"  {record['message']}{error}"
        )

    _emit_records(iter(records), chosen_fmt, _render, limit)


# -- sessions ----------------------------------------------------------------


@trace_app.command("sessions")
def sessions_cmd(
    ctx: typer.Context,
    purpose: str | None = typer.Option(None, "--purpose"),
    parent: str | None = typer.Option(None, "--parent"),
    active: bool = typer.Option(
        False,
        "--active",
        help="Only sessions with a current incomplete checkpoint.",
    ),
    latest: bool = typer.Option(
        False,
        "--latest",
        help="Return only the newest matching session.",
    ),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """List sessions in the trajectory store."""
    if limit is not None and limit < 0:
        raise typer.BadParameter("limit must be non-negative", param_hint="--limit")
    query = _get_query_store(ctx)
    rows = list(
        query.sessions(
            SessionFilter(
                parent_session_id=parent,
                purpose=purpose,
            )
        )
    )
    rows_with_checkpoints = [
        (row, bool(list(query.checkpoints(row.id)))) for row in rows
    ]
    if active:
        rows_with_checkpoints = [item for item in rows_with_checkpoints if item[1]]
    if latest and rows_with_checkpoints:
        rows_with_checkpoints = [
            max(rows_with_checkpoints, key=lambda item: item[0].created_at)
        ]
    if limit is not None:
        rows_with_checkpoints = rows_with_checkpoints[:limit]
    chosen_fmt = _resolve_format(fmt)

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(rows_with_checkpoints)} session(s)[/dim]")
        for row, has_checkpoint in rows_with_checkpoints:
            parent_id = row.parent_session_id or "---"
            sys.stdout.write(
                f"  {row.id:<20} purpose={row.purpose:<8} "
                f"active={str(has_checkpoint).lower():<5} parent={parent_id}\n"
            )
    else:
        for row, has_checkpoint in rows_with_checkpoints:
            _emit_json(
                {
                    "id": row.id,
                    "parent_session_id": row.parent_session_id,
                    "root_session_id": row.root_session_id,
                    "purpose": row.purpose,
                    "cwd": row.cwd,
                    "created_at": row.created_at,
                    "active_checkpoint": has_checkpoint,
                }
            )


# -- turns -------------------------------------------------------------------


def _turn_summary_record(summary: TraceTurnSummary) -> _TurnSummary:
    record: _TurnSummary = {
        "status": summary.status,
        "turn_index": summary.turn_index,
        "turn_id": summary.turn_id,
        "run_id": summary.run_id,
        "run_step": summary.run_step,
        "trigger_source": summary.trigger_source,
        "tool_calls": list(summary.tool_names),
        "tool_call_count": summary.tool_calls,
        "tool_error_count": summary.tool_errors,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "cache_read": summary.cache_read_tokens,
        "model": summary.model,
        "cause": summary.cause,
    }
    if summary.error_type is not None:
        record["error_type"] = summary.error_type
        record["error"] = summary.error or ""
    return record


@trace_app.command("turns")
def turns_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    status: str | None = typer.Option(None, "--status"),
    run_id: str | None = typer.Option(None, "--run-id"),
    from_turn: int | None = typer.Option(None, "--from-turn"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print per-turn summaries for a session."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)
    try:
        snapshot = _load_trace_snapshot(query, sid)
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    summaries = [_turn_summary_record(summary) for summary in snapshot.turns]
    if status is not None:
        if status not in {"committed", "incomplete"}:
            raise typer.BadParameter(
                "status must be 'committed' or 'incomplete'",
                param_hint="--status",
            )
        summaries = [summary for summary in summaries if summary["status"] == status]
    if run_id is not None:
        summaries = [summary for summary in summaries if summary["run_id"] == run_id]
    if from_turn is not None:
        if from_turn < 0:
            raise typer.BadParameter(
                "from turn must be non-negative",
                param_hint="--from-turn",
            )
        summaries = [
            summary for summary in summaries if summary["turn_index"] >= from_turn
        ]

    def _render(summary: _TurnSummary) -> str:
        tools = ", ".join(summary["tool_calls"]) if summary["tool_calls"] else "---"
        state = summary["cause"] if summary["cause"] is not None else summary["status"]
        rendered = (
            f"  [{summary['turn_index']}] {state:<20} tools=[{tools}] "
            f"in={summary['input_tokens']} out={summary['output_tokens']}"
        )
        if "error_type" in summary:
            rendered += (
                f"\n      error={summary['error_type']}: {summary.get('error', '')}"
            )
        return rendered

    if chosen_fmt == "text":
        stderr_console.print(
            f"[dim]session {sid}: {snapshot.metrics.committed_turns} committed, "
            f"{snapshot.metrics.incomplete_turns} incomplete[/dim]"
        )
    _emit_records(iter(summaries), chosen_fmt, _render, limit)


# -- messages ----------------------------------------------------------------


@trace_app.command("messages")
def messages_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    role: str | None = typer.Option(None, "--role"),
    hide_thinking: bool = typer.Option(False, "--hide-thinking"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print the conversation messages for a session."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)
    try:
        snapshot = _load_trace_snapshot(query, sid)
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    all_msgs = _message_records_from_snapshot(
        snapshot,
        hide_thinking=hide_thinking,
    )
    if role:
        all_msgs = [m for m in all_msgs if m["role"] == role]

    def _render(message: _MessageRecord) -> str:
        role = message["role"]
        color = _ROLE_ANSI.get(role, "")
        header = f"{color}── {role.upper()} ── turn={message['turn_index']}"
        if role == "tool_result":
            error = " ERROR" if message.get("is_error") else ""
            header += f" tool={message.get('tool', '?')}{error}"
        header += f" {'─' * 20}{_ANSI_RESET}"
        content = message["content"]
        if role == "tool_result" and not message.get("is_error"):
            content = f"{_ANSI_DIM}{content}{_ANSI_RESET}"
        elif role == "error":
            content = f"{_ANSI_RED}{content}{_ANSI_RESET}"
        return f"{header}\n{content}\n"

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(all_msgs)} message(s)[/dim]")
    _emit_records(iter(all_msgs), chosen_fmt, _render, limit)


# -- usage -------------------------------------------------------------------


@trace_app.command("usage")
def usage_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Token usage summary for a session."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)
    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    if not turns:
        stderr_console.print("[dim]no turns[/dim]")
        return
    total_in = sum(t.meta.total_input_tokens for t in turns)
    total_out = sum(t.meta.total_output_tokens for t in turns)
    cache_read = sum(t.meta.cache_read_tokens for t in turns)
    cache_write = sum(t.meta.cache_write_tokens for t in turns)
    hit_pct = (cache_read / total_in * 100) if total_in else 0.0
    summary = {
        "session_id": sid,
        "turns": len(turns),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "non_cached_input": total_in - cache_read,
        "cache_hit_rate": round(hit_pct, 1),
        "total_tokens": total_in + total_out,
    }
    chosen_fmt = _resolve_format(fmt)
    if chosen_fmt == "text":
        sys.stdout.write(
            f"session:          {sid}\nturns:            {summary['turns']}\ninput tokens:     {total_in:>12,}\n  cache read:     {cache_read:>12,}  ({hit_pct:.1f}%)\n  cache write:    {cache_write:>12,}\n  non-cached:     {total_in - cache_read:>12,}\noutput tokens:    {total_out:>12,}\ntotal tokens:     {total_in + total_out:>12,}\n"
        )
    else:
        _emit_json(summary)


# -- view (interactive) ------------------------------------------------------


@trace_app.command("view")
def view_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    follow: bool = typer.Option(False, "--follow", "-f"),
    legacy: bool = typer.Option(False, "--legacy", help="Use the old ANSI pager."),
) -> None:
    """Interactive trace viewer with turn navigation and expand/collapse."""
    if not sys.stdout.isatty():
        stderr_console.print("[red]error: interactive viewer requires a terminal[/red]")
        raise typer.Exit(2)

    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)

    if not legacy:
        try:
            from agentm.presenter.trajectory import run_textual_viewer
        except ImportError:
            stderr_console.print(
                "[yellow]warning: Textual viewer unavailable; "
                "falling back to legacy pager[/yellow]"
            )
        else:
            run_textual_viewer(query, sid, follow=follow)
            return

    from agentm.cli._trace_viewer import run_interactive_viewer

    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    checkpoints = list(query.checkpoints(sid))

    reload = None
    if follow:

        def reload() -> tuple[list[Turn], list[TurnCheckpoint]]:
            try:
                t = list(query.turns(sid))
            except KeyError:
                t = []
            return t, list(query.checkpoints(sid))

    run_interactive_viewer(turns, sid, checkpoints=checkpoints, reload=reload)


# -- tools -------------------------------------------------------------------


def _metadata_string(row: TraceRow, key: str) -> str | None:
    value = row.metadata.get(key)
    return value if isinstance(value, str) else None


def _metadata_mapping(row: TraceRow, key: str) -> dict[str, object]:
    value = row.metadata.get(key)
    if not isinstance(value, Mapping):
        return {}
    return {str(item_key): item_value for item_key, item_value in value.items()}


def _tool_records_from_snapshot(
    snapshot: TraceSnapshot,
    *,
    tool: str | None,
    result_chars: int | None,
) -> list[_ToolRecord]:
    args_by_call_id: dict[str, dict[str, object]] = {}
    for row in snapshot.rows:
        if row.kind != "tool_call":
            continue
        call_id = _metadata_string(row, "tool_call_id")
        if call_id is not None:
            args_by_call_id[call_id] = _metadata_mapping(row, "arguments")

    records: list[_ToolRecord] = []
    for row in snapshot.rows:
        if row.kind != "tool_result" or row.tool_name is None:
            continue
        if tool and row.tool_name != tool:
            continue
        call_id = _metadata_string(row, "tool_call_id")
        full_result = row.content
        result = full_result[:result_chars] if result_chars is not None else full_result
        records.append(
            {
                "turn_index": row.turn_index or 0,
                "run_id": row.run_id or "",
                "run_step": row.run_step or 0,
                "tool": row.tool_name,
                "args": args_by_call_id.get(call_id or "", {}),
                "is_error": row.is_error,
                "result": result,
                "result_chars": len(full_result),
                "result_truncated": len(result) < len(full_result),
            }
        )
    return records


@trace_app.command("tools")
def tools_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    tool: str | None = typer.Option(None, "--tool"),
    result_chars: int | None = typer.Option(None, "--result-chars"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print tool calls with arguments and results."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session)
    try:
        snapshot = _load_trace_snapshot(query, sid)
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    if result_chars is not None and result_chars < 0:
        raise typer.BadParameter(
            "result chars must be non-negative",
            param_hint="--result-chars",
        )
    records = _tool_records_from_snapshot(
        snapshot,
        tool=tool,
        result_chars=result_chars,
    )

    def _render(record: _ToolRecord) -> str:
        arguments = json.dumps(record["args"], ensure_ascii=False, indent=2)[:600]
        result = record["result"]
        if record["result_truncated"]:
            truncated_chars = record["result_chars"] - len(record["result"])
            result += f"\n... ({truncated_chars} chars truncated)"
        error_tag = (
            f" {_ROLE_ANSI['error']}ERROR{_ANSI_RESET}" if record["is_error"] else ""
        )
        tool_color = _ROLE_ANSI["tool_call"]
        header = (
            f"{tool_color}── {record['tool']}{error_tag}{tool_color} ── "
            f"turn={record['turn_index']} {'─' * 10}{_ANSI_RESET}"
        )
        result_styled = (
            f"{_ROLE_ANSI['error']}{result}{_ANSI_RESET}"
            if record["is_error"]
            else f"{_ANSI_DIM}{result}{_ANSI_RESET}"
        )
        return f"{header}\n  args:\n{arguments}\n  result:\n{result_styled}\n"

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(records)} tool call(s)[/dim]")
    _emit_records(iter(records), chosen_fmt, _render, limit)
