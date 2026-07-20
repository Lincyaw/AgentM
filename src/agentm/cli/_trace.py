"""``agentm trace`` -- query trajectories via the SDK store layer."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, NotRequired, TypeVar, TypedDict

import typer

from agentm.cli._display import EXIT_NOT_FOUND, is_tty, stderr_console
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
from agentm.core.abi.termination import ProviderRequestFailed
from agentm.core.abi.trajectory import Turn, TurnCheckpoint
from agentm.core.abi.trigger import UserInput
from agentm.core.runtime.stores.query import TrajectoryStoreQueryAdapter

TraceFormat = Literal["text", "ndjson"]
RecordT = TypeVar("RecordT")


class _TurnSummary(TypedDict):
    status: Literal["committed", "incomplete"]
    turn_index: int
    turn_id: str
    trigger_source: str
    rounds: int
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
    round_index: int | None
    role: str
    content: str
    tool: NotRequired[str]
    is_error: NotRequired[bool]


class _ToolRecord(TypedDict):
    turn_index: int
    round_index: int
    tool: str
    args: dict[str, object]
    is_error: bool
    result: str


@dataclass(frozen=True, slots=True)
class _TraceContext:
    query: TrajectoryQueryStore


trace_app = typer.Typer(
    name="trace",
    help="Query session trajectories.",
    no_args_is_help=True,
    add_completion=False,
)


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
    latest: bool,
) -> str:
    if session and latest:
        stderr_console.print(
            "[red]error: --session and --latest are mutually exclusive[/red]"
        )
        raise typer.Exit(2)
    if session:
        return session
    if latest:
        metas = list(query.sessions())
        if not metas:
            stderr_console.print("[red]error: no sessions in store[/red]")
            raise typer.Exit(EXIT_NOT_FOUND)
        return max(metas, key=lambda item: item.created_at).id
    stderr_console.print("[red]error: must specify --session or --latest[/red]")
    raise typer.Exit(2)


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


# -- sessions ----------------------------------------------------------------


@trace_app.command("sessions")
def sessions_cmd(
    ctx: typer.Context,
    purpose: str | None = typer.Option(None, "--purpose"),
    parent: str | None = typer.Option(None, "--parent"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """List sessions in the trajectory store."""
    query = _get_query_store(ctx)
    rows = list(
        query.sessions(
            SessionFilter(
                parent_session_id=parent,
                purpose=purpose,
                limit=limit,
            )
        )
    )
    chosen_fmt = _resolve_format(fmt)

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(rows)} session(s)[/dim]")
        for row in rows:
            parent_id = row.parent_session_id or "---"
            sys.stdout.write(
                f"  {row.id:<20} purpose={row.purpose:<8} parent={parent_id}\n"
            )
    else:
        for row in rows:
            _emit_json(
                {
                    "id": row.id,
                    "parent_session_id": row.parent_session_id,
                    "root_session_id": row.root_session_id,
                    "purpose": row.purpose,
                    "cwd": row.cwd,
                    "created_at": row.created_at,
                }
            )


# -- turns -------------------------------------------------------------------


def _turn_summary(turn: Turn) -> _TurnSummary:
    tool_names: list[str] = []
    tool_errors = 0
    for rnd in turn.rounds:
        for rec in rnd.tool_results:
            tool_names.append(rec.call.name)
            if rec.result.is_error:
                tool_errors += 1
    summary: _TurnSummary = {
        "status": "committed",
        "turn_index": turn.index,
        "turn_id": turn.id,
        "trigger_source": turn.trigger.source,
        "rounds": len(turn.rounds),
        "tool_calls": tool_names,
        "tool_call_count": len(tool_names),
        "tool_error_count": tool_errors,
        "input_tokens": turn.meta.total_input_tokens,
        "output_tokens": turn.meta.total_output_tokens,
        "cache_read": turn.meta.cache_read_tokens,
        "model": turn.meta.model_id,
        "cause": type(turn.outcome.cause).__name__,
    }
    if isinstance(turn.outcome.cause, ProviderRequestFailed):
        summary["error_type"] = turn.outcome.cause.error_type
        summary["error"] = turn.outcome.cause.detail
    return summary


def _checkpoint_summary(checkpoint: TurnCheckpoint) -> _TurnSummary:
    tool_names = [
        record.call.name
        for round_ in checkpoint.rounds
        for record in round_.tool_results
    ]
    return {
        "status": "incomplete",
        "turn_index": checkpoint.index,
        "turn_id": checkpoint.id,
        "trigger_source": checkpoint.trigger.source,
        "rounds": len(checkpoint.rounds),
        "tool_calls": tool_names,
        "tool_call_count": len(tool_names),
        "tool_error_count": sum(
            1
            for round_ in checkpoint.rounds
            for record in round_.tool_results
            if record.result.is_error
        ),
        "input_tokens": checkpoint.meta.total_input_tokens,
        "output_tokens": checkpoint.meta.total_output_tokens,
        "cache_read": checkpoint.meta.cache_read_tokens,
        "model": checkpoint.meta.model_id,
        "cause": None,
    }


@trace_app.command("turns")
def turns_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print per-turn summaries for a session."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session, latest)
    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    checkpoints = list(query.checkpoints(sid))
    summaries = [
        *(_turn_summary(turn) for turn in turns),
        *(_checkpoint_summary(checkpoint) for checkpoint in checkpoints),
    ]
    summaries.sort(key=lambda item: item["turn_index"])

    def _render(d: _TurnSummary) -> str:
        tools = ", ".join(d["tool_calls"]) if d["tool_calls"] else "---"
        state = d["cause"] if d["cause"] is not None else d["status"]
        rendered = (
            f"  [{d['turn_index']}] {state:<20} tools=[{tools}] "
            f"in={d['input_tokens']} out={d['output_tokens']}"
        )
        if "error_type" in d:
            rendered += f"\n      error={d['error_type']}: {d.get('error', '')}"
        return rendered

    if chosen_fmt == "text":
        stderr_console.print(
            f"[dim]session {sid}: {len(turns)} committed, "
            f"{len(checkpoints)} incomplete[/dim]"
        )
    _emit_records(iter(summaries), chosen_fmt, _render, limit)


# -- messages ----------------------------------------------------------------


@trace_app.command("messages")
def messages_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    role: str | None = typer.Option(None, "--role"),
    hide_thinking: bool = typer.Option(False, "--hide-thinking"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print the conversation messages for a session."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session, latest)
    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    all_msgs: list[_MessageRecord] = []
    records: list[Turn | TurnCheckpoint] = [
        *turns,
        *query.checkpoints(sid),
    ]
    records.sort(key=lambda item: item.index)
    _shown_system_hash: str | None = None
    for turn_record in records:
        sp = turn_record.meta.system_prompt
        if sp is not None:
            sp_hash = hashlib.sha256(sp.encode("utf-8")).hexdigest()[:16]
            if sp_hash != _shown_system_hash:
                _shown_system_hash = sp_hash
                all_msgs.append(
                    {
                        "turn_index": turn_record.index,
                        "round_index": None,
                        "role": "system",
                        "content": sp,
                    }
                )
        if isinstance(turn_record.trigger, UserInput):
            trigger_content = [
                block.text
                if isinstance(block, TextContent)
                else (
                    f"[image {block.mime_type}, {len(block.data)} bytes]"
                    if isinstance(block, ImageContent)
                    else f"[{type(block).__name__}]"
                )
                for block in turn_record.trigger.content
            ]
            all_msgs.append(
                {
                    "turn_index": turn_record.index,
                    "round_index": None,
                    "role": "user",
                    "content": "\n".join(trigger_content),
                }
            )
        for ri, rnd in enumerate(turn_record.rounds):
            blocks: list[str] = []
            for block in rnd.response.content:
                if isinstance(block, TextContent):
                    blocks.append(block.text)
                elif isinstance(block, ThinkingBlock) and not hide_thinking:
                    blocks.append(f"[thinking] {block.text}")
                elif isinstance(block, OpaqueThinkingBlock) and not hide_thinking:
                    blocks.append(f"[opaque thinking: {block.provider}]")
                elif isinstance(block, ToolCallBlock):
                    arguments = json.dumps(
                        dict(block.arguments),
                        ensure_ascii=False,
                    )[:200]
                    blocks.append(f"[tool_call: {block.name}({arguments})]")
            all_msgs.append(
                {
                    "turn_index": turn_record.index,
                    "round_index": ri,
                    "role": "assistant",
                    "content": "\n".join(blocks),
                }
            )
            for rec in rnd.tool_results:
                txt = "".join(
                    block.text
                    for block in rec.result.content
                    if isinstance(block, TextContent)
                )[:500]
                all_msgs.append(
                    {
                        "turn_index": turn_record.index,
                        "round_index": ri,
                        "role": "tool_result",
                        "tool": rec.call.name,
                        "is_error": rec.result.is_error,
                        "content": txt,
                    }
                )
        if isinstance(turn_record, Turn) and isinstance(
            turn_record.outcome.cause,
            ProviderRequestFailed,
        ):
            all_msgs.append(
                {
                    "turn_index": turn_record.index,
                    "round_index": None,
                    "role": "error",
                    "content": (
                        f"{turn_record.outcome.cause.error_type}: "
                        f"{turn_record.outcome.cause.detail}"
                    ),
                }
            )
    if role:
        all_msgs = [m for m in all_msgs if m["role"] == role]

    _ROLE_ANSI = {
        "system": "\033[1;35m",  # bold magenta
        "user": "\033[1;32m",  # bold green
        "assistant": "\033[1;34m",  # bold blue
        "tool_result": "\033[36m",  # cyan
        "error": "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"
    _DIM = "\033[2m"

    def _render(m: _MessageRecord) -> str:
        round_label = str(m["round_index"]) if m["round_index"] is not None else "---"
        role = m["role"]
        color = _ROLE_ANSI.get(role, "")
        hdr = f"{color}── {role.upper()} ── turn={m['turn_index']} round={round_label}"
        if role == "tool_result":
            error = " ERROR" if m.get("is_error") else ""
            hdr += f" tool={m.get('tool', '?')}{error}"
        hdr += f" {'─' * 20}{_RESET}"
        content = m["content"]
        if role == "tool_result" and not m.get("is_error"):
            content = f"{_DIM}{content}{_RESET}"
        elif role == "error":
            content = f"\033[31m{content}{_RESET}"
        return f"{hdr}\n{content}\n"

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(all_msgs)} message(s)[/dim]")
    _emit_records(iter(all_msgs), chosen_fmt, _render, limit)


# -- usage -------------------------------------------------------------------


@trace_app.command("usage")
def usage_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Token usage summary for a session."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session, latest)
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
    latest: bool = typer.Option(False, "--latest"),
) -> None:
    """Interactive trace viewer with turn navigation and expand/collapse."""
    from agentm.cli._trace_viewer import run_interactive_viewer

    if not sys.stdout.isatty():
        stderr_console.print("[red]error: interactive viewer requires a terminal[/red]")
        raise typer.Exit(2)

    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session, latest)
    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)

    run_interactive_viewer(turns, sid)


# -- tools -------------------------------------------------------------------


@trace_app.command("tools")
def tools_cmd(
    ctx: typer.Context,
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    tool: str | None = typer.Option(None, "--tool"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print tool calls with arguments and results."""
    query = _get_query_store(ctx)
    sid = _resolve_session_id(query, session, latest)
    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    records: list[_ToolRecord] = []
    trajectory_records: list[Turn | TurnCheckpoint] = [
        *turns,
        *query.checkpoints(sid),
    ]
    trajectory_records.sort(key=lambda item: item.index)
    for turn_record in trajectory_records:
        for ri, rnd in enumerate(turn_record.rounds):
            for rec in rnd.tool_results:
                name = rec.call.name
                if tool and name != tool:
                    continue
                txt = "".join(
                    block.text
                    for block in rec.result.content
                    if isinstance(block, TextContent)
                )
                records.append(
                    {
                        "turn_index": turn_record.index,
                        "round_index": ri,
                        "tool": name,
                        "args": dict(rec.call.arguments),
                        "is_error": rec.result.is_error,
                        "result": txt,
                    }
                )

    _T_YELLOW = "\033[1;33m"
    _T_RED = "\033[1;31m"
    _T_DIM = "\033[2m"
    _T_RESET = "\033[0m"

    def _render(d: _ToolRecord) -> str:
        a = json.dumps(d["args"], ensure_ascii=False, indent=2)[:600]
        r = d["result"][:800]
        error_tag = f" {_T_RED}ERROR{_T_RESET}" if d["is_error"] else ""
        hdr = (
            f"{_T_YELLOW}── {d['tool']}{error_tag}{_T_YELLOW} ── "
            f"turn={d['turn_index']} round={d['round_index']} {'─' * 10}{_T_RESET}"
        )
        result_styled = (
            f"{_T_RED}{r}{_T_RESET}" if d["is_error"] else f"{_T_DIM}{r}{_T_RESET}"
        )
        return f"{hdr}\n  args:\n{a}\n  result:\n{result_styled}\n"

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(records)} tool call(s)[/dim]")
    _emit_records(iter(records), chosen_fmt, _render, limit)
