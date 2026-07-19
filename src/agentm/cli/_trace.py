"""``agentm trace`` -- query trajectories via the SDK store layer."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from typing import Literal, NotRequired, TypeVar, TypedDict

import typer

from agentm.cli._display import EXIT_NOT_FOUND, is_tty, stderr_console
from agentm.cli._store import resolve_trajectory_store
from agentm.core.abi.messages import (
    ImageContent,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
)
from agentm.core.abi.query import (
    SessionFilter,
    TrajectoryQueryStore,
)
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.termination import ProviderRequestFailed
from agentm.core.abi.trajectory import Turn
from agentm.core.abi.trigger import UserInput
from agentm.core.runtime.stores.query import TrajectoryStoreQueryAdapter

TraceFormat = Literal["text", "ndjson"]
RecordT = TypeVar("RecordT")


class _TurnSummary(TypedDict):
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
    cause: str
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

trace_app = typer.Typer(
    name="trace",
    help="Query session trajectories.",
    no_args_is_help=True,
    add_completion=False,
)


def _get_store() -> TrajectoryStore:
    store = resolve_trajectory_store()
    if store is not None:
        return store
    stderr_console.print(
        "[red]error: no trajectory store found[/red]\n"
        "[dim]Set AGENTM_TRAJECTORY_DSN for Postgres, "
        "AGENTM_TRAJECTORY_DIR for JSONL, or run from a project "
        "with .agentm/trajectory.[/dim]"
    )
    raise typer.Exit(EXIT_NOT_FOUND)


def _get_query_store() -> TrajectoryQueryStore:
    return TrajectoryStoreQueryAdapter(_get_store())


def _resolve_session_id(
    query: TrajectoryQueryStore,
    session: str | None,
    latest: bool,
) -> str:
    if session and latest:
        stderr_console.print("[red]error: --session and --latest are mutually exclusive[/red]")
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
    purpose: str | None = typer.Option(None, "--purpose"),
    parent: str | None = typer.Option(None, "--parent"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """List sessions in the trajectory store."""
    query = _get_query_store()
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
                f"  {row.id:<20} purpose={row.purpose:<8} "
                f"parent={parent_id}\n"
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
        "turn_index": turn.index, "turn_id": turn.id,
        "trigger_source": turn.trigger.source,
        "rounds": len(turn.rounds), "tool_calls": tool_names,
        "tool_call_count": len(tool_names), "tool_error_count": tool_errors,
        "input_tokens": turn.meta.total_input_tokens, "output_tokens": turn.meta.total_output_tokens,
        "cache_read": turn.meta.cache_read_tokens, "model": turn.meta.model_id,
        "cause": type(turn.outcome.cause).__name__,
    }
    if isinstance(turn.outcome.cause, ProviderRequestFailed):
        summary["error_type"] = turn.outcome.cause.error_type
        summary["error"] = turn.outcome.cause.detail
    return summary


@trace_app.command("turns")
def turns_cmd(
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print per-turn summaries for a session."""
    query = _get_query_store()
    sid = _resolve_session_id(query, session, latest)
    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    summaries = [_turn_summary(t) for t in turns]

    def _render(d: _TurnSummary) -> str:
        tools = ", ".join(d["tool_calls"]) if d["tool_calls"] else "---"
        rendered = (
            f"  [{d['turn_index']}] {d['cause']:<20} tools=[{tools}] "
            f"in={d['input_tokens']} out={d['output_tokens']}"
        )
        if "error_type" in d:
            rendered += f"\n      error={d['error_type']}: {d.get('error', '')}"
        return rendered

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]session {sid}: {len(summaries)} turn(s)[/dim]")
    _emit_records(iter(summaries), chosen_fmt, _render, limit)


# -- messages ----------------------------------------------------------------


@trace_app.command("messages")
def messages_cmd(
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    role: str | None = typer.Option(None, "--role"),
    hide_thinking: bool = typer.Option(False, "--hide-thinking"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print the conversation messages for a session."""
    query = _get_query_store()
    sid = _resolve_session_id(query, session, latest)
    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    all_msgs: list[_MessageRecord] = []
    for turn in turns:
        if isinstance(turn.trigger, UserInput):
            trigger_content = [
                block.text
                if isinstance(block, TextContent)
                else (
                    f"[image {block.mime_type}, {len(block.data)} bytes]"
                    if isinstance(block, ImageContent)
                    else f"[{type(block).__name__}]"
                )
                for block in turn.trigger.content
            ]
            all_msgs.append(
                {
                    "turn_index": turn.index,
                    "round_index": None,
                    "role": "user",
                    "content": "\n".join(trigger_content),
                }
            )
        for ri, rnd in enumerate(turn.rounds):
            blocks: list[str] = []
            for block in rnd.response.content:
                if isinstance(block, TextContent):
                    blocks.append(block.text)
                elif isinstance(block, ThinkingBlock) and not hide_thinking:
                    blocks.append(f"[thinking] {block.text}")
                elif isinstance(block, ToolCallBlock):
                    arguments = json.dumps(
                        dict(block.arguments),
                        ensure_ascii=False,
                    )[:200]
                    blocks.append(
                        f"[tool_call: {block.name}({arguments})]"
                    )
            all_msgs.append(
                {
                    "turn_index": turn.index,
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
                        "turn_index": turn.index,
                        "round_index": ri,
                        "role": "tool_result",
                        "tool": rec.call.name,
                        "is_error": rec.result.is_error,
                        "content": txt,
                    }
                )
        if isinstance(turn.outcome.cause, ProviderRequestFailed):
            all_msgs.append(
                {
                    "turn_index": turn.index,
                    "round_index": None,
                    "role": "error",
                    "content": (
                        f"{turn.outcome.cause.error_type}: "
                        f"{turn.outcome.cause.detail}"
                    ),
                }
            )
    if role:
        all_msgs = [m for m in all_msgs if m["role"] == role]

    def _render(m: _MessageRecord) -> str:
        round_label = (
            str(m["round_index"]) if m["round_index"] is not None else "---"
        )
        hdr = f"[{m['role']}] turn={m['turn_index']} round={round_label}"
        if m["role"] == "tool_result":
            error = " ERROR" if m.get("is_error") else ""
            hdr += f" tool={m.get('tool', '?')}{error}"
        return f"{hdr}\n{m['content']}\n"

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(all_msgs)} message(s)[/dim]")
    _emit_records(iter(all_msgs), chosen_fmt, _render, limit)


# -- usage -------------------------------------------------------------------


@trace_app.command("usage")
def usage_cmd(
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Token usage summary for a session."""
    query = _get_query_store()
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
    summary = {"session_id": sid, "turns": len(turns), "input_tokens": total_in, "output_tokens": total_out, "cache_read": cache_read, "cache_write": cache_write, "non_cached_input": total_in - cache_read, "cache_hit_rate": round(hit_pct, 1), "total_tokens": total_in + total_out}
    chosen_fmt = _resolve_format(fmt)
    if chosen_fmt == "text":
        sys.stdout.write(f"session:          {sid}\nturns:            {summary['turns']}\ninput tokens:     {total_in:>12,}\n  cache read:     {cache_read:>12,}  ({hit_pct:.1f}%)\n  cache write:    {cache_write:>12,}\n  non-cached:     {total_in - cache_read:>12,}\noutput tokens:    {total_out:>12,}\ntotal tokens:     {total_in + total_out:>12,}\n")
    else:
        _emit_json(summary)


# -- tools -------------------------------------------------------------------


@trace_app.command("tools")
def tools_cmd(
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    tool: str | None = typer.Option(None, "--tool"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print tool calls with arguments and results."""
    query = _get_query_store()
    sid = _resolve_session_id(query, session, latest)
    try:
        turns = list(query.turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    records: list[_ToolRecord] = []
    for turn in turns:
        for ri, rnd in enumerate(turn.rounds):
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
                        "turn_index": turn.index,
                        "round_index": ri,
                        "tool": name,
                        "args": dict(rec.call.arguments),
                        "is_error": rec.result.is_error,
                        "result": txt,
                    }
                )

    def _render(d: _ToolRecord) -> str:
        a = json.dumps(d["args"], ensure_ascii=False)[:300]
        r = d["result"][:500]
        error = "  ERROR" if d["is_error"] else ""
        return (
            f"[{d['tool']}{error}] turn={d['turn_index']} "
            f"round={d['round_index']}\n  args: {a}\n  result: {r}\n"
        )

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(records)} tool call(s)[/dim]")
    _emit_records(iter(records), chosen_fmt, _render, limit)
