"""``agentm trace`` -- query trajectories via the SDK store layer."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer

from agentm.cli._display import EXIT_NOT_FOUND, is_tty, stderr_console
from agentm.core.abi.messages import TextContent, ThinkingBlock, ToolCallBlock
from agentm.core.abi.query import SessionFilter
from agentm.core.abi.trajectory import Turn

trace_app = typer.Typer(
    name="trace",
    help="Query session trajectories.",
    no_args_is_help=True,
    add_completion=False,
)


def _get_store() -> Any:
    dsn = os.environ.get("AGENTM_TRAJECTORY_DSN")
    if dsn:
        from agentm.cli._store import _postgres_store
        store = _postgres_store(dsn)
        if store is not None:
            return store

    explicit_dir = os.environ.get("AGENTM_TRAJECTORY_DIR")
    if explicit_dir:
        d = Path(explicit_dir).expanduser()
    else:
        home = os.environ.get("AGENTM_HOME")
        d = Path(home).expanduser() / "trajectory" if home else Path.home() / ".agentm" / "trajectory"

    if not d.is_dir():
        project_local = Path.cwd() / ".agentm" / "trajectory"
        if project_local.is_dir():
            d = project_local
        else:
            stderr_console.print(
                "[red]error: no trajectory store found[/red]\n"
                "[dim]Set AGENTM_TRAJECTORY_DSN for Postgres, or AGENTM_TRAJECTORY_DIR for JSONL.[/dim]"
            )
            raise typer.Exit(EXIT_NOT_FOUND)

    from agentm.core.runtime.stores.jsonl import JsonlTrajectoryStore
    return JsonlTrajectoryStore(d)


def _get_query_store() -> Any:
    from agentm.core.runtime.stores.query import TrajectoryStoreQueryAdapter
    return TrajectoryStoreQueryAdapter(_get_store())


def _resolve_session_id(session: str | None, latest: bool) -> str:
    if session and latest:
        stderr_console.print("[red]error: --session and --latest are mutually exclusive[/red]")
        raise typer.Exit(2)
    if session:
        return session
    if latest:
        store = _get_store()
        metas = store.list_sessions()
        if not metas:
            stderr_console.print("[red]error: no sessions in store[/red]")
            raise typer.Exit(EXIT_NOT_FOUND)
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas[0].id
    stderr_console.print("[red]error: must specify --session or --latest[/red]")
    raise typer.Exit(2)


def _resolve_format(fmt: str | None) -> str:
    if fmt is not None:
        return fmt
    return "text" if is_tty() else "ndjson"


def _emit_json(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _emit_records(records: Iterable[dict[str, Any]], fmt: str, render_fn: Any, limit: int | None) -> int:
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
    rows = list(query.sessions(SessionFilter(parent_session_id=parent, purpose=purpose, limit=limit)))
    chosen_fmt = _resolve_format(fmt)
    dicts = [{"id": r.id, "parent_session_id": r.parent_session_id, "purpose": r.purpose, "cwd": r.cwd, "created_at": r.created_at} for r in rows]

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(dicts)} session(s)[/dim]")
        for d in dicts:
            sys.stdout.write(f"  {d['id']:<20} purpose={d['purpose']:<8} parent={d.get('parent_session_id') or '---'}\n")
    else:
        for d in dicts:
            _emit_json(d)


# -- turns -------------------------------------------------------------------


def _turn_summary(turn: Turn) -> dict[str, Any]:
    tool_names: list[str] = []
    tool_errors = 0
    for rnd in turn.rounds:
        for rec in rnd.tool_results:
            tool_names.append(rec.call.name if hasattr(rec.call, "name") else "?")
            if rec.result.is_error:
                tool_errors += 1
    return {
        "turn_index": turn.index, "turn_id": turn.id,
        "trigger_source": getattr(turn.trigger, "source", "?"),
        "rounds": len(turn.rounds), "tool_calls": tool_names,
        "tool_call_count": len(tool_names), "tool_error_count": tool_errors,
        "input_tokens": turn.meta.total_input_tokens, "output_tokens": turn.meta.total_output_tokens,
        "cache_read": turn.meta.cache_read_tokens, "model": turn.meta.model_id,
        "cause": type(turn.outcome.cause).__name__,
    }


@trace_app.command("turns")
def turns_cmd(
    session: str | None = typer.Option(None, "--session", "-s"),
    latest: bool = typer.Option(False, "--latest"),
    limit: int | None = typer.Option(None, "--limit"),
    fmt: str | None = typer.Option(None, "--format"),
) -> None:
    """Print per-turn summaries for a session."""
    sid = _resolve_session_id(session, latest)
    try:
        turns = list(_get_query_store().turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    summaries = [_turn_summary(t) for t in turns]

    def _render(d: dict[str, Any]) -> str:
        tools = ", ".join(d["tool_calls"]) if d["tool_calls"] else "---"
        return f"  [{d['turn_index']}] {d['cause']:<20} tools=[{tools}] in={d['input_tokens']} out={d['output_tokens']}"

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
    sid = _resolve_session_id(session, latest)
    try:
        turns = list(_get_query_store().turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    all_msgs: list[dict[str, Any]] = []
    for turn in turns:
        for ri, rnd in enumerate(turn.rounds):
            blocks: list[str] = []
            for block in rnd.response.content:
                if isinstance(block, TextContent):
                    blocks.append(block.text)
                elif isinstance(block, ThinkingBlock) and not hide_thinking:
                    blocks.append(f"[thinking] {block.text}")
                elif isinstance(block, ToolCallBlock):
                    blocks.append(f"[tool_call: {block.name}({json.dumps(dict(block.arguments), ensure_ascii=False)[:200]})]")
            all_msgs.append({"turn_index": turn.index, "round_index": ri, "role": "assistant", "content": "\n".join(blocks)})
            for rec in rnd.tool_results:
                txt = "".join(b.text for b in rec.result.content if isinstance(b, TextContent))[:500]
                all_msgs.append({"turn_index": turn.index, "round_index": ri, "role": "tool_result", "tool": rec.call.name, "is_error": rec.result.is_error, "content": txt})
    if role:
        all_msgs = [m for m in all_msgs if m["role"] == role]

    def _render(m: dict[str, Any]) -> str:
        hdr = f"[{m['role']}] turn={m['turn_index']} round={m['round_index']}"
        if m["role"] == "tool_result":
            hdr += f" tool={m.get('tool','?')}{' ERROR' if m.get('is_error') else ''}"
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
    sid = _resolve_session_id(session, latest)
    try:
        turns = list(_get_query_store().turns(sid))
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
    sid = _resolve_session_id(session, latest)
    try:
        turns = list(_get_query_store().turns(sid))
    except KeyError:
        stderr_console.print(f"[red]error: session not found: {sid}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    chosen_fmt = _resolve_format(fmt)
    records: list[dict[str, Any]] = []
    for turn in turns:
        for ri, rnd in enumerate(turn.rounds):
            for rec in rnd.tool_results:
                name = rec.call.name if hasattr(rec.call, "name") else "?"
                if tool and name != tool:
                    continue
                txt = "".join(b.text for b in rec.result.content if isinstance(b, TextContent))
                records.append({"turn_index": turn.index, "round_index": ri, "tool": name, "args": dict(rec.call.arguments) if hasattr(rec.call, "arguments") else {}, "is_error": rec.result.is_error, "result": txt})

    def _render(d: dict[str, Any]) -> str:
        a = json.dumps(d["args"], ensure_ascii=False)[:300]
        r = d["result"][:500]
        return f"[{d['tool']}{'  ERROR' if d['is_error'] else ''}] turn={d['turn_index']} round={d['round_index']}\n  args: {a}\n  result: {r}\n"

    if chosen_fmt == "text":
        stderr_console.print(f"[dim]{len(records)} tool call(s)[/dim]")
    _emit_records(iter(records), chosen_fmt, _render, limit)
