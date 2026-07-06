"""``agentm trace`` — query / filter / project an OTLP/JSON session log.

Thin shell over :class:`agentm.core.abi.TraceReader`. Every verb is a 5-15
line dispatch to a reader method plus a shared output formatter; complex
filtering is intentionally NOT a built-in DSL — pipe through ``jq`` for
that. See ``.claude/designs/single-event-log.md`` for the on-disk format.

Contract (stable across minor versions):

* stdout carries data only; logs / warnings / errors all go to stderr.
* ``--format ndjson`` is the canonical machine surface. ``--format text``
  is for humans. Default switches by TTY.
* Exit codes follow the cli-design skill table (0/2/3/4/7).

Dispatched as the ``agentm trace`` subcommand by ``agentm.cli.main`` (which
lazily hands argv to ``main`` below so the prompt path never imports this).
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any, Callable, TextIO

import typer
from loguru import logger

from agentm.cli_trace_format import (
    _ANSI,
    _color_enabled,
    _emit,
    _fail,
    _info,
    _log_to_dict,
    _open_output,
    _paint,
    _parse_where,
    _render_content_blocks,
    _resolve_format,
    _span_to_dict,
    _within_window,
)
from agentm.core.abi import Span, TraceReader
from agentm.env import autoload_dotenv, resolve_cli_cwd

# ClickHouse backend — lazy import to avoid urllib cost on the prompt path.
_ch_mod: Any = None


def _ch() -> Any:
    """Lazily import the ClickHouse backend module."""
    global _ch_mod
    if _ch_mod is None:
        from agentm.core.observability import clickhouse
        _ch_mod = clickhouse
    return _ch_mod


def _trace_cwd(cwd: Path | None = None) -> Path:
    return resolve_cli_cwd(cwd)


def _trace_clickhouse_url(cwd: Path | None = None) -> str | None:
    """Return a ClickHouse URL only when remote trace storage is configured.

    ``clickhouse.get_url()`` also auto-detects localhost for operators who call
    the backend directly. The trace CLI is stricter: a local ClickHouse query
    endpoint should not shadow a freshly written JSONL fallback session unless
    the environment says this run is exporting remote traces.
    """
    autoload_dotenv(_trace_cwd(cwd))
    if not (
        os.environ.get("AGENTM_CLICKHOUSE_URL")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return None
    return _ch().get_url()


def _local_session_file(session: str | None, cwd: Path | None) -> Path | None:
    """Return the local JSONL path for an explicit session if it exists."""
    if session is None:
        return None
    from agentm.core.observability.otel_export import resolve_observability_dir

    base = _trace_cwd(cwd)
    path = resolve_observability_dir(base) / f"{session}.jsonl"
    return path if path.is_file() else None


def _ch_session(
    file: Path | None, session: str | None, latest: bool, cwd: Path | None = None,
) -> tuple[str, str] | None:
    """Try to resolve a ClickHouse-backed session.

    Returns ``(ch_url, session_id)`` when remote trace storage is configured
    and ``--file`` is not forcing JSONL mode. Returns ``None`` to fall back
    to the local JSONL path.
    """
    if file is not None:
        return None
    url = _trace_clickhouse_url(cwd)
    if url is None:
        return None
    local_file = _local_session_file(session, cwd)
    if local_file is not None:
        try:
            if _ch().session_header(url, session) is None:
                logger.debug(
                    "trace: ClickHouse has no header for session {}; "
                    "using local JSONL {}",
                    session,
                    local_file,
                )
                return None
        except Exception as exc:
            logger.debug(
                "trace: ClickHouse session lookup failed for {}; "
                "using local JSONL {}: {}",
                session,
                local_file,
                exc,
            )
            return None
    resolved_cwd = _trace_cwd(cwd).resolve()
    sid = _ch().resolve_session(
        url, session, latest, str(resolved_cwd)
    )
    if sid is None:
        return None
    return url, sid

app = typer.Typer(
    name="trace",
    help=(
        "Query an OTLP/JSON session log written by the observability atom. "
        "Use one of --file / --session / --latest to pick a file."
    ),
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_source(
    file: Path | None,
    session: str | None,
    latest: bool,
    cwd: Path | None,
) -> Path:
    """Pick exactly one of ``--file`` / ``--session`` / ``--latest``.

    Mutual exclusion is validated explicitly (cli-design §2). Missing
    file → exit 3; unreadable → exit 4.
    """
    resolved_cwd = _trace_cwd(cwd)

    chosen = sum([file is not None, session is not None, latest])
    if chosen == 0:
        _fail(
            2,
            "argument",
            "must specify one of --file, --session, or --latest",
            "pass --latest to grab the most recent session under $AGENTM_HOME/observability/",
        )
    if chosen > 1:
        _fail(
            2,
            "argument",
            "--file, --session, and --latest are mutually exclusive",
            "pick one",
        )
    if file is not None:
        path = file
    elif session is not None:
        from agentm.core.observability.otel_export import resolve_observability_dir

        path = resolve_observability_dir(resolved_cwd) / f"{session}.jsonl"
    else:
        from agentm.core.observability.otel_export import resolve_observability_dir

        obs_dir = resolve_observability_dir(resolved_cwd)
        if not obs_dir.is_dir():
            _fail(
                3,
                "not_found",
                f"observability directory not found: {obs_dir}",
                "run `agentm` once to produce a session log, or set AGENTM_OBSERVABILITY_DIR",
            )
        candidates = sorted(
            (p for p in obs_dir.glob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            _fail(
                3,
                "not_found",
                f"no *.jsonl files under {obs_dir}",
                "run `agentm` once to produce a session log, "
                "or set AGENTM_CLICKHOUSE_URL for remote trace storage",
            )
        path = candidates[0]
    if not path.is_file():
        _fail(3, "not_found", f"trace file not found: {path}")
    if not os.access(path, os.R_OK):
        _fail(4, "permission", f"trace file not readable: {path}")
    return path


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------


FileOpt = Annotated[
    Path | None,
    typer.Option(
        "--file",
        help="Path to the session JSONL. Mutually exclusive with --session/--latest.",
    ),
]
SessionOpt = Annotated[
    str | None,
    typer.Option(
        "--session",
        help="Session id; resolves to $AGENTM_HOME/observability/<id>.jsonl.",
    ),
]
LatestOpt = Annotated[
    bool,
    typer.Option(
        "--latest",
        help="Use the most recently modified *.jsonl under $AGENTM_HOME/observability/.",
    ),
]
CwdOpt = Annotated[
    Path | None,
    typer.Option(
        "--cwd",
        help=(
            "Working directory for .env loading and ClickHouse --latest "
            "filtering. Defaults to AGENTM_CWD, then the process cwd."
        ),
    ),
]
FormatOpt = Annotated[
    str | None,
    typer.Option(
        "--format",
        help="Output format: ndjson, json, text. Defaults: ndjson off-TTY, text on-TTY.",
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
WhereOpt = Annotated[
    list[str] | None,
    typer.Option("--where", help="Attribute filter K=V (repeatable, equality only)."),
]
SinceOpt = Annotated[
    int | None,
    typer.Option("--since", help="Only records with time/start >= NS (Unix nanos)."),
]
UntilOpt = Annotated[
    int | None,
    typer.Option("--until", help="Only records with time/start <= NS (Unix nanos)."),
]
UnwrapAttrsOpt = Annotated[
    bool,
    typer.Option(
        "--unwrap-attrs",
        help="Hoist OTLP attribute keys to the top level of each record.",
    ),
]


# ---------- follow (tail -f for messages) ------------------------------------


def _follow_ch_messages(
    url: str,
    sid: str,
    roles: set[str],
    types: set[str],
    seen: int,
    fmt: str,
    render_fn: Callable[[dict[str, Any]], str],
    sink: TextIO,
) -> None:
    """Poll ClickHouse for new messages, ``tail -f`` style."""
    offset = seen
    try:
        while True:
            time.sleep(1.0)
            records = list(
                _ch().messages(
                    url, sid, roles=roles or None, types=types or None,
                )
            )
            if len(records) <= offset:
                continue
            for entry in records[offset:]:
                if fmt == "ndjson" or fmt == "json":
                    sink.write(json.dumps(entry, ensure_ascii=False))
                    sink.write("\n")
                else:
                    sink.write(render_fn(entry))
                    sink.write("\n")
                sink.flush()
            offset = len(records)
    except KeyboardInterrupt:
        # User stopped the follow loop with Ctrl-C — clean exit.
        logger.debug("trace: messages follow interrupted by user")


def _tail_messages(
    path: Path,
    roles: set[str],
    types: set[str],
    fmt: str,
    render_fn: Callable[[dict[str, Any]], str],
    sink: TextIO,
) -> None:
    """Tail the JSONL file for new messages, ``tail -f`` style."""

    from agentm.core.observability.otlp import (
        iter_log_records as _iter_lr,
        otlp_unwrap,
    )

    with path.open("r", encoding="utf-8") as fh:
        fh.seek(0, 2)
        buf = ""
        try:
            while True:
                chunk = fh.read()
                if not chunk:
                    time.sleep(0.3)
                    continue
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(raw, dict):
                        continue
                    for lr in _iter_lr(raw):
                        if lr.get("eventName") != "agentm.message.appended":
                            continue
                        body = otlp_unwrap(lr.get("body"))
                        if not isinstance(body, dict):
                            continue
                        payload = body.get("payload") or {}
                        if types and body.get("type") not in types:
                            continue
                        if roles and payload.get("role") not in roles:
                            continue
                        if fmt == "ndjson":
                            sink.write(json.dumps(body, ensure_ascii=False))
                            sink.write("\n")
                        elif fmt == "json":
                            sink.write(json.dumps(body, ensure_ascii=False))
                            sink.write("\n")
                        else:
                            sink.write(render_fn(body))
                            sink.write("\n")
                        sink.flush()
        except KeyboardInterrupt:
            # User stopped the follow loop with Ctrl-C — clean exit.
            logger.debug("trace: logs follow interrupted by user")


# ---------- messages --------------------------------------------------------


@app.command("messages")
def messages_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    role: Annotated[
        list[str] | None,
        typer.Option("--role", help="Filter by payload.role (repeatable)."),
    ] = None,
    entry_type: Annotated[
        list[str] | None,
        typer.Option(
            "--type",
            help=(
                "Filter by SessionEntry.type (repeatable). "
                "Default: 'message' (standard messages only). "
                "Pass '--type all' to include every entry type."
            ),
        ),
    ] = None,
    follow: Annotated[
        bool,
        typer.Option(
            "--follow", "-f",
            help="Follow the session log in real time (like tail -f). Ctrl-C to stop.",
        ),
    ] = False,
    hide_thinking: Annotated[
        bool,
        typer.Option(
            "--hide-thinking",
            help="Drop assistant thinking blocks from text output (noisy on long runs).",
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Disable ANSI colors. Default: on if stdout is a TTY, off otherwise.",
        ),
    ] = False,
    limit: LimitOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Print the conversation trajectory (user / assistant / tool messages).

    The system prompt is persisted by default and surfaces as a synthetic
    ``[system]`` message #0. Set ``AGENTM_TRACE_SYSTEM_PROMPT=0`` to
    disable persistence (reduces JSONL size for bulk eval runs).

    Examples:

      agentm trace messages --latest
      agentm trace messages --latest --role assistant --hide-thinking
      agentm trace messages --session 7b0f... --format ndjson > traj.ndjson
    """

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        roles = set(role or [])
        if entry_type is None:
            types = {"message"}
        elif entry_type == ["all"]:
            types = set()
        else:
            types = set(entry_type)
        color = _color_enabled(sink, no_color)

        def _render(entry: dict[str, Any]) -> str:
            payload = entry.get("payload") or {}
            role_str = str(payload.get("role") or entry.get("type") or "?")
            entry_id = str(entry.get("id") or "")
            id_suffix = f" id={entry_id}" if entry_id else ""
            color_key = role_str if role_str in _ANSI else "kind"
            header = _paint(f"[{role_str}{id_suffix}]", color_key, color)
            lines = _render_content_blocks(
                payload.get("content"),
                color=color,
                hide_thinking=hide_thinking,
            )
            if not lines and payload and "content" not in payload:
                lines = [
                    f"  {line}"
                    for line in json.dumps(
                        payload, ensure_ascii=False, indent=2
                    ).splitlines()
                ]
            body = "\n".join([header, *lines]) if lines else header
            return body + "\n"

        # ClickHouse path (primary)
        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            records = _ch().messages(
                url, sid, roles=roles or None, types=types or None,
            )
            n = _emit(records, chosen_fmt, _render, sink, limit)
            _info(f"{n} message(s)")
            if follow:
                _info("following (Ctrl-C to stop)…")
                _follow_ch_messages(
                    url, sid, roles or set(), types, n,
                    chosen_fmt, _render, sink,
                )
            return

        # JSONL fallback
        path = _resolve_source(file, session, latest, cwd)

        def _filtered() -> Iterator[dict[str, Any]]:
            reader = TraceReader(path)
            sys_iter = reader.iter_log_records(name="agentm.llm.system_prompt")
            first_sys = next(sys_iter, None)
            if first_sys is not None:
                body = getattr(first_sys, "body", None) or {}
                text = ""
                if isinstance(body, dict):
                    text = str(body.get("text") or "")
                synth = {
                    "type": "message",
                    "id": "system-prompt-turn0",
                    "parent_id": None,
                    "timestamp": 0,
                    "payload": {
                        "role": "system",
                        "content": [{"type": "text", "text": text}],
                    },
                }
                if (not types or synth["type"] in types) and (
                    not roles or "system" in roles
                ):
                    yield synth
            for entry in reader.load_messages():
                if types and entry.get("type") not in types:
                    continue
                payload = entry.get("payload") or {}
                if roles and payload.get("role") not in roles:
                    continue
                yield entry

        n = _emit(_filtered(), chosen_fmt, _render, sink, limit)
        _info(f"{n} message(s)")
        if follow:
            _info("following (Ctrl-C to stop)…")
            _tail_messages(path, roles, types, chosen_fmt, _render, sink)
    finally:
        if close:
            sink.close()


# ---------- turns -----------------------------------------------------------


@app.command("turns")
def turns_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    limit: LimitOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Print per-turn summaries (stop reason, tool counts, tokens).

    Examples:

      agentm trace turns --latest
      agentm trace turns --latest --format ndjson | jq '.input_tokens'
    """

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)

        def _render(turn: dict[str, Any]) -> str:
            turn_id = turn.get("turn_id")
            id_part = f" id={turn_id}" if turn_id is not None else ""
            return (
                f"[turn index={turn.get('turn_index','?')}{id_part}] "
                f"stop={turn.get('stop_reason','?')} "
                f"tool_calls={turn.get('tool_call_count',0)} "
                f"errors={turn.get('tool_error_count',0)} "
                f"in={turn.get('input_tokens',0)} "
                f"out={turn.get('output_tokens',0)}"
            )

        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            records: Any = _ch().turns(url, sid)
        else:
            path = _resolve_source(file, session, latest, cwd)
            records = TraceReader(path).load_turn_summaries()
        n = _emit(records, chosen_fmt, _render, sink, limit)
        _info(f"{n} turn(s)")
    finally:
        if close:
            sink.close()


# ---------- usage ------------------------------------------------------------


@app.command("usage")
def usage_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Token economics summary: input, cache hit, output, cost estimate.

    Examples:

      agentm trace usage --latest
      agentm trace usage --file path/to/session.jsonl --format ndjson
    """

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)

        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            summary = _ch().usage(url, sid)
        else:
            path = _resolve_source(file, session, latest, cwd)
            records = TraceReader(path).load_turn_summaries()
            if not records:
                summary = None
            else:
                total_input = sum(r.get("input_tokens", 0) for r in records)
                total_output = sum(r.get("output_tokens", 0) for r in records)
                cache_read = sum(r.get("cache_read", 0) for r in records)
                cache_write = sum(r.get("cache_write", 0) for r in records)
                non_cached = total_input - cache_read
                hit_pct = (cache_read / total_input * 100) if total_input else 0.0
                summary = {
                    "turns": len(records),
                    "input_tokens": total_input,
                    "cache_read": cache_read,
                    "cache_write": cache_write,
                    "non_cached_input": non_cached,
                    "cache_hit_rate": round(hit_pct, 1),
                    "output_tokens": total_output,
                    "total_tokens": total_input + total_output,
                }

        if summary is None:
            _info("no turns found")
            return

        if chosen_fmt == "ndjson":
            sink.write(json.dumps(summary) + "\n")
        else:
            sink.write(
                f"turns:            {summary['turns']}\n"
                f"input tokens:     {summary['input_tokens']:>12,}\n"
                f"  cache read:     {summary['cache_read']:>12,}  ({summary['cache_hit_rate']:.1f}%)\n"
                f"  cache write:    {summary['cache_write']:>12,}\n"
                f"  non-cached:     {summary['non_cached_input']:>12,}\n"
                f"output tokens:    {summary['output_tokens']:>12,}\n"
                f"total tokens:     {summary['total_tokens']:>12,}\n"
            )
    finally:
        if close:
            sink.close()


# ---------- chats -----------------------------------------------------------


@app.command("chats")
def chats_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help="Filter by gen_ai.request.model (repeatable)."),
    ] = None,
    where: WhereOpt = None,
    since: SinceOpt = None,
    until: UntilOpt = None,
    limit: LimitOpt = None,
    unwrap_attrs: UnwrapAttrsOpt = False,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Print every LLM call (one row per chat request, with duration).

    Examples:

      agentm trace chats --latest
      agentm trace chats --latest --model Doubao-Seed-2.0-pro
    """

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        where_filters = _parse_where(where)
        wanted_models = set(model or [])

        def _render(d: dict[str, Any]) -> str:
            attrs = d if "gen_ai.request.model" in d else d.get("attributes", {})
            duration_ns = attrs.get("agentm.llm.duration_ns")
            if isinstance(duration_ns, str):
                try:
                    duration_ns = int(duration_ns)
                except ValueError:
                    duration_ns = None
            duration = (
                f"{duration_ns / 1e9:.2f}s" if isinstance(duration_ns, int) else "?"
            )
            return (
                f"[chat] model={attrs.get('gen_ai.request.model','?')} "
                f"turn={attrs.get('agentm.turn.index','?')} "
                f"messages={attrs.get('agentm.llm.message_count',0)} "
                f"duration={duration}"
            )

        def _apply_filters(records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
            for d in records:
                attrs = d.get("attributes", {})
                if not _within_window(d.get("start_time_unix_nano"), since, until):
                    continue
                if any(attrs.get(k) != v for k, v in where_filters.items()):
                    continue
                if wanted_models:
                    m = attrs.get("gen_ai.request.model")
                    if m not in wanted_models:
                        continue
                yield d

        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            raw = _ch().chats(url, sid)
        else:
            path = _resolve_source(file, session, latest, cwd)
            raw = (
                _span_to_dict(span, unwrap_attrs)
                for span in TraceReader(path).chat_calls()
            )

        n = _emit(_apply_filters(raw), chosen_fmt, _render, sink, limit)
        _info(f"{n} chat call(s)")
    finally:
        if close:
            sink.close()


# ---------- tools -----------------------------------------------------------


def _jsonl_tool_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield tool-call dicts from a JSONL file (shared by tools_cmd)."""
    for span, args_log, result_log in TraceReader(path).tool_calls():
        tool_name = span.attributes.get(
            "gen_ai.tool.name"
        ) or span.name.removeprefix("execute_tool ").strip()
        args_payload: Any = args_log.body if args_log is not None else None
        if args_payload is None:
            raw = span.attributes.get("gen_ai.tool.call.arguments")
            if isinstance(raw, str):
                try:
                    args_payload = json.loads(raw)
                except (TypeError, ValueError):
                    args_payload = raw
        result_payload: Any = result_log.body if result_log is not None else None
        if result_payload is None:
            raw = span.attributes.get("gen_ai.tool.call.result")
            if isinstance(raw, str):
                try:
                    result_payload = json.loads(raw)
                except (TypeError, ValueError):
                    result_payload = raw
        yield {
            "tool": tool_name,
            "span_id": span.span_id,
            "start_time_unix_nano": span.start_time_unix_nano,
            "end_time_unix_nano": span.end_time_unix_nano,
            "args": args_payload,
            "result": result_payload,
            "attributes": span.attributes,
        }


@app.command("tools")
def tools_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    tool: Annotated[
        list[str] | None,
        typer.Option("--tool", help="Filter by tool name (repeatable)."),
    ] = None,
    where: WhereOpt = None,
    since: SinceOpt = None,
    until: UntilOpt = None,
    limit: LimitOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Print every tool call as one row (args + result joined).

    Each row carries {tool, args, result, span_id, start/end, attributes}.
    The OTLP wire splits args/result into separate log records; this
    verb joins them since consumers almost always want them together.

    Examples:

      agentm trace tools --latest
      agentm trace tools --latest --tool write --tool edit
      agentm trace tools --latest --format ndjson | jq '.tool'
    """

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        where_filters = _parse_where(where)
        wanted = set(tool or [])

        def _render(d: dict[str, Any]) -> str:
            args_repr = json.dumps(d["args"], ensure_ascii=False)
            result_repr = json.dumps(d["result"], ensure_ascii=False)
            return (
                f"[tool {d['tool']}] args={args_repr}\n"
                f"  → result={result_repr}"
            )

        def _apply_filters(records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
            for d in records:
                if wanted and d["tool"] not in wanted:
                    continue
                if not _within_window(d.get("start_time_unix_nano"), since, until):
                    continue
                if any(
                    d.get("attributes", {}).get(k) != v
                    for k, v in where_filters.items()
                ):
                    continue
                yield d

        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            raw = _ch().tools(url, sid)
        else:
            path = _resolve_source(file, session, latest, cwd)
            raw = _jsonl_tool_records(path)

        n = _emit(_apply_filters(raw), chosen_fmt, _render, sink, limit)
        _info(f"{n} tool call(s)")
    finally:
        if close:
            sink.close()


# ---------- info (session metadata: header + fingerprint) -------------------


@app.command("info")
def info_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    what: Annotated[
        str,
        typer.Option(
            "--what",
            help="Which metadata to show: header, fingerprint, or both.",
        ),
    ] = "both",
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Print session metadata (header + atom fingerprint).

    Header carries id / cwd / parent / scenario; fingerprint carries
    task_meta and atom hashes (your build identity). Missing either
    one is exit 3 unless --what scopes it down.

    Examples:

      agentm trace info --latest
      agentm trace info --latest --what fingerprint --format json
    """

    if what not in {"header", "fingerprint", "both"}:
        _fail(2, "argument", f"--what {what!r} not in {{header,fingerprint,both}}")
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)

        payload: dict[str, Any]
        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            payload = _ch().info(url, sid)
        else:
            path = _resolve_source(file, session, latest, cwd)
            reader = TraceReader(path)
            payload = {}
            if what in {"header", "both"}:
                header = reader.load_session_header()
                if header is None and what == "header":
                    _fail(3, "not_found", "no agentm.session.header record in trace")
                if header is not None:
                    payload["header"] = header
            if what in {"fingerprint", "both"}:
                fp = reader.load_session_fingerprint()
                if fp is None and what == "fingerprint":
                    _fail(
                        3, "not_found", "no agentm.session.fingerprint record in trace"
                    )
                if fp is not None:
                    payload["fingerprint"] = fp

        if what != "both" and what not in payload:
            _fail(3, "not_found", f"no agentm.session.{what} record in trace")
        if not payload:
            _fail(3, "not_found", "no session metadata found in trace")

        def _render(d: dict[str, Any]) -> str:
            lines: list[str] = []
            for key in ("header", "fingerprint"):
                if key not in d:
                    continue
                lines.append(f"[{key}]")
                for line in json.dumps(d[key], ensure_ascii=False, indent=2).splitlines():
                    lines.append(f"  {line}")
                lines.append("")
            return "\n".join(lines).rstrip("\n")

        _emit([payload], chosen_fmt, _render, sink, None)
    finally:
        if close:
            sink.close()


# ---------- generic spans / logs -------------------------------------------


@app.command("spans")
def spans_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Exact span name match."),
    ] = None,
    name_prefix: Annotated[
        str | None,
        typer.Option("--name-prefix", help="Span name prefix match."),
    ] = None,
    where: WhereOpt = None,
    since: SinceOpt = None,
    until: UntilOpt = None,
    limit: LimitOpt = None,
    unwrap_attrs: UnwrapAttrsOpt = False,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Generic span query (custom --name / --where / --since).

    Use the high-level verbs (chats / tools / turns) first; reach for
    this only when you need something they don't surface.

    Examples:

      agentm trace spans --latest --name-prefix execute_tool
      agentm trace spans --latest --name 'chat Doubao-Seed-2.0-pro' --unwrap-attrs
    """

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        where_filters = _parse_where(where)

        def _render(d: dict[str, Any]) -> str:
            return f"[span] {d['name']}"

        def _apply_filters(records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
            for d in records:
                if not _within_window(d.get("start_time_unix_nano"), since, until):
                    continue
                attrs = d.get("attributes", {})
                if any(attrs.get(k) != v for k, v in where_filters.items()):
                    continue
                yield d

        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            raw = _ch().spans(url, sid, name=name, name_prefix=name_prefix)
        else:
            path = _resolve_source(file, session, latest, cwd)

            def _jsonl_spans() -> Iterator[dict[str, Any]]:
                for span in TraceReader(path).iter_spans(
                    name=name, attribute_filters=where_filters or None
                ):
                    if name_prefix is not None and not span.name.startswith(name_prefix):
                        continue
                    yield _span_to_dict(span, unwrap_attrs)

            raw = _jsonl_spans()

        n = _emit(_apply_filters(raw), chosen_fmt, _render, sink, limit)
        _info(f"{n} span(s)")
    finally:
        if close:
            sink.close()


@app.command("logs")
def logs_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Exact eventName match."),
    ] = None,
    name_prefix: Annotated[
        str | None,
        typer.Option("--name-prefix", help="eventName prefix match."),
    ] = None,
    where: WhereOpt = None,
    since: SinceOpt = None,
    until: UntilOpt = None,
    limit: LimitOpt = None,
    unwrap_attrs: UnwrapAttrsOpt = False,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Generic log query (custom --name / --where / --since).

    Use the high-level verbs (messages / turns / info) first; reach for
    this only when you need something they don't surface.

    Examples:

      agentm trace logs --latest --name agentm.diagnostic
      agentm trace logs --latest --name-prefix agentm.handler --limit 10
    """

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        where_filters = _parse_where(where)

        def _render(d: dict[str, Any]) -> str:
            return f"[log] {d['event_name']}"

        def _apply_filters(records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
            for d in records:
                if not _within_window(d.get("time_unix_nano"), since, until):
                    continue
                attrs = d.get("attributes", {})
                if any(attrs.get(k) != v for k, v in where_filters.items()):
                    continue
                yield d

        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            raw = _ch().logs(url, sid, name=name, name_prefix=name_prefix)
        else:
            path = _resolve_source(file, session, latest, cwd)

            def _jsonl_logs() -> Iterator[dict[str, Any]]:
                for rec in TraceReader(path).iter_log_records(
                    name=name, attribute_filters=where_filters or None
                ):
                    if name_prefix is not None and not rec.event_name.startswith(
                        name_prefix
                    ):
                        continue
                    yield _log_to_dict(rec, unwrap_attrs)

            raw = _jsonl_logs()

        n = _emit(_apply_filters(raw), chosen_fmt, _render, sink, limit)
        _info(f"{n} log record(s)")
    finally:
        if close:
            sink.close()


# ---------- stats -----------------------------------------------------------


@app.command("stats")
def stats_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Show a histogram of every eventName / span name (orientation probe).

    Run this first on an unfamiliar session to see what's worth digging
    into with the other verbs.

    Examples:

      agentm trace stats --latest
      agentm trace stats --latest --format json | jq '.spans'
    """

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)

        ch = _ch_session(file, session, latest, cwd)
        if ch:
            url, sid = ch
            summary = _ch().stats(url, sid)
        else:
            path = _resolve_source(file, session, latest, cwd)
            log_counts: Counter[str] = Counter()
            span_counts: Counter[str] = Counter()
            for item in TraceReader(path).iter_all():
                if isinstance(item, Span):
                    span_counts[item.name] += 1
                else:
                    log_counts[item.event_name] += 1
            logs_sorted: dict[str, int] = dict(
                sorted(log_counts.items(), key=lambda kv: -kv[1])
            )
            spans_sorted: dict[str, int] = dict(
                sorted(span_counts.items(), key=lambda kv: -kv[1])
            )
            summary = {
                "file": str(path),
                "logs": logs_sorted,
                "spans": spans_sorted,
                "log_total": sum(log_counts.values()),
                "span_total": sum(span_counts.values()),
            }
        if chosen_fmt == "text":
            sink.write(f"file: {summary['file']}\n")
            sink.write(f"logs ({summary['log_total']}):\n")
            for k, v in summary["logs"].items():
                sink.write(f"  {v:>5}  {k}\n")
            sink.write(f"spans ({summary['span_total']}):\n")
            for k, v in summary["spans"].items():
                sink.write(f"  {v:>5}  {k}\n")
        elif chosen_fmt == "json":
            json.dump(summary, sink, ensure_ascii=False, indent=2)
            sink.write("\n")
        else:  # ndjson
            sink.write(json.dumps(summary, ensure_ascii=False))
            sink.write("\n")
    finally:
        if close:
            sink.close()


# ---------- index (directory-granular session topology) ---------------------

_CACHE_FILE = ".trace_index_cache.json"


def _scan_file(path: Path) -> dict[str, Any] | None:
    """Single-pass scan: identity + line count. Returns ``None`` to skip."""
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
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        # Missing/corrupt index cache — rebuild from scratch (returns empty).
        logger.debug("trace: index cache unreadable at {}, rebuilding: {}", cache_path, exc)
    return {}


def _save_index_cache(obs_dir: Path, cache: dict[str, Any]) -> None:
    cache_path = obs_dir / _CACHE_FILE
    tmp = cache_path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, separators=(",", ":"))
        tmp.replace(cache_path)
    except OSError:
        tmp.unlink(missing_ok=True)


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
    """Apply index filters to a stream of identity rows."""
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
            rc = row.get("records")
            if rc is not None and rc < min_records:
                continue
        yield row


@app.command("index")
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
        typer.Option(
            "--trace",
            help="Filter by trace_id (exact match).",
        ),
    ] = None,
    purpose: Annotated[
        list[str] | None,
        typer.Option(
            "--purpose",
            help="Filter by purpose (repeatable, e.g. root / cognitive_audit_extractor).",
        ),
    ] = None,
    scenario: Annotated[
        list[str] | None,
        typer.Option(
            "--scenario",
            help="Filter by scenario name (repeatable).",
        ),
    ] = None,
    roots_only: Annotated[
        bool,
        typer.Option(
            "--roots-only",
            help="Only show root sessions (no parent).",
        ),
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
            "--jobs", "-j",
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
    """Map every session file to its trace-tree identity (one row per file).

    A logical "trace" spans many JSONL files — one root session plus N
    spawned children (extractor / auditor / ...). This is the only
    directory-granular verb: it scans the observability dir and emits one
    identity row per ``agentm.session.start`` file, so jq + shell can go
    from a ``trace_id`` to its session files. Files with no session.start
    record are skipped.

    Results are cached in ``.trace_index_cache.json`` inside the
    observability directory. Only new or modified files are re-scanned on
    subsequent runs. Pass ``--no-cache`` to force a full rescan.

    Each row: {path, trace_id, session_id, parent_session_id, purpose,
    scenario, records}.

    Examples:

      agentm trace index --trace abc123
      agentm trace index --roots-only --scenario rca
      agentm trace index --children-of abc123 --purpose cognitive_audit_auditor
      agentm trace index --min-records 10 --format ndjson
      agentm trace index --dir ~/.agentm/observability --format text
    """

    if roots_only and children_of is not None:
        _fail(2, "argument", "--roots-only and --children-of are mutually exclusive")

    purposes = set(purpose or [])
    scenarios = set(scenario or [])

    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        resolved_cwd = _trace_cwd(cwd)
        autoload_dotenv(resolved_cwd)

        def _render(d: dict[str, Any]) -> str:
            return (
                f"[session {d.get('session_id') or '?'}] "
                f"trace={d.get('trace_id') or '?'} "
                f"parent={d.get('parent_session_id') or '-'} "
                f"purpose={d.get('purpose') or '-'} "
                f"records={d.get('records') if d.get('records') is not None else '?'} "
                f"{d.get('path', '-')}"
            )

        # ClickHouse fast path (skip when --dir forces JSONL scan)
        if directory is None:
            ch_url = _trace_clickhouse_url(cwd)
            if ch_url is not None:
                raw = _ch().index(
                    ch_url,
                    trace_id=trace_id,
                    purposes=purposes or None,
                    scenarios=scenarios or None,
                    roots_only=roots_only,
                    children_of=children_of,
                )
                filtered = _index_filter(
                    raw,
                    trace_id=None,
                    purposes=set(),
                    scenarios=set(),
                    roots_only=False,
                    children_of=None,
                    min_records=min_records,
                )
                n = _emit(filtered, chosen_fmt, _render, sink, limit)
                _info(f"{n} session(s) (clickhouse)")
                return

        # JSONL scan path
        if directory is not None:
            obs_dir = directory
        else:
            from agentm.core.observability.otel_export import resolve_observability_dir

            obs_dir = resolve_observability_dir(resolved_cwd)
        if not obs_dir.is_dir():
            _fail(
                3,
                "not_found",
                f"observability directory not found: {obs_dir}",
                "pass --dir, run `agentm` once, or set AGENTM_OBSERVABILITY_DIR",
            )
        files = sorted(p for p in obs_dir.glob("*.jsonl") if p.is_file())

        cache = {} if no_cache else _load_index_cache(obs_dir)
        cached_rows: list[dict[str, Any]] = []
        stale_files: list[Path] = []

        for path in files:
            fname = path.name
            entry = cache.get(fname)
            if entry is not None:
                try:
                    st = path.stat()
                except OSError:
                    continue
                if entry.get("_mtime") == st.st_mtime and entry.get("_size") == st.st_size:
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
                for path, row in zip(stale_files, pool.map(_scan_file, stale_files)):
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    cache[path.name] = {
                        "_mtime": st.st_mtime,
                        "_size": st.st_size,
                        "row": row,
                    }
                    if row is not None:
                        scanned_rows.append(row)

        live_names = {p.name for p in files}
        for dead in [k for k in cache if k not in live_names]:
            del cache[dead]

        if stale_files or len(cache) != len(files):
            _save_index_cache(obs_dir, cache)

        _info(
            f"{len(cached_rows)} cached, {len(scanned_rows)} scanned"
            f" ({len(stale_files)} file(s) re-scanned)"
        )

        all_rows = sorted(cached_rows + scanned_rows, key=lambda d: d["path"])
        filtered = _index_filter(
            all_rows,
            trace_id=trace_id,
            purposes=purposes,
            scenarios=scenarios,
            roots_only=roots_only,
            children_of=children_of,
            min_records=min_records,
        )
        n = _emit(filtered, chosen_fmt, _render, sink, limit)
        _info(f"{n} session file(s)")
    finally:
        if close:
            sink.close()


# ---------------------------------------------------------------------------
# export-dataset
# ---------------------------------------------------------------------------


@app.command("export-dataset")
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
        typer.Option("--compression", help="Parquet compression (zstd, snappy, gzip, none)."),
    ] = "zstd",
) -> None:
    """Export traces to a HuggingFace-compatible Parquet or JSONL dataset.

    Each row is one session with an OpenAI-compatible ``messages`` array,
    session metadata, and token-usage stats. Parquet is written via DuckDB;
    JSONL is a plain newline-delimited JSON fallback.

    Examples:

        agentm trace export-dataset traces.parquet

        agentm trace export-dataset --scenario chatbot --roots-only out.parquet

        agentm trace export-dataset --session abc123 --session def456 out.jsonl
    """
    from agentm.dataset_export import DatasetExporter

    autoload_dotenv(_trace_cwd(cwd))

    scenarios = set(scenario) if scenario else None
    purposes = set(purpose) if purpose else None
    session_ids = list(session) if session else None

    ch_url = _trace_clickhouse_url(cwd)
    if ch_url is not None:
        exporter = DatasetExporter.from_clickhouse(ch_url)
        _info("Using ClickHouse backend")
    else:
        exporter = DatasetExporter.from_local()
        _info("Using local JSONL backend")

    suffix = output.suffix.lower()
    if suffix == ".jsonl":
        count = exporter.export_jsonl(
            output,
            session_ids=session_ids,
            scenarios=scenarios,
            purposes=purposes,
            roots_only=roots_only,
            include_system_prompt=include_system_prompt,
            include_thinking=include_thinking,
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
            compression=compression,
            limit=limit,
        )

    _info(f"Exported {count} conversation(s) to {output}")


# ---------------------------------------------------------------------------
# Entry point — lazily dispatched by ``agentm.cli.main`` for ``agentm trace``.
# ---------------------------------------------------------------------------


def main() -> None:
    """Console entry shim — ``agentm trace ...`` arrives here."""

    app()


__all__ = ["app", "main"]
