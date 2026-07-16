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
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Callable, TextIO

import typer
from loguru import logger

from agentm.cli.trace.backend import (
    clickhouse as _ch,
    clickhouse_session as _ch_session,
    trace_clickhouse_url as _trace_clickhouse_url,
    trace_cwd as _trace_cwd,
)
from agentm.cli.trace.dataset import export_dataset_cmd
from agentm.cli.trace.format import (
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
from agentm.cli.trace.index import index_cmd
from agentm.core.abi import Span, TraceReader

app = typer.Typer(
    name="trace",
    help=(
        "Query an OTLP/JSON session log written by the observability atom. "
        "Use one of --file / --session / --latest to pick a file."
    ),
    no_args_is_help=True,
    add_completion=False,
)
app.command("index")(index_cmd)
app.command("export-dataset")(export_dataset_cmd)


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
            _print_session_stats(sink, summary.get("session"))
            _print_tool_stats(sink, summary.get("tools"))
            _print_turn_stats(sink, summary.get("turns"))
            _print_context_snapshots(sink, summary.get("context_snapshots"))
            _print_context_composition(
                sink, summary.get("tools"), summary.get("turns")
            )
        elif chosen_fmt == "json":
            json.dump(summary, sink, ensure_ascii=False, indent=2)
            sink.write("\n")
        else:  # ndjson
            sink.write(json.dumps(summary, ensure_ascii=False))
            sink.write("\n")
    finally:
        if close:
            sink.close()


@app.command("doctor")
def doctor_cmd(
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Check data-quality invariants for a session (ClickHouse backend).

    Audits the raw tables: record duplication, session lifecycle counts,
    turn_index contiguity, turn span/summary pairing, tool-span parentage.
    Exits 1 when any error-severity violation is found — run it before
    trusting aggregates from an unfamiliar or freshly ingested session.

    Examples:

      agentm trace doctor --latest
      agentm trace doctor --session <sid> --format ndjson
    """
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)

        ch = _ch_session(None, session, latest, cwd)
        if not ch:
            typer.echo(
                "Error: doctor requires the ClickHouse backend "
                "(set AGENTM_CLICKHOUSE_URL / OTEL_EXPORTER_OTLP_ENDPOINT "
                "and pass --session/--latest).",
                err=True,
            )
            raise typer.Exit(2)
        url, sid = ch
        violations = _ch().doctor(url, sid)

        if chosen_fmt == "text":
            if not violations:
                sink.write(f"session {sid}: all invariants hold\n")
            else:
                sink.write(f"session {sid}: {len(violations)} violation(s)\n")
                for v in violations:
                    sink.write(
                        f"  [{v['severity'].upper():>7}] {v['check']}: "
                        f"expected {v['expected']}, got {v['actual']} — {v['detail']}\n"
                    )
        else:  # json / ndjson
            payload = {"session_id": sid, "violations": violations}
            if chosen_fmt == "json":
                json.dump(payload, sink, ensure_ascii=False, indent=2)
            else:
                sink.write(json.dumps(payload, ensure_ascii=False))
            sink.write("\n")
    finally:
        if close:
            sink.close()
    if any(v["severity"] == "error" for v in violations):
        raise typer.Exit(1)


@app.command("scan")
def scan_cmd(
    window: Annotated[int, typer.Option("--window-hours")] = 48,
    min_cohort: Annotated[int, typer.Option("--min-cohort")] = 5,
    limit: Annotated[int, typer.Option("--limit")] = 50,
    cwd: CwdOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Flag outlier sessions against their (scenario, task_class) cohort.

    Sessions whose turns / tokens / wall time / peak context / tool-error
    rate exceed the cohort p95 (and 1.5x median) are attribution entry
    points: something in the framework, the model, or the task itself made
    them expensive. Requires the ClickHouse backend.

    Examples:

      agentm trace scan --window-hours 24
      agentm trace scan --format ndjson | jq '.findings[]'
    """
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        url = _trace_clickhouse_url(cwd)
        if url is None:
            typer.echo(
                "Error: scan requires the ClickHouse backend "
                "(set AGENTM_CLICKHOUSE_URL / OTEL_EXPORTER_OTLP_ENDPOINT).",
                err=True,
            )
            raise typer.Exit(2)
        report = _ch().scan(
            url, window_hours=window, min_cohort=min_cohort, limit=limit,
        )

        if chosen_fmt == "text":
            sink.write(
                f"window: {report['window_hours']}h | "
                f"sessions: {report['sessions']} | "
                f"findings: {len(report['findings'])}\n"
            )
            sink.write("cohorts:\n")
            for cohort_name, n in report["cohorts"].items():
                sink.write(f"  {n:>6}  {cohort_name}\n")
            if report["findings"]:
                sink.write("findings (value vs cohort p50/p95):\n")
                for f in report["findings"]:
                    sink.write(
                        f"  {f['session_id']}  {f['metric']}={f['value']} "
                        f"(p50={f['p50']}, p95={f['p95']}, "
                        f"x{f['ratio_to_p50']}, n={f['cohort_size']}) "
                        f"{f['scenario']}/{f['task_class'] or '-'}\n"
                    )
        elif chosen_fmt == "json":
            json.dump(report, sink, ensure_ascii=False, indent=2)
            sink.write("\n")
        else:
            sink.write(json.dumps(report, ensure_ascii=False))
            sink.write("\n")
    finally:
        if close:
            sink.close()


def _print_session_stats(sink: Any, session: dict[str, Any] | None) -> None:
    if not session:
        return
    sink.write("session:\n")
    if "duration_s" in session:
        mins = session["duration_s"] // 60
        secs = session["duration_s"] % 60
        sink.write(f"  duration:            {mins}m{secs}s\n")
        sink.write(f"  start:               {session.get('start_time', '')}\n")
        sink.write(f"  end:                 {session.get('end_time', '')}\n")
    children = session.get("child_sessions")
    if children:
        total = sum(children.values())
        sink.write(f"  child_sessions:      {total}\n")
        for purpose, cnt in children.items():
            sink.write(f"    {cnt:>4}  {purpose}\n")
    stops = session.get("stop_reasons")
    if stops:
        sink.write("  stop_reasons:\n")
        for reason, cnt in stops.items():
            sink.write(f"    {cnt:>4}  {reason}\n")


def _print_tool_stats(sink: Any, tools: dict[str, Any] | None) -> None:
    if not tools:
        return
    sink.write("tools:\n")
    sink.write(
        f"  {'name':<22} {'calls':>5} {'err':>4} {'avg_ch':>7} "
        f"{'max_ch':>8} {'total_ch':>10} {'avg_ms':>7} {'p95_ms':>7} {'max_ms':>7}\n"
    )
    sink.write(f"  {'-' * 83}\n")
    for name, s in tools.items():
        sink.write(
            f"  {name:<22} {s['calls']:>5} {s.get('errors',0):>4} "
            f"{s['avg_result_chars']:>7} {s['max_result_chars']:>8} "
            f"{s['total_result_chars']:>10} {s['avg_duration_ms']:>7} "
            f"{s['p95_duration_ms']:>7} {s['max_duration_ms']:>7}\n"
        )


def _print_turn_stats(sink: Any, turns: dict[str, Any] | None) -> None:
    if not turns:
        return
    sink.write("turns:\n")
    sink.write(f"  total_turns:         {turns.get('total_turns', 0)}\n")
    sink.write(f"  total_tool_calls:    {turns.get('total_tool_calls', 0)}\n")
    sink.write(f"  total_tool_errors:   {turns.get('total_tool_errors', 0)}\n")
    sink.write(f"  total_input_tokens:  {turns.get('total_input_tokens', 0)}\n")
    sink.write(f"  total_output_tokens: {turns.get('total_output_tokens', 0)}\n")
    sink.write(f"  total_cache_read:    {turns.get('total_cache_read', 0)}\n")
    sink.write(
        f"  context_tokens:      "
        f"min={turns.get('min_input_tokens', 0)} "
        f"avg={turns.get('avg_input_tokens', 0)} "
        f"max={turns.get('max_input_tokens', 0)}\n"
    )


def _print_context_snapshots(
    sink: Any, snapshots: list[dict[str, Any]] | None
) -> None:
    if not snapshots:
        return
    for snap in snapshots:
        total = snap.get("total_chars", 0)
        if total <= 0:
            continue
        label = snap.get("label", "snapshot")
        turn = snap.get("turn_index", "?")
        in_tok = snap.get("input_tokens", "?")
        sink.write(f"context_snapshot [{label}] turn={turn} input_tokens={in_tok}:\n")

        for category in ("system", "user", "assistant", "tool_result"):
            chars = snap.get(category, 0)
            pct = chars * 100 / total if total else 0
            tokens = chars // 4
            sink.write(f"  {category:<20} {tokens:>8} tokens  {pct:>5.1f}%\n")

        by_name = snap.get("tool_result_by_name")
        if by_name:
            sink.write("  tool_result breakdown:\n")
            for name, chars in by_name.items():
                pct = chars * 100 / total if total else 0
                tokens = chars // 4
                if pct >= 0.5:
                    sink.write(f"    {name:<18} {tokens:>8} tokens  {pct:>5.1f}%\n")


def _print_context_composition(
    sink: Any,
    tools: dict[str, Any] | None,
    turns: dict[str, Any] | None,
) -> None:
    if not tools or not turns:
        return
    total_input = turns.get("total_input_tokens", 0)
    total_output = turns.get("total_output_tokens", 0)
    if total_input <= 0:
        return

    sink.write("context_composition (estimated):\n")

    tool_items: list[tuple[str, int]] = []
    total_tool_chars = 0
    for name, s in tools.items():
        chars = s.get("total_result_chars", 0)
        total_tool_chars += chars
        tool_items.append((name, chars))

    total_tool_tokens = total_tool_chars // 4
    assistant_tokens = total_output
    other_tokens = max(0, total_input - total_tool_tokens - assistant_tokens)

    sink.write(
        f"  {'category':<30} {'~tokens':>10} {'%':>6}\n"
    )
    sink.write(f"  {'-' * 50}\n")

    for name, chars in sorted(tool_items, key=lambda x: -x[1]):
        tokens = chars // 4
        pct = tokens * 100 / total_input if total_input else 0
        if pct >= 0.5:
            sink.write(f"  tool:{name:<24} {tokens:>10} {pct:>5.1f}%\n")

    sink.write(
        f"  {'tool_results (total)':<30} {total_tool_tokens:>10} "
        f"{total_tool_tokens * 100 / total_input:>5.1f}%\n"
    )
    sink.write(
        f"  {'assistant_output':<30} {assistant_tokens:>10} "
        f"{assistant_tokens * 100 / total_input:>5.1f}%\n"
    )
    sink.write(
        f"  {'system+user+overhead':<30} {other_tokens:>10} "
        f"{other_tokens * 100 / total_input:>5.1f}%\n"
    )
    sink.write(
        f"  {'total_input':<30} {total_input:>10}\n"
    )


# ---------------------------------------------------------------------------
# Entry point — lazily dispatched by ``agentm.cli.main`` for ``agentm trace``.
# ---------------------------------------------------------------------------


def main() -> None:
    """Console entry shim — ``agentm trace ...`` arrives here."""

    app()


__all__ = ["app", "main"]
