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

Mounted as a builtin subcommand by ``agentm.cli._BUILTIN_SUBCOMMANDS``.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any, Callable, TextIO

import typer

from agentm.core.abi import LogRecord, SessionIdentity, Span, TraceReader

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


def _fail(code: int, kind: str, message: str, fix: str | None = None) -> None:
    """Emit a structured error to stderr and exit with ``code``.

    The single-line JSON payload is parseable by callers that need to
    branch on the failure kind without grepping prose.
    """

    payload: dict[str, Any] = {"kind": kind, "message": message}
    if fix:
        payload["fix"] = fix
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    raise typer.Exit(code)


def _resolve_source(
    file: Path | None,
    session: str | None,
    latest: bool,
    cwd: Path,
) -> Path:
    """Pick exactly one of ``--file`` / ``--session`` / ``--latest``.

    Mutual exclusion is validated explicitly (cli-design §2). Missing
    file → exit 3; unreadable → exit 4.
    """

    chosen = sum([file is not None, session is not None, latest])
    if chosen == 0:
        _fail(
            2,
            "argument",
            "must specify one of --file, --session, or --latest",
            "pass --latest to grab the most recent session under <cwd>/.agentm/observability/",
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
        from agentm.core.runtime.otel_export import resolve_observability_dir

        path = resolve_observability_dir(cwd) / f"{session}.jsonl"
    else:
        from agentm.core.runtime.otel_export import resolve_observability_dir

        obs_dir = resolve_observability_dir(cwd)
        if not obs_dir.is_dir():
            _fail(
                3,
                "not_found",
                f"observability directory not found: {obs_dir}",
                "cd into a run directory or pass --cwd",
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
                "run `agentm` once to produce a session log first",
            )
        path = candidates[0]
    if not path.is_file():
        _fail(3, "not_found", f"trace file not found: {path}")
    if not os.access(path, os.R_OK):
        _fail(4, "permission", f"trace file not readable: {path}")
    return path


def _default_format(output: TextIO) -> str:
    """Pick ndjson off-TTY, text on-TTY (cli-design §7).

    Off-TTY means "something is consuming this programmatically" —
    structured output is the safe default. On-TTY means a human is
    reading, so text wins.
    """

    try:
        return "text" if output.isatty() else "ndjson"
    except (AttributeError, ValueError):
        return "ndjson"


def _parse_where(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``--where K=V`` repeats into a flat ``{key: value}`` dict.

    Values are tried as JSON first (so ``--where n=3`` becomes int 3),
    falling back to raw strings. Equality-only by design — anything
    fancier belongs in jq.
    """

    out: dict[str, Any] = {}
    if not pairs:
        return out
    for raw in pairs:
        if "=" not in raw:
            _fail(2, "argument", f"--where {raw!r}: expected K=V form")
        k, v = raw.split("=", 1)
        if not k:
            _fail(2, "argument", f"--where {raw!r}: empty key")
        try:
            out[k] = json.loads(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def _within_window(
    ts_ns: int | None, since: int | None, until: int | None
) -> bool:
    """Inclusive ``[since, until]`` window on Unix-nanosecond timestamps.

    ``ts_ns is None`` means the record carried no timestamp; we keep it
    rather than silently dropping — the alternative is for ``--since``
    to discard everything header-shaped, which is rarely what callers
    want.
    """

    if ts_ns is None:
        return True
    if since is not None and ts_ns < since:
        return False
    if until is not None and ts_ns > until:
        return False
    return True


def _emit(
    records: Iterable[Any],
    fmt: str,
    text_fn: Callable[[Any], str],
    output: TextIO,
    limit: int | None,
) -> int:
    """Stream ``records`` to ``output`` in ``fmt`` and return the count.

    Format dispatch is centralised here so every verb gets identical
    stdout semantics. ``json`` materialises to a list (small datasets);
    ``ndjson`` and ``text`` stream lazily, so they're safe on big sessions.
    """

    count = 0
    if fmt == "json":
        buf: list[Any] = []
        for r in records:
            if limit is not None and count >= limit:
                break
            buf.append(r)
            count += 1
        json.dump(buf, output, ensure_ascii=False, indent=2)
        output.write("\n")
        return count
    for r in records:
        if limit is not None and count >= limit:
            break
        if fmt == "ndjson":
            output.write(json.dumps(r, ensure_ascii=False))
            output.write("\n")
        else:  # text
            output.write(text_fn(r))
            output.write("\n")
        count += 1
    return count


_ANSI: dict[str, str] = {
    "user": "1;36",       # bold cyan
    "assistant": "1;32",  # bold green
    "system": "1;34",     # bold blue
    "tool_result": "1;33",  # bold yellow
    "tool_error": "1;31",  # bold red
    "thinking": "2",      # dim
    "tool_call": "1;35",  # bold magenta
    "kind": "2",          # dim, for misc block-kind tags
}


def _paint(text: str, key: str, color: bool) -> str:
    """Wrap ``text`` in the ANSI code for ``key`` when ``color`` is enabled."""

    if not color:
        return text
    code = _ANSI.get(key)
    if not code:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _color_enabled(sink: TextIO, no_color: bool) -> bool:
    """Honour ``--no-color``, ``NO_COLOR``, and TTY detection.

    Off-TTY (pipe / redirect / ``--output FILE``) always disables color
    so captured logs never get ANSI noise injected.
    """

    if no_color:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(sink.isatty())
    except (AttributeError, ValueError):
        return False


def _render_tool_args(args: Any, indent: str) -> list[str]:
    """Render tool-call arguments as ``key: value`` when scalars-only.

    Falls back to indented pretty JSON when any value is structured
    (dict / list), because that's the only form that round-trips
    without ambiguity. Single-key + short-string args collapse to one
    line for compactness.
    """

    if not isinstance(args, dict):
        return [f"{indent}{json.dumps(args, ensure_ascii=False)}"]
    if not args:
        return [f"{indent}{{}}"]
    if any(isinstance(v, (dict, list)) for v in args.values()):
        return [
            f"{indent}{line}"
            for line in json.dumps(args, ensure_ascii=False, indent=2).splitlines()
        ]
    out: list[str] = []
    for k, v in args.items():
        v_str = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if "\n" in str(v_str):
            out.append(f"{indent}{k}:")
            for line in str(v_str).splitlines():
                out.append(f"{indent}  {line}")
        else:
            out.append(f"{indent}{k}: {v_str}")
    return out


def _unfold_shell_result(text: str) -> list[str] | None:
    """If ``text`` is JSON ``{exit_code, stdout, stderr, ...}``, expand it.

    The bash atom returns its result as a JSON-encoded string inside a
    text block, so naive rendering shows ``"stdout": "...\\ntotal 5452\\n..."``
    with literal ``\\n`` escapes. Detecting that shape and printing
    real lines makes shell output reviewable. Returns ``None`` when the
    text isn't this shape so the caller can fall back to verbatim.
    """

    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    if not (decoded.keys() & {"exit_code", "stdout", "stderr"}):
        return None
    out: list[str] = []
    if "exit_code" in decoded:
        out.append(f"exit_code={decoded['exit_code']}")
    if decoded.get("timed_out"):
        out.append("timed_out=true")
    for stream in ("stdout", "stderr"):
        body = decoded.get(stream) or ""
        if not body:
            continue
        out.append(f"{stream}:")
        for line in str(body).rstrip("\n").splitlines():
            out.append(f"  {line}")
    return out


def _render_content_blocks(
    content: Any,
    indent: str = "  ",
    *,
    color: bool = False,
    hide_thinking: bool = False,
) -> list[str]:
    """Render a SessionEntry ``payload.content`` list as readable lines.

    ``content`` mirrors Anthropic's content-block taxonomy (``text`` /
    ``thinking`` / ``tool_call`` / ``tool_result`` / ``image`` / ...).
    Returning a list of lines preserves natural newlines — squashing
    multi-line thinking and tool-result bodies into spaces makes args
    and results disappear into the surrounding prose.
    """

    if not isinstance(content, list):
        return []
    out: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = str(block.get("text") or "").strip("\n")
            if not text:
                continue
            for line in text.splitlines() or [""]:
                out.append(f"{indent}{line}")
        elif kind == "thinking":
            if hide_thinking:
                continue
            text = str(block.get("text") or "").strip("\n")
            if not text:
                continue
            tag = _paint("[thinking]", "thinking", color)
            lines = text.splitlines()
            out.append(f"{indent}{tag} {_paint(lines[0], 'thinking', color)}")
            pad = " " * len("[thinking] ")
            for line in lines[1:]:
                out.append(f"{indent}{pad}{_paint(line, 'thinking', color)}")
        elif kind == "tool_call":
            name = block.get("name") or "?"
            tag = _paint(f"[tool_call {name}]", "tool_call", color)
            out.append(f"{indent}{tag}")
            out.extend(_render_tool_args(block.get("arguments"), indent + "  "))
        elif kind == "tool_result":
            # The outer message already carries role=tool_result; don't
            # repeat the tag. Try to unfold the bash-style shell result
            # before falling back to verbatim text.
            if block.get("is_error"):
                out.append(f"{indent}{_paint('[error]', 'tool_error', color)}")
            inner = block.get("content")
            if isinstance(inner, list):
                for sub in inner:
                    if not isinstance(sub, dict):
                        continue
                    sub_kind = sub.get("type")
                    if sub_kind != "text":
                        out.extend(
                            _render_content_blocks(
                                [sub],
                                indent=indent,
                                color=color,
                                hide_thinking=hide_thinking,
                            )
                        )
                        continue
                    sub_text = str(sub.get("text") or "").strip("\n")
                    if not sub_text:
                        continue
                    unfolded = _unfold_shell_result(sub_text)
                    if unfolded is not None:
                        for line in unfolded:
                            out.append(f"{indent}{line}")
                    else:
                        for line in sub_text.splitlines():
                            out.append(f"{indent}{line}")
        elif kind == "image":
            out.append(f"{indent}{_paint('[image]', 'kind', color)}")
        else:
            out.append(f"{indent}{_paint(f'[{kind or chr(63)}]', 'kind', color)}")
    return out


def _span_to_dict(span: Span, unwrap_attrs: bool) -> dict[str, Any]:
    """Reduce a :class:`Span` to a plain JSON-able dict.

    When ``unwrap_attrs`` is set, attribute keys are hoisted to the top
    level (jq-friendly: ``.["gen_ai.tool.name"]`` instead of
    ``.attributes["gen_ai.tool.name"]``). Hoisted keys never collide
    with the core span fields because the OTel semconv prefixes are
    already namespaced.
    """

    base: dict[str, Any] = {
        "name": span.name,
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "start_time_unix_nano": span.start_time_unix_nano,
        "end_time_unix_nano": span.end_time_unix_nano,
    }
    if unwrap_attrs:
        base.update(span.attributes)
    else:
        base["attributes"] = span.attributes
    return base


def _log_to_dict(record: LogRecord, unwrap_attrs: bool) -> dict[str, Any]:
    """Reduce a :class:`LogRecord` to a plain JSON-able dict.

    Mirrors :func:`_span_to_dict` so spans/logs share the same shape
    conventions and TTY/agent rendering stays uniform.
    """

    base: dict[str, Any] = {
        "event_name": record.event_name,
        "body": record.body,
        "trace_id": record.trace_id,
        "span_id": record.span_id,
        "time_unix_nano": record.time_unix_nano,
    }
    if unwrap_attrs:
        base.update(record.attributes)
    else:
        base["attributes"] = record.attributes
    return base


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
        help="Session id; resolves to <cwd>/.agentm/observability/<id>.jsonl.",
    ),
]
LatestOpt = Annotated[
    bool,
    typer.Option(
        "--latest",
        help="Use the most recently modified *.jsonl under <cwd>/.agentm/observability/.",
    ),
]
CwdOpt = Annotated[
    Path,
    typer.Option("--cwd", help="Working directory for --session/--latest resolution."),
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


def _open_output(output: Path | None) -> tuple[TextIO, bool]:
    """Return ``(handle, should_close)`` for the chosen output sink."""

    if output is None:
        return sys.stdout, False
    return output.open("w", encoding="utf-8"), True


def _resolve_format(fmt: str | None, sink: TextIO) -> str:
    """Validate ``--format`` and apply the TTY-aware default."""

    if fmt is None:
        return _default_format(sink)
    if fmt not in {"ndjson", "json", "text"}:
        _fail(2, "argument", f"--format {fmt!r} not in {{ndjson,json,text}}")
    return fmt


def _info(message: str) -> None:
    """Emit a one-line stderr notice (cli-design §3: data → stdout only)."""

    print(message, file=sys.stderr)


# ---------- messages --------------------------------------------------------


@app.command("messages")
def messages_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = Path("."),
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

    The system prompt is rebuilt by the loop on every turn and is not part
    of the session message list, so it does NOT appear here by default.
    Set ``AGENTM_TRACE_SYSTEM_PROMPT=1`` before running the agent to
    persist it; the first turn's prompt will then surface as a synthetic
    ``[system]`` message #0.

    Examples:

      agentm trace messages --latest
      agentm trace messages --latest --role assistant --hide-thinking
      agentm trace messages --session 7b0f... --format ndjson > traj.ndjson
    """

    path = _resolve_source(file, session, latest, cwd)
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

        def _filtered() -> Iterator[dict[str, Any]]:
            reader = TraceReader(path)
            # If the loop recorded the system prompt (opt-in via
            # AGENTM_TRACE_SYSTEM_PROMPT=1), surface the very first
            # occurrence as a synthetic message #0 with role=system.
            # Subsequent turns may mutate it (a real concern for KV
            # prefix-cache invalidation) — use ``trace logs --name
            # agentm.llm.system_prompt`` to inspect per-turn drift.
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

        def _render(entry: dict[str, Any]) -> str:
            payload = entry.get("payload") or {}
            role_str = str(payload.get("role") or entry.get("type") or "?")
            color_key = role_str if role_str in _ANSI else "kind"
            header = _paint(f"[{role_str}]", color_key, color)
            lines = _render_content_blocks(
                payload.get("content"),
                color=color,
                hide_thinking=hide_thinking,
            )
            body = "\n".join([header, *lines]) if lines else header
            # Trailing blank line so consecutive entries are visually
            # separated; ``_emit`` adds one more ``\n`` per record.
            return body + "\n"

        n = _emit(_filtered(), chosen_fmt, _render, sink, limit)
        _info(f"{n} message(s)")
    finally:
        if close:
            sink.close()


# ---------- turns -----------------------------------------------------------


@app.command("turns")
def turns_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = Path("."),
    limit: LimitOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Print per-turn summaries (stop reason, tool counts, tokens).

    Examples:

      agentm trace turns --latest
      agentm trace turns --latest --format ndjson | jq '.input_tokens'
    """

    path = _resolve_source(file, session, latest, cwd)
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)

        def _render(turn: dict[str, Any]) -> str:
            return (
                f"[turn {turn.get('turn_index','?')}] "
                f"stop={turn.get('stop_reason','?')} "
                f"tool_calls={turn.get('tool_call_count',0)} "
                f"errors={turn.get('tool_error_count',0)} "
                f"in={turn.get('input_tokens',0)} "
                f"out={turn.get('output_tokens',0)}"
            )

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
    cwd: CwdOpt = Path("."),
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Token economics summary: input, cache hit, output, cost estimate.

    Examples:

      agentm trace usage --latest
      agentm trace usage --file path/to/session.jsonl --format ndjson
    """

    path = _resolve_source(file, session, latest, cwd)
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        records = TraceReader(path).load_turn_summaries()
        if not records:
            _info("no turns found")
            return

        total_input = sum(r.get("input_tokens", 0) for r in records)
        total_output = sum(r.get("output_tokens", 0) for r in records)
        cache_read = sum(r.get("cache_read", 0) for r in records)
        cache_write = sum(r.get("cache_write", 0) for r in records)
        non_cached = total_input - cache_read
        hit_pct = (cache_read / total_input * 100) if total_input else 0.0

        summary: dict[str, Any] = {
            "turns": len(records),
            "input_tokens": total_input,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "non_cached_input": non_cached,
            "cache_hit_rate": round(hit_pct, 1),
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

        if chosen_fmt == "ndjson":
            sink.write(json.dumps(summary) + "\n")
        else:
            sink.write(
                f"turns:            {summary['turns']}\n"
                f"input tokens:     {total_input:>12,}\n"
                f"  cache read:     {cache_read:>12,}  ({hit_pct:.1f}%)\n"
                f"  cache write:    {cache_write:>12,}\n"
                f"  non-cached:     {non_cached:>12,}\n"
                f"output tokens:    {total_output:>12,}\n"
                f"total tokens:     {total_input + total_output:>12,}\n"
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
    cwd: CwdOpt = Path("."),
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

    path = _resolve_source(file, session, latest, cwd)
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        where_filters = _parse_where(where)
        wanted_models = set(model or [])

        def _filtered() -> Iterator[dict[str, Any]]:
            for span in TraceReader(path).chat_calls():
                if not _within_window(span.start_time_unix_nano, since, until):
                    continue
                if any(
                    span.attributes.get(k) != v for k, v in where_filters.items()
                ):
                    continue
                if wanted_models:
                    m = span.attributes.get("gen_ai.request.model")
                    if m not in wanted_models:
                        continue
                yield _span_to_dict(span, unwrap_attrs)

        def _render(d: dict[str, Any]) -> str:
            attrs = d if "gen_ai.request.model" in d else d.get("attributes", {})
            duration_ns = attrs.get("agentm.llm.duration_ns")
            duration = (
                f"{duration_ns / 1e9:.2f}s" if isinstance(duration_ns, int) else "?"
            )
            return (
                f"[chat] model={attrs.get('gen_ai.request.model','?')} "
                f"turn={attrs.get('agentm.turn.index','?')} "
                f"messages={attrs.get('agentm.llm.message_count',0)} "
                f"duration={duration}"
            )

        n = _emit(_filtered(), chosen_fmt, _render, sink, limit)
        _info(f"{n} chat call(s)")
    finally:
        if close:
            sink.close()


# ---------- tools -----------------------------------------------------------


@app.command("tools")
def tools_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = Path("."),
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

    path = _resolve_source(file, session, latest, cwd)
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        where_filters = _parse_where(where)
        wanted = set(tool or [])

        def _filtered() -> Iterator[dict[str, Any]]:
            for span, args_log, result_log in TraceReader(path).tool_calls():
                tool_name = span.attributes.get(
                    "gen_ai.tool.name"
                ) or span.name.removeprefix("execute_tool ").strip()
                if wanted and tool_name not in wanted:
                    continue
                if not _within_window(span.start_time_unix_nano, since, until):
                    continue
                if any(
                    span.attributes.get(k) != v for k, v in where_filters.items()
                ):
                    continue
                args_payload: Any = (
                    args_log.body if args_log is not None else None
                )
                if args_payload is None:
                    raw = span.attributes.get("gen_ai.tool.call.arguments")
                    if isinstance(raw, str):
                        try:
                            args_payload = json.loads(raw)
                        except (TypeError, ValueError):
                            args_payload = raw
                result_payload: Any = (
                    result_log.body if result_log is not None else None
                )
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

        def _render(d: dict[str, Any]) -> str:
            args_repr = json.dumps(d["args"], ensure_ascii=False)
            result_repr = json.dumps(d["result"], ensure_ascii=False)
            return (
                f"[tool {d['tool']}] args={args_repr}\n"
                f"  → result={result_repr}"
            )

        n = _emit(_filtered(), chosen_fmt, _render, sink, limit)
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
    cwd: CwdOpt = Path("."),
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
    path = _resolve_source(file, session, latest, cwd)
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        reader = TraceReader(path)
        payload: dict[str, Any] = {}
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
    cwd: CwdOpt = Path("."),
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

    path = _resolve_source(file, session, latest, cwd)
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        where_filters = _parse_where(where)

        def _filtered() -> Iterator[dict[str, Any]]:
            for span in TraceReader(path).iter_spans(
                name=name, attribute_filters=where_filters or None
            ):
                if name_prefix is not None and not span.name.startswith(name_prefix):
                    continue
                if not _within_window(span.start_time_unix_nano, since, until):
                    continue
                yield _span_to_dict(span, unwrap_attrs)

        def _render(d: dict[str, Any]) -> str:
            return f"[span] {d['name']}"

        n = _emit(_filtered(), chosen_fmt, _render, sink, limit)
        _info(f"{n} span(s)")
    finally:
        if close:
            sink.close()


@app.command("logs")
def logs_cmd(
    file: FileOpt = None,
    session: SessionOpt = None,
    latest: LatestOpt = False,
    cwd: CwdOpt = Path("."),
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

    path = _resolve_source(file, session, latest, cwd)
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        where_filters = _parse_where(where)

        def _filtered() -> Iterator[dict[str, Any]]:
            for rec in TraceReader(path).iter_log_records(
                name=name, attribute_filters=where_filters or None
            ):
                if name_prefix is not None and not rec.event_name.startswith(
                    name_prefix
                ):
                    continue
                if not _within_window(rec.time_unix_nano, since, until):
                    continue
                yield _log_to_dict(rec, unwrap_attrs)

        def _render(d: dict[str, Any]) -> str:
            return f"[log] {d['event_name']}"

        n = _emit(_filtered(), chosen_fmt, _render, sink, limit)
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
    cwd: CwdOpt = Path("."),
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

    path = _resolve_source(file, session, latest, cwd)
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)
        logs: Counter[str] = Counter()
        spans: Counter[str] = Counter()
        for item in TraceReader(path).iter_all():
            if isinstance(item, Span):
                spans[item.name] += 1
            else:
                logs[item.event_name] += 1
        logs_sorted: dict[str, int] = dict(
            sorted(logs.items(), key=lambda kv: -kv[1])
        )
        spans_sorted: dict[str, int] = dict(
            sorted(spans.items(), key=lambda kv: -kv[1])
        )
        summary: dict[str, Any] = {
            "file": str(path),
            "logs": logs_sorted,
            "spans": spans_sorted,
            "log_total": sum(logs.values()),
            "span_total": sum(spans.values()),
        }
        if chosen_fmt == "text":
            sink.write(f"file: {summary['file']}\n")
            sink.write(f"logs ({summary['log_total']}):\n")
            for k, v in logs_sorted.items():
                sink.write(f"  {v:>5}  {k}\n")
            sink.write(f"spans ({summary['span_total']}):\n")
            for k, v in spans_sorted.items():
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


def _count_lines(path: Path) -> int | None:
    """Cheap line count for a session file (proxy for record volume).

    Returns ``None`` if the file can't be read — the identity fields are
    the priority, so a missing count never blocks a row.
    """

    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


@app.command("index")
def index_cmd(
    cwd: CwdOpt = Path("."),
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="Observability directory to scan (default <cwd>/.agentm/observability/).",
        ),
    ] = None,
    limit: LimitOpt = None,
    fmt: FormatOpt = None,
    out: OutputOpt = None,
) -> None:
    """Map every session file to its trace-tree identity (one row per file).

    A logical "trace" spans many JSONL files — one root session plus N
    spawned children (extractor / auditor / ...). This is the only
    directory-granular verb: it scans the observability dir and emits one
    identity row per ``agentm.session.start`` file, so jq + shell can go
    from a ``trace_id`` to its session files. Files with no session.start
    record are skipped. Filtering is the consumer's job — pipe through jq.

    Each row: {path, trace_id, session_id, parent_session_id, purpose,
    scenario, records}.

    Examples:

      agentm trace index --format ndjson | jq 'select(.trace_id=="…")'
      agentm trace index --dir /path/to/.agentm/observability --format text
    """

    if directory is not None:
        obs_dir = directory
    else:
        from agentm.core.runtime.otel_export import resolve_observability_dir

        obs_dir = resolve_observability_dir(cwd)
    if not obs_dir.is_dir():
        _fail(
            3,
            "not_found",
            f"observability directory not found: {obs_dir}",
            "pass --dir, or cd into a run directory / pass --cwd",
        )
    sink, close = _open_output(out)
    try:
        chosen_fmt = _resolve_format(fmt, sink)

        def _rows() -> Iterator[dict[str, Any]]:
            for path in sorted(obs_dir.glob("*.jsonl")):
                if not path.is_file():
                    continue
                identity: SessionIdentity | None = TraceReader(
                    path
                ).first_session_identity()
                if identity is None:
                    continue
                yield {
                    "path": str(path),
                    "trace_id": identity.trace_id,
                    "session_id": identity.session_id,
                    "parent_session_id": identity.parent_session_id,
                    "purpose": identity.purpose,
                    "scenario": identity.scenario,
                    "records": _count_lines(path),
                }

        def _render(d: dict[str, Any]) -> str:
            return (
                f"[session {d.get('session_id') or '?'}] "
                f"trace={d.get('trace_id') or '?'} "
                f"parent={d.get('parent_session_id') or '-'} "
                f"purpose={d.get('purpose') or '-'} "
                f"records={d.get('records') if d.get('records') is not None else '?'} "
                f"{d['path']}"
            )

        n = _emit(_rows(), chosen_fmt, _render, sink, limit)
        _info(f"{n} session file(s)")
    finally:
        if close:
            sink.close()


# ---------------------------------------------------------------------------
# Entry point — wired into ``agentm.cli._BUILTIN_SUBCOMMANDS``.
# ---------------------------------------------------------------------------


def main() -> None:
    """Console entry shim — ``agentm trace ...`` arrives here."""

    app()


__all__ = ["app", "main"]
