"""Output formatting helpers for ``agentm trace``."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TextIO

import typer

from agentm.core.abi import LogRecord, Span


def _fail(code: int, kind: str, message: str, fix: str | None = None) -> None:
    """Emit a structured error to stderr and exit with ``code``."""

    payload: dict[str, Any] = {"kind": kind, "message": message}
    if fix:
        payload["fix"] = fix
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    raise typer.Exit(code)


def _default_format(output: TextIO) -> str:
    """Pick ndjson off-TTY, text on-TTY (cli-design §7)."""

    try:
        return "text" if output.isatty() else "ndjson"
    except (AttributeError, ValueError):
        return "ndjson"


def _parse_where(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``--where K=V`` repeats into a flat ``{key: value}`` dict."""

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
    """Inclusive ``[since, until]`` window on Unix-nanosecond timestamps."""

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
    """Stream ``records`` to ``output`` in ``fmt`` and return the count."""

    count = 0
    if fmt == "json":
        buf: list[Any] = []
        for record in records:
            if limit is not None and count >= limit:
                break
            buf.append(record)
            count += 1
        json.dump(buf, output, ensure_ascii=False, indent=2)
        output.write("\n")
        return count
    for record in records:
        if limit is not None and count >= limit:
            break
        if fmt == "ndjson":
            output.write(json.dumps(record, ensure_ascii=False))
            output.write("\n")
        else:
            output.write(text_fn(record))
            output.write("\n")
        count += 1
    return count


_ANSI: dict[str, str] = {
    "user": "1;36",
    "assistant": "1;32",
    "system": "1;34",
    "tool_result": "1;33",
    "tool_error": "1;31",
    "thinking": "2",
    "tool_call": "1;35",
    "kind": "2",
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
    """Honour ``--no-color``, ``NO_COLOR``, and TTY detection."""

    if no_color:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(sink.isatty())
    except (AttributeError, ValueError):
        return False


def _render_tool_args(args: Any, indent: str) -> list[str]:
    """Render tool-call arguments as ``key: value`` when scalars-only."""

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
    for key, value in args.items():
        value_str = value if isinstance(value, str) else json.dumps(
            value,
            ensure_ascii=False,
        )
        if "\n" in str(value_str):
            out.append(f"{indent}{key}:")
            for line in str(value_str).splitlines():
                out.append(f"{indent}  {line}")
        else:
            out.append(f"{indent}{key}: {value_str}")
    return out


def _unfold_shell_result(text: str) -> list[str] | None:
    """If ``text`` is JSON ``{exit_code, stdout, stderr, ...}``, expand it."""

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
    """Render a SessionEntry ``payload.content`` list as readable lines."""

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
    """Reduce a :class:`Span` to a plain JSON-able dict."""

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
    """Reduce a :class:`LogRecord` to a plain JSON-able dict."""

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
