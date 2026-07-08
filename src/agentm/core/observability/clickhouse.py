"""ClickHouse query backend for ``agentm trace``.

SQL templates against the OTel Collector's ``otel_logs`` / ``otel_traces``
tables.  Uses ``urllib.request`` — zero extra dependencies.  Activated when
``AGENTM_CLICKHOUSE_URL`` is set or ClickHouse is reachable on
``http://localhost:8123``.

Every reader queries ``(SELECT DISTINCT * FROM <table>)`` instead of the raw
table: rows ingested before the single-sink fix (89f20ea7) exist twice as
byte-identical duplicates (core auto-attach + otlp_export floor atom both
exported every record), and DISTINCT makes historical sessions read clean.
ClickHouse pushes the outer WHERE into the subquery, so scans stay
session-scoped.  ``doctor`` is the one deliberate exception — it audits the
raw tables precisely to catch new duplication bugs that DISTINCT would hide.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, Iterator

from loguru import logger


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def get_url() -> str | None:
    """Return ClickHouse HTTP URL if available, else ``None``."""
    url = os.environ.get("AGENTM_CLICKHOUSE_URL")
    if url:
        return url.rstrip("/")
    try:
        with urllib.request.urlopen(
            urllib.request.Request("http://localhost:8123/ping"),
            timeout=1,
        ) as r:
            if r.status == 200:
                return "http://localhost:8123"
    except (urllib.error.URLError, OSError) as exc:
        # No local ClickHouse reachable — expected when running without the
        # observability stack; caller falls back to the JSONL backend.
        logger.debug("clickhouse: localhost:8123 ping failed, no CH backend: {}", exc)
    return None


def _encode_param(value: str | list[str]) -> str:
    """Encode a bind-parameter value for the CH HTTP interface.

    Lists become an ``Array(String)`` literal for ``{name:Array(String)}``
    placeholders. Binding happens server-side — the SQL text never contains
    caller data, so this encoding is a wire format, not SQL escaping.
    """
    if isinstance(value, list):
        quoted = (
            "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'" for v in value
        )
        return "[" + ",".join(quoted) + "]"
    return value


def query(
    url: str,
    sql: str,
    params: Mapping[str, str | list[str]] | None = None,
    *,
    database: str = "otel",
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Execute SQL against ClickHouse HTTP interface, return rows as dicts."""
    qp: dict[str, str] = {"database": database}
    if params:
        for k, v in params.items():
            qp[f"param_{k}"] = _encode_param(v)
    full_url = f"{url}/?{urllib.parse.urlencode(qp)}"
    body = f"{sql}\nFORMAT JSONEachRow"
    req = urllib.request.Request(
        full_url, data=body.encode("utf-8"), method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return [json.loads(line) for line in resp if line.strip()]


def _int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Session resolution
# ---------------------------------------------------------------------------


def resolve_session(
    url: str, session: str | None, latest: bool, cwd: str | None = None,
) -> str | None:
    """Map ``--session``/``--latest`` to a concrete session_id."""
    if session:
        return session
    if latest:
        return most_recent_session_id(url, cwd)
    return None


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def index(
    url: str,
    *,
    trace_id: str | None = None,
    purposes: set[str] | None = None,
    scenarios: set[str] | None = None,
    roots_only: bool = False,
    children_of: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Session topology — one row per ``session.start``."""
    where = ["EventName = 'agentm.session.start'"]
    params: dict[str, str] = {}
    if trace_id is not None:
        where.append("LogAttributes['agentm.session.root_id'] = {tid:String}")
        params["tid"] = trace_id
    if roots_only:
        where.append("LogAttributes['agentm.session.parent_id'] = ''")
    if children_of is not None:
        where.append("LogAttributes['agentm.session.parent_id'] = {parent:String}")
        params["parent"] = children_of
    sql = (
        "SELECT "
        "  LogAttributes['agentm.session.id']        AS session_id, "
        "  LogAttributes['agentm.session.root_id']    AS trace_id, "
        "  LogAttributes['agentm.session.parent_id']  AS parent_session_id, "
        "  LogAttributes['agentm.session.purpose']    AS purpose, "
        "  LogAttributes['agentm.session.scenario']   AS scenario "
        f"FROM (SELECT DISTINCT * FROM otel_logs) WHERE {' AND '.join(where)} "
        "ORDER BY Timestamp"
    )
    for r in query(url, sql, params):
        purpose = r["purpose"] or None
        scenario = r["scenario"] or None
        if purposes and purpose not in purposes:
            continue
        if scenarios and scenario not in scenarios:
            continue
        yield {
            "session_id": r["session_id"] or None,
            "trace_id": r["trace_id"] or None,
            "parent_session_id": r["parent_session_id"] or None,
            "purpose": purpose,
            "scenario": scenario,
            "records": None,
        }


# ---------------------------------------------------------------------------
# session state (for resume)
# ---------------------------------------------------------------------------


def session_header(url: str, sid: str) -> dict[str, Any] | None:
    """Fetch the latest ``agentm.session.header`` body for *sid*."""
    rows = query(
        url,
        "SELECT Body FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE EventName = 'agentm.session.header' "
        "  AND LogAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp DESC LIMIT 1",
        params={"sid": sid},
    )
    if not rows:
        return None
    body = parse_body(rows[0].get("Body"))
    return body if isinstance(body, dict) else None


def session_entries(url: str, sid: str) -> list[dict[str, Any]]:
    """Fetch all ``agentm.message.appended`` bodies for *sid*, ordered."""
    result: list[dict[str, Any]] = []
    for r in query(
        url,
        "SELECT Body FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE EventName = 'agentm.message.appended' "
        "  AND LogAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp",
        params={"sid": sid},
    ):
        body = parse_body(r.get("Body"))
        if isinstance(body, dict):
            result.append(body)
    return result


def most_recent_session_id(url: str, cwd: str | None = None) -> str | None:
    """Return the session_id of the most recently started session."""
    where = ["EventName = 'agentm.session.start'"]
    params: dict[str, Any] = {}
    if cwd:
        where.append("LogAttributes['agentm.session.cwd'] = {cwd:String}")
        params["cwd"] = cwd
    rows = query(
        url,
        "SELECT LogAttributes['agentm.session.id'] AS sid "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY Timestamp DESC LIMIT 1",
        params=params,
    )
    if not rows and cwd:
        # Backward compatibility for traces written before session.cwd was
        # stamped as an attribute.
        return most_recent_session_id(url, None)
    return rows[0]["sid"] if rows else None


def recent_sessions(
    url: str,
    *,
    cwd: str | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Return the most recent root sessions, newest first.

    Each dict: ``{session_id, scenario, created_at}`` where ``created_at``
    is an ISO-8601 timestamp string.
    """
    where = [
        "EventName = 'agentm.session.start'",
        "LogAttributes['agentm.session.parent_id'] = ''",
    ]
    params: dict[str, str] = {"lim": str(limit)}
    if cwd:
        where.append("LogAttributes['agentm.session.cwd'] = {cwd:String}")
        params["cwd"] = cwd
    rows = query(
        url,
        "SELECT "
        "  LogAttributes['agentm.session.id']       AS session_id, "
        "  LogAttributes['agentm.session.scenario']  AS scenario, "
        "  Timestamp                                  AS ts "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY Timestamp DESC "
        "LIMIT {lim:UInt32}",
        params=params,
    )
    result: list[dict[str, Any]] = []
    for r in rows:
        sid = r.get("session_id") or ""
        if not sid:
            continue
        result.append({
            "session_id": sid,
            "scenario": r.get("scenario") or "",
            "created_at": r.get("ts") or "",
        })
    return result


def first_user_message(url: str, sid: str) -> str:
    """Best-effort first user message from a session (for title display)."""
    rows = query(
        url,
        "SELECT Body FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE EventName = 'agentm.message.appended' "
        "  AND LogAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp LIMIT 5",
        params={"sid": sid},
    )
    for r in rows:
        body = parse_body(r.get("Body"))
        if not isinstance(body, dict):
            continue
        raw_payload = body.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else body
        if payload.get("role") != "user":
            continue
        content = payload.get("content", "")
        if isinstance(content, str):
            return content[:120]
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str):
                        return text[:120]
    return ""


def message_stats(url: str, sid: str) -> dict[str, int]:
    """Return lightweight transcript stats for a session."""
    rows = query(
        url,
        "SELECT count() AS messages, sum(length(toString(Body))) AS bytes "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE EventName = 'agentm.message.appended' "
        "  AND LogAttributes['agentm.session.id'] = {sid:String}",
        params={"sid": sid},
    )
    if not rows:
        return {"messages": 0, "bytes": 0}
    row = rows[0]
    return {
        "messages": int(row.get("messages") or 0),
        "bytes": int(row.get("bytes") or 0),
    }


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


def messages(
    url: str,
    sid: str,
    *,
    roles: set[str] | None = None,
    types: set[str] | None = None,
    include_system_prompt: bool = True,
) -> Iterator[dict[str, Any]]:
    """Conversation trajectory — same dict shape as ``TraceReader.load_messages``."""
    if include_system_prompt:
        if (not types or "message" in types) and (not roles or "system" in roles):
            sys_rows = query(
                url,
                "SELECT Body FROM (SELECT DISTINCT * FROM otel_logs) "
                "WHERE EventName = 'agentm.llm.system_prompt' "
                "  AND LogAttributes['agentm.session.id'] = {sid:String} "
                "ORDER BY Timestamp LIMIT 1",
                params={"sid": sid},
            )
            if sys_rows:
                body = parse_body(sys_rows[0].get("Body"))
                text = body.get("text", "") if isinstance(body, dict) else ""
                yield {
                    "type": "message",
                    "id": "system-prompt-turn0",
                    "parent_id": None,
                    "timestamp": 0,
                    "payload": {
                        "role": "system",
                        "content": [{"type": "text", "text": text}],
                    },
                }

    for r in query(
        url,
        "SELECT Body FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE EventName = 'agentm.message.appended' "
        "  AND LogAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp",
        params={"sid": sid},
    ):
        body = parse_body(r.get("Body"))
        if not isinstance(body, dict):
            continue
        if types and body.get("type") not in types:
            continue
        payload = body.get("payload") or {}
        if roles and payload.get("role") not in roles:
            continue
        yield body


# ---------------------------------------------------------------------------
# turns / usage
# ---------------------------------------------------------------------------


def turns(url: str, sid: str) -> Iterator[dict[str, Any]]:
    """Per-turn summaries — bodies of ``agentm.turn.summary`` log records."""
    for r in query(
        url,
        "SELECT Body FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE EventName = 'agentm.turn.summary' "
        "  AND LogAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp",
        params={"sid": sid},
    ):
        body = parse_body(r.get("Body"))
        if isinstance(body, dict):
            yield body


def usage(url: str, sid: str) -> dict[str, Any] | None:
    """Token economics — derived from turn summaries."""
    records = list(turns(url, sid))
    if not records:
        return None
    total_in = sum(r.get("input_tokens", 0) for r in records)
    total_out = sum(r.get("output_tokens", 0) for r in records)
    cache_r = sum(r.get("cache_read", 0) for r in records)
    cache_w = sum(r.get("cache_write", 0) for r in records)
    non_cached = total_in - cache_r
    hit_pct = (cache_r / total_in * 100) if total_in else 0.0
    return {
        "turns": len(records),
        "input_tokens": total_in,
        "cache_read": cache_r,
        "cache_write": cache_w,
        "non_cached_input": non_cached,
        "cache_hit_rate": round(hit_pct, 1),
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
    }


# ---------------------------------------------------------------------------
# chats
# ---------------------------------------------------------------------------


def chats(url: str, sid: str) -> Iterator[dict[str, Any]]:
    """LLM calls — one row per ``chat <model>`` span."""
    for r in query(
        url,
        "SELECT SpanName, SpanId, SpanAttributes, Duration, "
        "  toUnixTimestamp64Nano(Timestamp) AS start_ns "
        "FROM (SELECT DISTINCT * FROM otel_traces) "
        "WHERE startsWith(SpanName, 'chat ') "
        "  AND SpanAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp",
        params={"sid": sid},
    ):
        attrs = r.get("SpanAttributes", {})
        start_ns = _int(r.get("start_ns"))
        duration = _int(r.get("Duration"))
        yield {
            "name": r["SpanName"],
            "span_id": r.get("SpanId"),
            "trace_id": None,
            "parent_span_id": None,
            "start_time_unix_nano": start_ns or None,
            "end_time_unix_nano": (start_ns + duration) if start_ns and duration else None,
            "attributes": attrs,
        }


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def tools(url: str, sid: str) -> Iterator[dict[str, Any]]:
    """Tool calls — from ``execute_tool`` spans + span attributes."""
    for r in query(
        url,
        "SELECT SpanName, SpanId, SpanAttributes, Duration, "
        "  toUnixTimestamp64Nano(Timestamp) AS start_ns "
        "FROM (SELECT DISTINCT * FROM otel_traces) "
        "WHERE startsWith(SpanName, 'execute_tool ') "
        "  AND SpanAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp",
        params={"sid": sid},
    ):
        attrs = r.get("SpanAttributes", {})
        tool_name = (
            attrs.get("gen_ai.tool.name")
            or r["SpanName"].removeprefix("execute_tool ").strip()
        )
        start_ns = _int(r.get("start_ns"))
        duration = _int(r.get("Duration"))

        args_payload = _try_json(attrs.get("gen_ai.tool.call.arguments"))
        result_payload = _try_json(attrs.get("gen_ai.tool.call.result"))

        yield {
            "tool": tool_name,
            "span_id": r.get("SpanId"),
            "start_time_unix_nano": start_ns or None,
            "end_time_unix_nano": (start_ns + duration) if start_ns and duration else None,
            "args": args_payload,
            "result": result_payload,
            "attributes": attrs,
        }


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def info(url: str, sid: str) -> dict[str, Any]:
    """Session metadata — header + fingerprint."""
    payload: dict[str, Any] = {}
    for event_name, key in (
        ("agentm.session.header", "header"),
        ("agentm.session.fingerprint", "fingerprint"),
    ):
        rows = query(
            url,
            "SELECT Body FROM (SELECT DISTINCT * FROM otel_logs) "
            "WHERE EventName = {ev:String} "
            "  AND LogAttributes['agentm.session.id'] = {sid:String} "
            "ORDER BY Timestamp DESC LIMIT 1",
            params={"ev": event_name, "sid": sid},
        )
        if rows:
            body = parse_body(rows[0].get("Body"))
            if isinstance(body, dict):
                payload[key] = body
    return payload


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def stats(url: str, sid: str) -> dict[str, Any]:
    """Event-name histogram + tool/turn statistics for a session."""
    log_rows = query(
        url,
        "SELECT EventName, count() AS cnt FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String} "
        "GROUP BY EventName ORDER BY cnt DESC",
        params={"sid": sid},
    )
    span_rows = query(
        url,
        "SELECT SpanName, count() AS cnt FROM (SELECT DISTINCT * FROM otel_traces) "
        "WHERE SpanAttributes['agentm.session.id'] = {sid:String} "
        "GROUP BY SpanName ORDER BY cnt DESC",
        params={"sid": sid},
    )
    logs_sorted = {r["EventName"]: _int(r["cnt"]) for r in log_rows}
    spans_sorted = {r["SpanName"]: _int(r["cnt"]) for r in span_rows}

    tool_stats = _tool_stats(url, sid)
    turn_stats = _turn_stats(url, sid)
    session_stats = _session_stats(url, sid)
    context_snapshots = _context_snapshots(url, sid)

    return {
        "file": f"clickhouse (session {sid})",
        "logs": logs_sorted,
        "spans": spans_sorted,
        "log_total": sum(logs_sorted.values()),
        "span_total": sum(spans_sorted.values()),
        "tools": tool_stats,
        "turns": turn_stats,
        "session": session_stats,
        "context_snapshots": context_snapshots,
    }


def _tool_stats(url: str, sid: str) -> dict[str, Any]:
    """Per-tool call count, result size, and duration stats."""
    rows = query(
        url,
        "SELECT "
        "  SpanAttributes['gen_ai.tool.name'] AS tool, "
        "  count() AS calls, "
        "  countIf(SpanAttributes['agentm.tool.is_error'] = 'true') AS errors, "
        "  avg(length(SpanAttributes['gen_ai.tool.call.result'])) AS avg_result_chars, "
        "  max(length(SpanAttributes['gen_ai.tool.call.result'])) AS max_result_chars, "
        "  sum(length(SpanAttributes['gen_ai.tool.call.result'])) AS total_result_chars, "
        "  avg(Duration) / 1e6 AS avg_duration_ms, "
        "  quantile(0.95)(Duration) / 1e6 AS p95_duration_ms, "
        "  max(Duration) / 1e6 AS max_duration_ms "
        "FROM (SELECT DISTINCT * FROM otel_traces) "
        "WHERE SpanAttributes['agentm.session.id'] = {sid:String} "
        "  AND SpanAttributes['gen_ai.operation.name'] = 'execute_tool' "
        "GROUP BY tool ORDER BY calls DESC",
        params={"sid": sid},
    )
    tools: dict[str, Any] = {}
    for r in rows:
        name = r["tool"]
        tools[name] = {
            "calls": _int(r["calls"]),
            "errors": _int(r.get("errors", 0)),
            "avg_result_chars": int(float(r.get("avg_result_chars", 0))),
            "max_result_chars": _int(r.get("max_result_chars", 0)),
            "total_result_chars": _int(r.get("total_result_chars", 0)),
            "avg_duration_ms": int(float(r.get("avg_duration_ms", 0))),
            "p95_duration_ms": int(float(r.get("p95_duration_ms", 0))),
            "max_duration_ms": int(float(r.get("max_duration_ms", 0))),
        }
    return tools


def _turn_stats(url: str, sid: str) -> dict[str, Any]:
    """Aggregate turn-level statistics from turn summary logs."""
    rows = query(
        url,
        "SELECT "
        "  count() AS total_turns, "
        "  sum(JSONExtractInt(Body, 'tool_call_count')) AS total_tool_calls, "
        "  sum(JSONExtractInt(Body, 'tool_error_count')) AS total_tool_errors, "
        "  sum(JSONExtractInt(Body, 'input_tokens')) AS total_input_tokens, "
        "  sum(JSONExtractInt(Body, 'output_tokens')) AS total_output_tokens, "
        "  sum(JSONExtractInt(Body, 'cache_read')) AS total_cache_read, "
        "  avg(JSONExtractInt(Body, 'input_tokens')) AS avg_input_tokens, "
        "  max(JSONExtractInt(Body, 'input_tokens')) AS max_input_tokens, "
        "  min(JSONExtractInt(Body, 'input_tokens')) AS min_input_tokens "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String} "
        "  AND EventName = 'agentm.turn.summary'",
        params={"sid": sid},
    )
    if not rows:
        return {}
    r = rows[0]
    return {
        "total_turns": _int(r.get("total_turns", 0)),
        "total_tool_calls": _int(r.get("total_tool_calls", 0)),
        "total_tool_errors": _int(r.get("total_tool_errors", 0)),
        "total_input_tokens": _int(r.get("total_input_tokens", 0)),
        "total_output_tokens": _int(r.get("total_output_tokens", 0)),
        "total_cache_read": _int(r.get("total_cache_read", 0)),
        "avg_input_tokens": int(float(r.get("avg_input_tokens", 0))),
        "max_input_tokens": _int(r.get("max_input_tokens", 0)),
        "min_input_tokens": _int(r.get("min_input_tokens", 0)),
    }


def _context_snapshots(url: str, sid: str) -> list[dict[str, Any]]:
    """Extract context_breakdown from peak-context turns.

    Only available for sessions recorded with the breakdown instrumentation.
    Returns snapshots for the turn with highest input_tokens and
    the last turn.
    """
    rows = query(
        url,
        "SELECT "
        "  JSONExtractInt(Body, 'turn_index') AS turn_idx, "
        "  JSONExtractInt(Body, 'input_tokens') AS input_tokens, "
        "  JSONExtractRaw(Body, 'context_breakdown') AS breakdown_raw "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String} "
        "  AND EventName = 'agentm.turn.summary' "
        "  AND JSONHas(Body, 'context_breakdown') = 1 "
        "ORDER BY input_tokens DESC "
        "LIMIT 1",
        params={"sid": sid},
    )
    snapshots: list[dict[str, Any]] = []
    for r in rows:
        import json as _json

        try:
            breakdown = _json.loads(r.get("breakdown_raw", "{}"))
        except (ValueError, TypeError):
            continue
        snapshots.append({
            "turn_index": _int(r.get("turn_idx", 0)),
            "input_tokens": _int(r.get("input_tokens", 0)),
            "label": "peak",
            **breakdown,
        })
    return snapshots


def _session_stats(url: str, sid: str) -> dict[str, Any]:
    """Session-level metadata: duration, child sessions, stop reasons."""
    duration_rows = query(
        url,
        "SELECT "
        "  min(Timestamp) AS start_time, "
        "  max(Timestamp) AS end_time, "
        "  dateDiff('second', min(Timestamp), max(Timestamp)) AS duration_s "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String}",
        params={"sid": sid},
    )
    child_rows = query(
        url,
        "SELECT "
        "  LogAttributes['agentm.session.purpose'] AS purpose, "
        "  count() AS cnt "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE LogAttributes['agentm.root.session.id'] = {sid:String} "
        "  AND EventName = 'agentm.session.start' "
        "  AND LogAttributes['agentm.session.id'] != {sid:String} "
        "GROUP BY purpose ORDER BY cnt DESC",
        params={"sid": sid},
    )
    stop_rows = query(
        url,
        "SELECT "
        "  JSONExtractString(Body, 'stop_reason') AS stop_reason, "
        "  count() AS cnt "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String} "
        "  AND EventName = 'agentm.turn.summary' "
        "GROUP BY stop_reason ORDER BY cnt DESC",
        params={"sid": sid},
    )
    result: dict[str, Any] = {}
    if duration_rows:
        r = duration_rows[0]
        result["start_time"] = str(r.get("start_time", ""))
        result["end_time"] = str(r.get("end_time", ""))
        result["duration_s"] = _int(r.get("duration_s", 0))
    if child_rows:
        result["child_sessions"] = {
            r["purpose"]: _int(r["cnt"]) for r in child_rows
        }
    if stop_rows:
        result["stop_reasons"] = {
            r["stop_reason"]: _int(r["cnt"]) for r in stop_rows
        }
    return result


# ---------------------------------------------------------------------------
# generic logs / spans
# ---------------------------------------------------------------------------


def logs(
    url: str,
    sid: str,
    *,
    name: str | None = None,
    name_prefix: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Generic log query filtered by session + optional event name."""
    where = ["LogAttributes['agentm.session.id'] = {sid:String}"]
    params: dict[str, str] = {"sid": sid}
    if name:
        where.append("EventName = {name:String}")
        params["name"] = name
    elif name_prefix:
        where.append("startsWith(EventName, {prefix:String})")
        params["prefix"] = name_prefix
    sql = (
        "SELECT EventName, Body, LogAttributes, "
        "  toUnixTimestamp64Nano(Timestamp) AS time_ns, TraceId, SpanId "
        f"FROM (SELECT DISTINCT * FROM otel_logs) WHERE {' AND '.join(where)} "
        "ORDER BY Timestamp"
    )
    for r in query(url, sql, params):
        body = parse_body(r.get("Body"))
        yield {
            "event_name": r["EventName"],
            "body": body,
            "trace_id": r.get("TraceId"),
            "span_id": r.get("SpanId"),
            "time_unix_nano": _int(r.get("time_ns")) or None,
            "attributes": r.get("LogAttributes", {}),
        }


def spans(
    url: str,
    sid: str,
    *,
    name: str | None = None,
    name_prefix: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Generic span query filtered by session + optional span name."""
    where = ["SpanAttributes['agentm.session.id'] = {sid:String}"]
    params: dict[str, str] = {"sid": sid}
    if name:
        where.append("SpanName = {name:String}")
        params["name"] = name
    elif name_prefix:
        where.append("startsWith(SpanName, {prefix:String})")
        params["prefix"] = name_prefix
    sql = (
        "SELECT SpanName, SpanId, ParentSpanId, SpanAttributes, Duration, "
        "  toUnixTimestamp64Nano(Timestamp) AS start_ns, TraceId "
        f"FROM (SELECT DISTINCT * FROM otel_traces) WHERE {' AND '.join(where)} "
        "ORDER BY Timestamp"
    )
    for r in query(url, sql, params):
        start_ns = _int(r.get("start_ns"))
        duration = _int(r.get("Duration"))
        yield {
            "name": r["SpanName"],
            "span_id": r.get("SpanId"),
            "parent_span_id": r.get("ParentSpanId"),
            "trace_id": r.get("TraceId"),
            "start_time_unix_nano": start_ns or None,
            "end_time_unix_nano": (start_ns + duration) if start_ns and duration else None,
            "attributes": r.get("SpanAttributes", {}),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_body(raw: Any) -> Any:
    """Parse a Body column value — JSON string → dict, passthrough otherwise."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw
    return raw


def _try_json(raw: Any) -> Any:
    """Try to JSON-parse a string; return as-is on failure."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def query_binary(
    url: str,
    sql: str,
    params: Mapping[str, str | list[str]] | None = None,
    *,
    database: str = "otel",
    timeout: int = 60,
) -> bytes:
    """Execute SQL and return the raw response body as bytes.

    Use with ``FORMAT Parquet`` or any other binary output format — the
    caller writes the bytes directly to a file without parsing.
    """
    qp: dict[str, str] = {"database": database}
    if params:
        for k, v in params.items():
            qp[f"param_{k}"] = _encode_param(v)
    full_url = f"{url}/?{urllib.parse.urlencode(qp)}"
    req = urllib.request.Request(
        full_url, data=sql.encode("utf-8"), method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Bulk (multi-session) helpers
# ---------------------------------------------------------------------------
#
# The two aggregation fragments below are the single home of the
# messages-per-session and turn-usage-per-session schema knowledge; the bulk
# helpers and the raw Parquet export all compose from them.

_BULK_MESSAGES_SQL = (
    "SELECT session_id, groupArray(Body) AS bodies FROM ("
    "  SELECT _session_id AS session_id, Body, Timestamp "
    "  FROM otel_logs "
    "  WHERE EventName = 'agentm.message.appended' "
    "    AND _session_id IN {sids:Array(String)} "
    "  ORDER BY Timestamp ASC"
    ") GROUP BY session_id"
)

_BULK_TURN_USAGE_SQL = (
    "SELECT "
    "  _session_id AS session_id, "
    "  count(*) AS turns, "
    "  sum(JSONExtractInt(Body, 'input_tokens')) AS input_tokens, "
    "  sum(JSONExtractInt(Body, 'output_tokens')) AS output_tokens "
    "FROM otel_logs "
    "WHERE EventName = 'agentm.turn.summary' "
    "  AND _session_id IN {sids:Array(String)} "
    "GROUP BY session_id"
)


def bulk_session_entries(
    url: str, session_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """``agentm.message.appended`` bodies for many sessions in one query."""
    if not session_ids:
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for row in query(url, _BULK_MESSAGES_SQL, {"sids": session_ids}, timeout=120):
        bodies = row.get("bodies", [])
        entries = [
            parsed
            for b in (bodies if isinstance(bodies, list) else [])
            if isinstance(parsed := parse_body(b), dict)
        ]
        result[row.get("session_id", "")] = entries
    return result


def bulk_turn_usage(
    url: str, session_ids: list[str],
) -> dict[str, dict[str, int]]:
    """Turn counts + token sums for many sessions in one query."""
    if not session_ids:
        return {}
    result: dict[str, dict[str, int]] = {}
    for row in query(url, _BULK_TURN_USAGE_SQL, {"sids": session_ids}, timeout=60):
        result[row.get("session_id", "")] = {
            "turns": _int(row.get("turns")),
            "input_tokens": _int(row.get("input_tokens")),
            "output_tokens": _int(row.get("output_tokens")),
        }
    return result


def bulk_system_prompts(url: str, session_ids: list[str]) -> dict[str, str]:
    """First ``agentm.llm.system_prompt`` text for many sessions."""
    if not session_ids:
        return {}
    result: dict[str, str] = {}
    for row in query(
        url,
        "SELECT "
        "  _session_id AS session_id, "
        "  any(Body) AS body "
        "FROM otel_logs "
        "WHERE EventName = 'agentm.llm.system_prompt' "
        "  AND _session_id IN {sids:Array(String)} "
        "GROUP BY session_id",
        {"sids": session_ids},
        timeout=60,
    ):
        body = parse_body(row.get("body"))
        text = body.get("text", "") if isinstance(body, dict) else ""
        if text:
            result[row.get("session_id", "")] = text
    return result


def bulk_models(url: str, session_ids: list[str]) -> dict[str, str]:
    """Model name (from ``chat <model>`` spans) for many sessions."""
    if not session_ids:
        return {}
    result: dict[str, str] = {}
    for row in query(
        url,
        "SELECT "
        "  _session_id AS session_id, "
        "  any(SpanName) AS model_span "
        "FROM otel_traces "
        "WHERE startsWith(SpanName, 'chat ') "
        "  AND _session_id IN {sids:Array(String)} "
        "GROUP BY session_id",
        {"sids": session_ids},
        timeout=60,
    ):
        name = row.get("model_span", "")
        if name.startswith("chat "):
            result[row["session_id"]] = name.removeprefix("chat ").strip()
    return result


def raw_parquet_export(url: str, session_ids: list[str]) -> bytes:
    """Messages + turn usage joined per session, as Parquet bytes."""
    sql = (
        "SELECT "
        "  m.session_id, "
        "  m.bodies AS messages, "
        "  u.turns, "
        "  u.input_tokens, "
        "  u.output_tokens "
        f"FROM ({_BULK_MESSAGES_SQL}) m "
        f"LEFT JOIN ({_BULK_TURN_USAGE_SQL}) u "
        "ON m.session_id = u.session_id "
        "FORMAT Parquet"
    )
    return query_binary(url, sql, {"sids": session_ids}, timeout=120)


# ---------------------------------------------------------------------------
# doctor — data-quality invariants
# ---------------------------------------------------------------------------


def doctor(url: str, sid: str) -> list[dict[str, Any]]:
    """Run data-quality invariants for one session; return violations.

    Duplication checks deliberately hit the **raw** tables (the readers'
    DISTINCT wrapper would hide exactly the bug class they exist to catch);
    logical checks run on deduplicated rows so a known-duplicated historical
    session reports only its duplication, not phantom double lifecycles.

    Each violation: ``{check, severity, expected, actual, detail}`` with
    severity ``error`` or ``warning``.
    """
    violations: list[dict[str, Any]] = []

    def _add(check: str, severity: str, expected: Any, actual: Any, detail: str) -> None:
        violations.append({
            "check": check, "severity": severity,
            "expected": expected, "actual": actual, "detail": detail,
        })

    # 1. Span duplication (raw table).
    r = query(
        url,
        "SELECT count() AS rows, uniqExact(SpanId) AS uniq FROM otel_traces "
        "WHERE SpanAttributes['agentm.session.id'] = {sid:String}",
        params={"sid": sid},
    )[0]
    if _int(r["rows"]) != _int(r["uniq"]):
        _add(
            "span_duplication", "error", _int(r["uniq"]), _int(r["rows"]),
            "same SpanId stored multiple times — double export at emission "
            "or collector retry; sessions ingested before 89f20ea7 are "
            "expected to show exactly 2x",
        )
    span_total = _int(r["uniq"])

    # 2. Log duplication (raw table). (Timestamp, EventName, Body) is unique
    # per logical record at ns precision.
    r = query(
        url,
        "SELECT count() AS rows, "
        "  uniqExact((Timestamp, EventName, Body)) AS uniq "
        "FROM otel_logs "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String}",
        params={"sid": sid},
    )[0]
    if _int(r["rows"]) != _int(r["uniq"]):
        _add(
            "log_duplication", "error", _int(r["uniq"]), _int(r["rows"]),
            "identical log records stored multiple times — see span_duplication",
        )

    if span_total == 0 and _int(r["uniq"]) == 0:
        _add(
            "session_exists", "error", ">0 records", 0,
            "no spans or logs found for this session id",
        )
        return violations

    # 3. Session lifecycle (deduplicated): exactly one header and start;
    # a missing end means abnormal teardown (warning, not error).
    rows = query(
        url,
        "SELECT EventName, count() AS cnt "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String} "
        "  AND EventName IN "
        "    ('agentm.session.header', 'agentm.session.start', 'agentm.session.end') "
        "GROUP BY EventName",
        params={"sid": sid},
    )
    lifecycle = {row["EventName"]: _int(row["cnt"]) for row in rows}
    # Headers are re-emitted on update (stub at creation, resolved config
    # later); readers take the latest, so the invariant is presence only.
    if lifecycle.get("agentm.session.header", 0) == 0:
        _add(
            "session_lifecycle", "error", ">=1", 0,
            "agentm.session.header missing",
        )
    if lifecycle.get("agentm.session.start", 0) != 1:
        _add(
            "session_lifecycle", "error", 1,
            lifecycle.get("agentm.session.start", 0),
            "agentm.session.start must occur exactly once",
        )
    if lifecycle.get("agentm.session.end", 0) == 0:
        _add(
            "session_lifecycle", "warning", 1, 0,
            "agentm.session.end missing — session crashed or was killed",
        )
    elif lifecycle.get("agentm.session.end", 0) > 1:
        _add(
            "session_lifecycle", "error", 1, lifecycle["agentm.session.end"],
            "agentm.session.end emitted more than once",
        )

    # 4. Turn contiguity (deduplicated): turn_index values 0..N-1, no holes,
    # no repeats.
    r = query(
        url,
        "SELECT count() AS cnt, "
        "  uniqExact(JSONExtractInt(Body, 'turn_index')) AS uniq, "
        "  min(JSONExtractInt(Body, 'turn_index')) AS mn, "
        "  max(JSONExtractInt(Body, 'turn_index')) AS mx "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String} "
        "  AND EventName = 'agentm.turn.summary'",
        params={"sid": sid},
    )[0]
    turn_count = _int(r["cnt"])
    if turn_count > 0:
        if _int(r["uniq"]) != turn_count:
            _add(
                "turn_contiguity", "error", turn_count, _int(r["uniq"]),
                "duplicate turn_index in turn summaries",
            )
        elif _int(r["mn"]) != 0 or _int(r["mx"]) != turn_count - 1:
            _add(
                "turn_contiguity", "error",
                f"0..{turn_count - 1}", f"{_int(r['mn'])}..{_int(r['mx'])}",
                "turn_index sequence has holes or wrong origin",
            )

    # 5. Turn span/summary pairing (deduplicated): every turn emits both an
    # agentm.turn span and a turn.summary log.
    r = query(
        url,
        "SELECT count() AS cnt FROM (SELECT DISTINCT * FROM otel_traces) "
        "WHERE SpanAttributes['agentm.session.id'] = {sid:String} "
        "  AND SpanName = 'agentm.turn'",
        params={"sid": sid},
    )[0]
    turn_spans = _int(r["cnt"])
    if turn_spans != turn_count:
        _add(
            "turn_span_pairing",
            # A missing final pair happens on abnormal teardown; large gaps
            # mean instrumentation loss.
            "warning" if abs(turn_spans - turn_count) <= 1 else "error",
            turn_count, turn_spans,
            "agentm.turn span count != turn.summary log count",
        )

    # 6. Tool span parentage (deduplicated): every execute_tool span hangs
    # off an agentm.turn span of the same session.
    r = query(
        url,
        "SELECT count() AS orphans "
        "FROM (SELECT DISTINCT * FROM otel_traces) "
        "WHERE SpanAttributes['agentm.session.id'] = {sid:String} "
        "  AND SpanName LIKE 'execute_tool%' "
        "  AND ParentSpanId NOT IN ( "
        "    SELECT SpanId FROM (SELECT DISTINCT * FROM otel_traces) "
        "    WHERE SpanAttributes['agentm.session.id'] = {sid:String} "
        "      AND SpanName = 'agentm.turn')",
        params={"sid": sid},
    )[0]
    if _int(r["orphans"]) > 0:
        _add(
            "tool_span_parentage", "error", 0, _int(r["orphans"]),
            "execute_tool spans not parented to an agentm.turn span",
        )

    return violations


# ---------------------------------------------------------------------------
# scan — cohort-baseline outlier detection
# ---------------------------------------------------------------------------

_SCAN_METRICS = ("turns", "input_tokens", "duration_s", "peak_input", "tool_error_rate")


def scan(
    url: str,
    *,
    window_hours: int = 48,
    min_cohort: int = 5,
    limit: int = 50,
) -> dict[str, Any]:
    """Flag sessions that are outliers against their cohort baseline.

    A cohort is ``(scenario, task_class)``. For every session in the window,
    per-session features (turns, tokens, wall time, peak context, tool-error
    rate) are compared against the cohort's p50/p95; a session is flagged on
    a metric when it exceeds p95 AND 1.5x the median, in cohorts of at least
    ``min_cohort`` sessions. Anomalies are entry points for attribution —
    the scan says *where to look*, never *why*.
    """
    feature_rows = query(
        url,
        "SELECT "
        "  LogAttributes['agentm.session.id'] AS sid, "
        "  count() AS turns, "
        "  sum(JSONExtractInt(Body, 'input_tokens')) AS input_tokens, "
        "  max(JSONExtractInt(Body, 'input_tokens')) AS peak_input, "
        "  sum(JSONExtractInt(Body, 'tool_call_count')) AS tool_calls, "
        "  sum(JSONExtractInt(Body, 'tool_error_count')) AS tool_errors, "
        "  (max(toUnixTimestamp64Nano(Timestamp)) "
        "   - min(toUnixTimestamp64Nano(Timestamp))) / 1e9 AS duration_s "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE EventName = 'agentm.turn.summary' "
        "  AND Timestamp > now() - INTERVAL {h:UInt32} HOUR "
        "GROUP BY sid",
        params={"h": str(window_hours)},
        timeout=120,
    )
    identity_rows = query(
        url,
        "SELECT "
        "  LogAttributes['agentm.session.id'] AS sid, "
        "  anyLast(LogAttributes['agentm.session.scenario']) AS scenario, "
        "  anyLast(LogAttributes['agentm.task.class']) AS task_class, "
        "  anyLast(LogAttributes['agentm.task.eval_run_id']) AS eval_run_id "
        "FROM (SELECT DISTINCT * FROM otel_logs) "
        "WHERE EventName = 'agentm.session.fingerprint' "
        "  AND Timestamp > now() - INTERVAL {h:UInt32} HOUR "
        "GROUP BY sid",
        params={"h": str(window_hours)},
        timeout=120,
    )
    identity = {r["sid"]: r for r in identity_rows}

    cohorts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in feature_rows:
        ident = identity.get(r["sid"], {})
        tool_calls = _int(r.get("tool_calls"))
        session = {
            "session_id": r["sid"],
            "scenario": ident.get("scenario", ""),
            "task_class": ident.get("task_class", ""),
            "eval_run_id": ident.get("eval_run_id", ""),
            "turns": _int(r.get("turns")),
            "input_tokens": _int(r.get("input_tokens")),
            "peak_input": _int(r.get("peak_input")),
            "duration_s": float(r.get("duration_s") or 0.0),
            "tool_error_rate": (
                _int(r.get("tool_errors")) / tool_calls if tool_calls else 0.0
            ),
        }
        key = (session["scenario"], session["task_class"])
        cohorts.setdefault(key, []).append(session)

    def _quantile(sorted_values: list[float], q: float) -> float:
        if not sorted_values:
            return 0.0
        idx = min(int(q * len(sorted_values)), len(sorted_values) - 1)
        return sorted_values[idx]

    findings: list[dict[str, Any]] = []
    for (scenario, task_class), sessions in cohorts.items():
        if len(sessions) < min_cohort:
            continue
        for metric in _SCAN_METRICS:
            values = sorted(float(s[metric]) for s in sessions)
            p50 = _quantile(values, 0.50)
            p95 = _quantile(values, 0.95)
            for s in sessions:
                value = float(s[metric])
                if value > p95 and value > 1.5 * p50 and value > 0:
                    findings.append({
                        "session_id": s["session_id"],
                        "scenario": scenario,
                        "task_class": task_class,
                        "eval_run_id": s["eval_run_id"],
                        "metric": metric,
                        "value": round(value, 2),
                        "p50": round(p50, 2),
                        "p95": round(p95, 2),
                        "ratio_to_p50": round(value / p50, 2) if p50 else 0.0,
                        "cohort_size": len(sessions),
                    })

    findings.sort(key=lambda f: -f["ratio_to_p50"])
    return {
        "window_hours": window_hours,
        "sessions": len(feature_rows),
        "cohorts": {
            f"{sc or '<none>'}/{tc or '<none>'}": len(ss)
            for (sc, tc), ss in sorted(cohorts.items(), key=lambda kv: -len(kv[1]))
        },
        "findings": findings[:limit],
    }


__all__ = [
    "get_url",
    "resolve_session",
    "index",
    "messages",
    "turns",
    "usage",
    "chats",
    "tools",
    "info",
    "stats",
    "doctor",
    "scan",
    "logs",
    "spans",
    "query",
    "query_binary",
    "parse_body",
    "bulk_session_entries",
    "bulk_turn_usage",
    "bulk_system_prompts",
    "bulk_models",
    "raw_parquet_export",
]
