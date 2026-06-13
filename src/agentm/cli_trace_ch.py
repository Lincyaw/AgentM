"""ClickHouse query backend for ``agentm trace``.

SQL templates against the OTel Collector's ``otel_logs`` / ``otel_traces``
tables.  Uses ``urllib.request`` — zero extra dependencies.  Activated when
``AGENTM_CLICKHOUSE_URL`` is set or ClickHouse is reachable on
``http://localhost:8123``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator


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
    except (urllib.error.URLError, OSError):
        pass
    return None


def _query(
    url: str,
    sql: str,
    params: dict[str, str] | None = None,
    *,
    database: str = "otel",
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Execute SQL against ClickHouse HTTP interface, return rows as dicts."""
    qp: dict[str, str] = {"database": database}
    if params:
        for k, v in params.items():
            qp[f"param_{k}"] = v
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
    url: str, session: str | None, latest: bool,
) -> str | None:
    """Map ``--session``/``--latest`` to a concrete session_id."""
    if session:
        return session
    if latest:
        rows = _query(
            url,
            "SELECT LogAttributes['agentm.session.id'] AS sid "
            "FROM otel_logs "
            "WHERE EventName = 'agentm.session.start' "
            "ORDER BY Timestamp DESC LIMIT 1",
        )
        return rows[0]["sid"] if rows else None
    return None


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def index(url: str) -> Iterator[dict[str, Any]]:
    """Session topology — one row per ``session.start``."""
    for r in _query(
        url,
        "SELECT "
        "  LogAttributes['agentm.session.id']        AS session_id, "
        "  LogAttributes['agentm.session.root_id']    AS trace_id, "
        "  LogAttributes['agentm.session.parent_id']  AS parent_session_id, "
        "  LogAttributes['agentm.session.purpose']    AS purpose, "
        "  LogAttributes['agentm.session.scenario']   AS scenario "
        "FROM otel_logs "
        "WHERE EventName = 'agentm.session.start' "
        "ORDER BY Timestamp",
    ):
        yield {
            "session_id": r["session_id"] or None,
            "trace_id": r["trace_id"] or None,
            "parent_session_id": r["parent_session_id"] or None,
            "purpose": r["purpose"] or None,
            "scenario": r["scenario"] or None,
            "records": None,
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
            sys_rows = _query(
                url,
                "SELECT Body FROM otel_logs "
                "WHERE EventName = 'agentm.llm.system_prompt' "
                "  AND LogAttributes['agentm.session.id'] = {sid:String} "
                "ORDER BY Timestamp LIMIT 1",
                params={"sid": sid},
            )
            if sys_rows:
                body = _parse_body(sys_rows[0].get("Body"))
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

    for r in _query(
        url,
        "SELECT Body FROM otel_logs "
        "WHERE EventName = 'agentm.message.appended' "
        "  AND LogAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp",
        params={"sid": sid},
    ):
        body = _parse_body(r.get("Body"))
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
    for r in _query(
        url,
        "SELECT Body FROM otel_logs "
        "WHERE EventName = 'agentm.turn.summary' "
        "  AND LogAttributes['agentm.session.id'] = {sid:String} "
        "ORDER BY Timestamp",
        params={"sid": sid},
    ):
        body = _parse_body(r.get("Body"))
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
    for r in _query(
        url,
        "SELECT SpanName, SpanId, SpanAttributes, Duration, "
        "  toUnixTimestamp64Nano(Timestamp) AS start_ns "
        "FROM otel_traces "
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
    for r in _query(
        url,
        "SELECT SpanName, SpanId, SpanAttributes, Duration, "
        "  toUnixTimestamp64Nano(Timestamp) AS start_ns "
        "FROM otel_traces "
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
        rows = _query(
            url,
            "SELECT Body FROM otel_logs "
            "WHERE EventName = {ev:String} "
            "  AND LogAttributes['agentm.session.id'] = {sid:String} "
            "LIMIT 1",
            params={"ev": event_name, "sid": sid},
        )
        if rows:
            body = _parse_body(rows[0].get("Body"))
            if isinstance(body, dict):
                payload[key] = body
    return payload


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def stats(url: str, sid: str) -> dict[str, Any]:
    """Event-name histogram for a session."""
    log_rows = _query(
        url,
        "SELECT EventName, count() AS cnt FROM otel_logs "
        "WHERE LogAttributes['agentm.session.id'] = {sid:String} "
        "GROUP BY EventName ORDER BY cnt DESC",
        params={"sid": sid},
    )
    span_rows = _query(
        url,
        "SELECT SpanName, count() AS cnt FROM otel_traces "
        "WHERE SpanAttributes['agentm.session.id'] = {sid:String} "
        "GROUP BY SpanName ORDER BY cnt DESC",
        params={"sid": sid},
    )
    logs_sorted = {r["EventName"]: _int(r["cnt"]) for r in log_rows}
    spans_sorted = {r["SpanName"]: _int(r["cnt"]) for r in span_rows}
    return {
        "file": f"clickhouse (session {sid})",
        "logs": logs_sorted,
        "spans": spans_sorted,
        "log_total": sum(logs_sorted.values()),
        "span_total": sum(spans_sorted.values()),
    }


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
        f"FROM otel_logs WHERE {' AND '.join(where)} "
        "ORDER BY Timestamp"
    )
    for r in _query(url, sql, params):
        body = _parse_body(r.get("Body"))
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
        f"FROM otel_traces WHERE {' AND '.join(where)} "
        "ORDER BY Timestamp"
    )
    for r in _query(url, sql, params):
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


def _parse_body(raw: Any) -> Any:
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
    "logs",
    "spans",
]
