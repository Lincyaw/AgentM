"""Log search tools."""

from __future__ import annotations

from typing import Any

from agentm.tools.observability._builders import (
    _build_time_clause,
    _empty_hint,
    _get_available_services,
    _resolve_file,
    _result,
)
from agentm.tools.observability._core import _query, _safe_tool


async def _search_logs(
    keyword: str,
    period: str,
    match_mode: str = "contains",
    level: str | None = None,
    service_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 50,
) -> str:
    file = _resolve_file("logs", period)
    params: list[Any] = []

    if match_mode == "exact":
        match_clause = "message = ?"
        params.append(keyword)
    elif match_mode == "regex":
        match_clause = "regexp_matches(message, ?)"
        params.append(keyword)
    else:
        match_clause = "message ILIKE ?"
        params.append(f"%{keyword}%")

    extra = ""
    if level:
        extra += " AND level = ?"
        params.append(level)
    if service_name:
        extra += " AND service_name = ?"
        params.append(service_name)

    tc, tc_params = _build_time_clause(start_time, end_time)
    params.extend(tc_params)

    sql = f"""
        SELECT time, level, service_name, message, trace_id, span_id
        FROM read_parquet('{file}')
        WHERE {match_clause}{extra}{tc}
        ORDER BY time
        LIMIT ?
    """
    params.append(int(limit))
    rows = _query(sql, params)
    if not rows:
        return _empty_hint(
            file,
            "search_logs",
            {"available_services": _get_available_services(file)},
        )
    return _result(rows, "search_logs")


@_safe_tool
async def search_logs_abnormal(
    keyword: str,
    match_mode: str = "contains",
    level: str | None = None,
    service_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 50,
) -> str:
    """Search logs in the ABNORMAL (incident) period.

    Args:
        keyword: Search term for log messages.
        match_mode: "exact", "contains", or "regex" (default "contains").
        level: Optional log level filter, e.g. "ERROR", "WARN".
        service_name: Optional service name filter.
        start_time: Optional start time (ISO format).
        end_time: Optional end time (ISO format).
        limit: Max results (default 50).
    """
    return await _search_logs(
        keyword, "abnormal", match_mode, level, service_name, start_time, end_time, limit
    )


@_safe_tool
async def search_logs_normal(
    keyword: str,
    match_mode: str = "contains",
    level: str | None = None,
    service_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 50,
) -> str:
    """Search logs in the NORMAL (baseline) period.

    Use to compare baseline log patterns against the incident window.

    Args:
        keyword: Search term for log messages.
        match_mode: "exact", "contains", or "regex" (default "contains").
        level: Optional log level filter, e.g. "ERROR", "WARN".
        service_name: Optional service name filter.
        start_time: Optional start time (ISO format).
        end_time: Optional end time (ISO format).
        limit: Max results (default 50).
    """
    return await _search_logs(keyword, "normal", match_mode, level, service_name, start_time, end_time, limit)
