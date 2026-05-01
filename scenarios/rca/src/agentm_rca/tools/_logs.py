"""Log search tools."""

from __future__ import annotations

from typing import Any

from ._builders import (
    _build_time_clause,
    _empty_hint,
    _get_available_services,
    _resolve_file,
    _result,
)
from ._core import _query, obs_safe_tool

_SEARCH_LOGS_ABNORMAL_DOC = """Search logs in the abnormal incident period."""
_SEARCH_LOGS_NORMAL_DOC = """Search logs in the normal baseline period."""


def _make_search_logs(period: str, doc: str):
    @obs_safe_tool
    async def search_logs(
        keyword: str,
        match_mode: str = "contains",
        level: str | None = None,
        service_name: str | list[str] | None = None,
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
            names = [service_name] if isinstance(service_name, str) else service_name
            placeholders = ", ".join("?" * len(names))
            extra += f" AND service_name IN ({placeholders})"
            params.extend(names)

        time_clause, time_params = _build_time_clause(start_time, end_time)
        params.extend(time_params)

        sql = f"""
            SELECT time, level, service_name, message, trace_id, span_id
            FROM read_parquet('{file}')
            WHERE {match_clause}{extra}{time_clause}
            ORDER BY time
            LIMIT ?
        """
        params.append(int(limit))
        rows = _query(sql, params)
        if not rows:
            return _empty_hint(
                file,
                {"available_services": _get_available_services(file)},
            )
        return _result(rows, "search_logs")

    search_logs.__doc__ = doc
    return search_logs


search_logs_abnormal = _make_search_logs("abnormal", _SEARCH_LOGS_ABNORMAL_DOC)
search_logs_normal = _make_search_logs("normal", _SEARCH_LOGS_NORMAL_DOC)
