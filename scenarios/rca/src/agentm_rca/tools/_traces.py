"""Trace stats, service call graph, and span call graph tools."""

from __future__ import annotations

import json

from ._builders import (
    _build_filter_clauses,
    _build_time_clause,
    _empty_hint,
    _get_available_services,
    _resolve_file,
    _result,
)
from ._core import INTERVAL_MAP, _query, enforce_token_budget, obs_safe_tool

_VALID_GROUP_BY: frozenset[str] = frozenset({"service_name", "span_name"})

_QUERY_TRACE_STATS_ABNORMAL_DOC = """Trace latency and error stats during the abnormal period."""
_QUERY_TRACE_STATS_NORMAL_DOC = """Trace latency and error stats during the normal period."""
_GET_SERVICE_CALL_GRAPH_ABNORMAL_DOC = """Service-level call graph from abnormal trace data."""
_GET_SERVICE_CALL_GRAPH_NORMAL_DOC = """Service-level call graph from normal trace data."""
_GET_SPAN_CALL_GRAPH_ABNORMAL_DOC = """Span-level call tree for a trace in the abnormal period."""
_GET_SPAN_CALL_GRAPH_NORMAL_DOC = """Span-level call tree for a trace in the normal period."""


def _make_query_trace_stats(period: str, doc: str):
    @obs_safe_tool
    async def query_trace_stats(
        request: str,
        group_by: str = "service_name",
        interval: str = "5m",
        start_time: str | None = None,
        end_time: str | None = None,
        filters: str | None = None,
    ) -> str:
        del request
        file = _resolve_file("traces", period)
        db_interval = INTERVAL_MAP.get(interval, "5 minutes")
        grouping = group_by if group_by in _VALID_GROUP_BY else "service_name"
        filter_clause, filter_params = _build_filter_clauses(filters)
        time_clause, time_params = _build_time_clause(start_time, end_time)

        sql = f"""
            SELECT
                time_bucket(INTERVAL '{db_interval}', time) AS time_bucket,
                "{grouping}" AS "group",
                round(avg(duration) / 1000.0, 2) AS avg_duration_ms,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration) / 1000.0, 2) AS p50_ms,
                round(percentile_cont(0.9) WITHIN GROUP (ORDER BY duration) / 1000.0, 2) AS p90_ms,
                round(percentile_cont(0.99) WITHIN GROUP (ORDER BY duration) / 1000.0, 2) AS p99_ms,
                count(*) AS count,
                count(*) FILTER (WHERE "attr.status_code" = 'ERROR') AS error_count
            FROM read_parquet('{file}')
            WHERE 1=1{filter_clause}{time_clause}
            GROUP BY time_bucket, "{grouping}"
            ORDER BY time_bucket, "{grouping}"
            LIMIT 200
        """
        rows = _query(sql, filter_params + time_params)
        if not rows:
            return _empty_hint(
                file,
                {"available_services": _get_available_services(file)},
            )
        return _result(rows, "query_trace_stats")

    query_trace_stats.__doc__ = doc
    return query_trace_stats


def _make_get_service_call_graph(period: str, doc: str):
    @obs_safe_tool
    async def get_service_call_graph(
        request: str,
        start_time: str | None = None,
        end_time: str | None = None,
        filters: str | None = None,
    ) -> str:
        del request
        file = _resolve_file("traces", period)
        filter_clause, filter_params = _build_filter_clauses(filters, "child")
        time_clause, time_params = _build_time_clause(start_time, end_time, "child")

        sql_edges = f"""
            SELECT
                parent.service_name AS source,
                child.service_name AS target,
                count(*) AS call_count,
                round(avg(child.duration) / 1000.0, 2) AS avg_duration_ms,
                count(*) FILTER (WHERE child."attr.status_code" = 'ERROR') AS error_count
            FROM read_parquet('{file}') child
            JOIN read_parquet('{file}') parent
                ON child.trace_id = parent.trace_id
                AND child.parent_span_id = parent.span_id
            WHERE parent.service_name != child.service_name{filter_clause}{time_clause}
            GROUP BY parent.service_name, child.service_name
            ORDER BY call_count DESC
            LIMIT 100
        """
        sql_nodes = f"""
            SELECT service_name AS service, count(*) AS span_count
            FROM read_parquet('{file}')
            GROUP BY service_name ORDER BY span_count DESC
            LIMIT 100
        """
        nodes = _query(sql_nodes)
        edges = _query(sql_edges, filter_params + time_params)
        payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2)
        return enforce_token_budget(payload, "get_service_call_graph")

    get_service_call_graph.__doc__ = doc
    return get_service_call_graph


def _make_get_span_call_graph(period: str, doc: str):
    @obs_safe_tool
    async def get_span_call_graph(trace_id: str) -> str:
        file = _resolve_file("traces", period)
        sql = f"""
            SELECT span_id, parent_span_id, span_name, service_name,
                   round(duration / 1000.0, 2) AS duration_ms,
                   "attr.status_code" AS status_code
            FROM read_parquet('{file}')
            WHERE trace_id = ?
            ORDER BY time
        """
        rows = _query(sql, [trace_id])
        if not rows:
            sample_trace_ids = _query(
                f"SELECT DISTINCT trace_id FROM read_parquet('{file}') LIMIT 5"
            )
            return _empty_hint(
                file,
                {
                    "available_services": _get_available_services(file),
                    "sample_trace_ids": [row["trace_id"] for row in sample_trace_ids],
                },
            )
        return _result(rows, "get_span_call_graph")

    get_span_call_graph.__doc__ = doc
    return get_span_call_graph


query_trace_stats_abnormal = _make_query_trace_stats(
    "abnormal", _QUERY_TRACE_STATS_ABNORMAL_DOC
)
query_trace_stats_normal = _make_query_trace_stats(
    "normal", _QUERY_TRACE_STATS_NORMAL_DOC
)
get_service_call_graph_abnormal = _make_get_service_call_graph(
    "abnormal", _GET_SERVICE_CALL_GRAPH_ABNORMAL_DOC
)
get_service_call_graph_normal = _make_get_service_call_graph(
    "normal", _GET_SERVICE_CALL_GRAPH_NORMAL_DOC
)
get_span_call_graph_abnormal = _make_get_span_call_graph(
    "abnormal", _GET_SPAN_CALL_GRAPH_ABNORMAL_DOC
)
get_span_call_graph_normal = _make_get_span_call_graph(
    "normal", _GET_SPAN_CALL_GRAPH_NORMAL_DOC
)
