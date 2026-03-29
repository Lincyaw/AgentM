"""Trace stats, service call graph, and span call graph tools."""

from __future__ import annotations

import json
from typing import Any

from agentm.tools.observability._builders import (
    _build_filter_clauses,
    _build_time_clause,
    _empty_hint,
    _get_available_services,
    _resolve_file,
    _result,
)
from agentm.tools._shared import enforce_token_budget
from agentm.tools.observability._core import (
    INTERVAL_MAP,
    _query,
    _safe_tool,
)

_VALID_GROUP_BY: frozenset[str] = frozenset({"service_name", "span_name"})


# -- Trace Stats --

# Docstrings for factory-generated functions
_QUERY_TRACE_STATS_ABNORMAL_DOC = """Trace latency/error stats during the ABNORMAL (incident) period.

Returns percentile latencies and error counts per time bucket,
grouped by service or span name.

Args:
    request: A short description of what you want to look up.
    group_by: "service_name" or "span_name" (default "service_name").
    interval: Bucket interval -- "1m", "5m", or "15m" (default "5m").
    start_time: Optional start time (ISO format).
    end_time: Optional end time (ISO format).
    filters: Optional JSON filters.
"""

_QUERY_TRACE_STATS_NORMAL_DOC = """Trace latency/error stats during the NORMAL (baseline) period.

Use to compare baseline latency against the abnormal period.

Args:
    request: A short description of what you want to look up.
    group_by: "service_name" or "span_name" (default "service_name").
    interval: Bucket interval -- "1m", "5m", or "15m" (default "5m").
    start_time: Optional start time (ISO format).
    end_time: Optional end time (ISO format).
    filters: Optional JSON filters.
"""


def _make_query_trace_stats(period: str, doc: str):
    """Factory for creating period-specific trace stats query tools."""

    @_safe_tool
    async def query_trace_stats(
        _request: str,
        group_by: str = "service_name",
        interval: str = "5m",
        start_time: str | None = None,
        end_time: str | None = None,
        filters: str | None = None,
    ) -> str:
        file = _resolve_file("traces", period)
        db_interval = INTERVAL_MAP.get(interval, "5 minutes")
        if group_by not in _VALID_GROUP_BY:
            group_by = "service_name"
        fc, fc_params = _build_filter_clauses(filters)
        tc, tc_params = _build_time_clause(start_time, end_time)

        sql = f"""
            SELECT
                time_bucket(INTERVAL '{db_interval}', time) AS time_bucket,
                "{group_by}" AS "group",
                round(avg(duration) / 1000.0, 2) AS avg_duration_ms,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration) / 1000.0, 2) AS p50_ms,
                round(percentile_cont(0.9) WITHIN GROUP (ORDER BY duration) / 1000.0, 2) AS p90_ms,
                round(percentile_cont(0.99) WITHIN GROUP (ORDER BY duration) / 1000.0, 2) AS p99_ms,
                count(*) AS count,
                count(*) FILTER (WHERE "attr.status_code" = 'ERROR') AS error_count
            FROM read_parquet('{file}')
            WHERE 1=1{fc}{tc}
            GROUP BY time_bucket, "{group_by}"
            ORDER BY time_bucket, "{group_by}"
            LIMIT 200
        """
        params = fc_params + tc_params + [200]
        rows = _query(sql, params)
        if not rows:
            return _empty_hint(
                file,
                "query_trace_stats",
                {"available_services": _get_available_services(file)},
            )
        return _result(rows, "query_trace_stats")

    query_trace_stats.__doc__ = doc
    return query_trace_stats


query_trace_stats_abnormal = _make_query_trace_stats("abnormal", _QUERY_TRACE_STATS_ABNORMAL_DOC)
query_trace_stats_normal = _make_query_trace_stats("normal", _QUERY_TRACE_STATS_NORMAL_DOC)


# -- Service Call Graph --

# Docstrings for factory-generated functions
_GET_SERVICE_CALL_GRAPH_ABNORMAL_DOC = """Service-level call graph from ABNORMAL (incident) trace data.

Builds service-to-service edges with call counts, latency, and errors.

Args:
    request: A short description of what you want to look up.
    start_time: Optional start time (ISO format).
    end_time: Optional end time (ISO format).
    filters: Optional JSON filters on child spans.

Returns:
    JSON {nodes: [{service, span_count}],
    edges: [{source, target, call_count, avg_duration_ms, error_count}]}.
"""

_GET_SERVICE_CALL_GRAPH_NORMAL_DOC = """Service-level call graph from NORMAL (baseline) trace data.

Use to compare baseline call patterns against the abnormal period.

Args:
    request: A short description of what you want to look up.
    start_time: Optional start time (ISO format).
    end_time: Optional end time (ISO format).
    filters: Optional JSON filters on child spans.

Returns:
    JSON {nodes: [{service, span_count}],
    edges: [{source, target, call_count, avg_duration_ms, error_count}]}.
"""


def _make_get_service_call_graph(period: str, doc: str):
    """Factory for creating period-specific service call graph tools."""

    @_safe_tool
    async def get_service_call_graph(
        _request: str,
        start_time: str | None = None,
        end_time: str | None = None,
        filters: str | None = None,
    ) -> str:
        file = _resolve_file("traces", period)
        fc, fc_params = _build_filter_clauses(filters, "child")
        tc, tc_params = _build_time_clause(start_time, end_time, "child")

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
            WHERE parent.service_name != child.service_name{fc}{tc}
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
        edge_params = fc_params + tc_params
        nodes = _query(sql_nodes)
        edges = _query(sql_edges, edge_params)
        result: dict[str, Any] = {"nodes": nodes, "edges": edges}
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        return enforce_token_budget(payload, "get_service_call_graph")

    get_service_call_graph.__doc__ = doc
    return get_service_call_graph


get_service_call_graph_abnormal = _make_get_service_call_graph("abnormal", _GET_SERVICE_CALL_GRAPH_ABNORMAL_DOC)
get_service_call_graph_normal = _make_get_service_call_graph("normal", _GET_SERVICE_CALL_GRAPH_NORMAL_DOC)


# -- Span Call Graph --

# Docstrings for factory-generated functions
_GET_SPAN_CALL_GRAPH_ABNORMAL_DOC = """Span-level call tree for a trace in the ABNORMAL (incident) period.

Returns a flat span list; reconstruct the tree via parent_span_id.

Args:
    trace_id: The trace ID to look up.

Returns:
    JSON array of spans with span_id, parent_span_id, span_name,
    service_name, duration_ms, status_code.
"""

_GET_SPAN_CALL_GRAPH_NORMAL_DOC = """Span-level call tree for a trace in the NORMAL (baseline) period.

Args:
    trace_id: The trace ID to look up.

Returns:
    JSON array of spans.
"""


def _make_get_span_call_graph(period: str, doc: str):
    """Factory for creating period-specific span call graph tools."""

    @_safe_tool
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
            sample_tids = _query(
                f"SELECT DISTINCT trace_id FROM read_parquet('{file}') LIMIT 5"
            )
            return _empty_hint(
                file,
                "get_span_call_graph",
                {
                    "available_services": _get_available_services(file),
                    "sample_trace_ids": [r["trace_id"] for r in sample_tids],
                },
            )
        return _result(rows, "get_span_call_graph")

    get_span_call_graph.__doc__ = doc
    return get_span_call_graph


get_span_call_graph_abnormal = _make_get_span_call_graph("abnormal", _GET_SPAN_CALL_GRAPH_ABNORMAL_DOC)
get_span_call_graph_normal = _make_get_span_call_graph("normal", _GET_SPAN_CALL_GRAPH_NORMAL_DOC)
