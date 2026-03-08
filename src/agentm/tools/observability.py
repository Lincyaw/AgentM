"""Observability tools for querying metrics, traces, logs, and topology from parquet files.

Flat async functions matching AgentM's tool pattern. Module-level ``_data_dir``
is set by the builder at startup via ``set_data_directory()``.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import duckdb

TOKEN_LIMIT = 5000

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Module-level state, set by builder at startup
_data_dir: str | None = None

# File mapping: (category, period) -> relative path under data_dir
_FILE_MAP: dict[tuple[str, str], str] = {
    ("metrics", "abnormal"): "abnormal_metrics.parquet",
    ("metrics", "normal"): "normal_metrics.parquet",
    ("metrics_histogram", "abnormal"): "abnormal_metrics_histogram.parquet",
    ("metrics_histogram", "normal"): "normal_metrics_histogram.parquet",
    ("metrics_sum", "abnormal"): "abnormal_metrics_sum.parquet",
    ("metrics_sum", "normal"): "normal_metrics_sum.parquet",
    ("traces", "abnormal"): "abnormal_traces.parquet",
    ("traces", "normal"): "normal_traces.parquet",
    ("logs", "abnormal"): "abnormal_logs.parquet",
    ("logs", "normal"): "normal_logs.parquet",
}


# ── Helpers ───────────────────────────────────────────────────────────


def _serialize_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings and handle NaN/Infinity."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj


def _estimate_token_count(text: str) -> int:
    return (len(text) + 2) // 3


def _enforce_token_limit(payload: str, context: str) -> str:
    if _estimate_token_count(payload) <= TOKEN_LIMIT:
        return payload
    estimated_tokens = _estimate_token_count(payload)
    ratio = TOKEN_LIMIT / estimated_tokens
    parsed = json.loads(payload)
    if isinstance(parsed, list):
        current_size = len(parsed)
        suggested_limit = max(1, int(current_size * ratio * 0.8))
        rows_returned = current_size
    elif isinstance(parsed, dict):
        rows_returned = {k: len(v) for k, v in parsed.items() if isinstance(v, list)}
        suggested_limit = (
            {k: max(1, int(n * ratio * 0.8)) for k, n in rows_returned.items()}
            if rows_returned
            else None
        )
    else:
        rows_returned = None
        suggested_limit = None
    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": estimated_tokens,
        "token_limit": TOKEN_LIMIT,
        "rows_returned": rows_returned,
        "suggested_limit": suggested_limit,
        "suggestion": "Add more filters or reduce the LIMIT.",
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)


def _query(sql: str) -> list[dict[str, Any]]:
    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(sql).fetchall()
        columns = [desc[0] for desc in conn.description]
        rows = [dict(zip(columns, row, strict=False)) for row in result]
        return cast(list[dict[str, Any]], _serialize_datetime(rows))
    except duckdb.Error as e:
        error_msg = str(e)
        raise QueryError(error_msg) from e
    finally:
        conn.close()


class QueryError(Exception):
    """Raised when a DuckDB query fails. Contains a user-friendly error message."""


def _safe_tool(func):
    """Decorator that catches tool errors and returns JSON error instead of crashing."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except QueryError as e:
            return json.dumps({
                "error": "Query failed",
                "detail": str(e),
                "suggestion": "Check your filter column names and parameter values. "
                "Only use column/metric/service names that appeared in previous tool results.",
            }, ensure_ascii=False, indent=2)
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            return json.dumps({
                "error": str(e),
            }, ensure_ascii=False, indent=2)

    return wrapper


def _build_filter_clauses(filters_json: str | None, table_alias: str = "") -> str:
    if not filters_json:
        return ""
    try:
        filters: dict[str, str] = json.loads(filters_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    prefix = f"{table_alias}." if table_alias else ""
    parts = []
    for col, val in filters.items():
        safe_val = str(val).replace("'", "''")
        parts.append(f'{prefix}"{col}" = \'{safe_val}\'')
    return " AND " + " AND ".join(parts) if parts else ""


def _build_time_clause(
    start_time: str | None,
    end_time: str | None,
    table_alias: str = "",
) -> str:
    prefix = f"{table_alias}." if table_alias else ""
    parts = []
    if start_time:
        safe = start_time.replace("'", "''")
        parts.append(f"{prefix}time >= '{safe}'")
    if end_time:
        safe = end_time.replace("'", "''")
        parts.append(f"{prefix}time <= '{safe}'")
    return " AND " + " AND ".join(parts) if parts else ""


def _resolve_file(category: str, period: str = "abnormal") -> str:
    if not _data_dir:
        raise RuntimeError(
            "Data directory not set. Call set_data_directory first."
        )
    key = (category, period)
    if key not in _FILE_MAP:
        raise ValueError(f"Unknown file mapping: category={category}, period={period}")
    path = Path(_data_dir) / _FILE_MAP[key]
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return str(path)


def _result(rows: list[dict], context: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    return _enforce_token_limit(payload, context)


def _empty_hint(file: str, context: str, extra: dict | None = None) -> str:
    tr = _get_time_range(file)
    hint: dict[str, Any] = {
        "warning": "No data found for the given parameters.",
        "available_time_range": tr,
    }
    if extra:
        hint.update(extra)
    return json.dumps(hint, ensure_ascii=False, indent=2)


def _get_time_range(file: str) -> dict[str, str | None]:
    rows = _query(f"SELECT min(time) AS min_time, max(time) AS max_time FROM read_parquet('{file}')")
    return rows[0] if rows else {"min_time": None, "max_time": None}


def _get_available_metrics(file: str) -> list[str]:
    rows = _query(f"SELECT DISTINCT metric FROM read_parquet('{file}') ORDER BY metric")
    return [r["metric"] for r in rows]


def _get_available_services(file: str) -> list[str]:
    rows = _query(
        f"SELECT DISTINCT service_name FROM read_parquet('{file}') WHERE service_name IS NOT NULL ORDER BY service_name"
    )
    return [r["service_name"] for r in rows]


# ── Public API ────────────────────────────────────────────────────────


def set_data_directory(directory: str) -> str:
    """Set the working data directory containing observability parquet files.

    Called by the builder/eval pipeline at startup, NOT by the LLM.

    Args:
        directory: Path to the data directory.

    Returns:
        JSON with status and discovered files.
    """
    global _data_dir
    p = Path(directory)
    if not p.is_dir():
        return json.dumps({"error": f"Directory not found: {directory}"})
    parquet_files = sorted(str(f.relative_to(p)) for f in p.rglob("*.parquet"))
    if not parquet_files:
        return json.dumps({"error": f"No parquet files found in {directory}"})
    _data_dir = str(p)
    return json.dumps({"status": "ok", "data_dir": _data_dir, "files": parquet_files}, indent=2)


# ── Metrics OHLC ──────────────────────────────────────────────────────


async def _query_metrics_ohlc(
    metric_name: str,
    period: str,
    interval: str = "5m",
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    file = _resolve_file("metrics", period)
    interval_map = {"1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes"}
    db_interval = interval_map.get(interval, "5 minutes")
    safe_metric = metric_name.replace("'", "''")
    fc = _build_filter_clauses(filters)
    tc = _build_time_clause(start_time, end_time)

    sql = f"""
        SELECT
            time_bucket(INTERVAL '{db_interval}', time) AS time_bucket,
            first(value) AS open, max(value) AS high,
            min(value) AS low, last(value) AS close,
            count(*) AS count
        FROM read_parquet('{file}')
        WHERE metric = '{safe_metric}'{fc}{tc}
        GROUP BY time_bucket ORDER BY time_bucket
    """
    rows = _query(sql)
    if not rows:
        return _empty_hint(
            file,
            "query_metrics_ohlc",
            {"available_metrics": _get_available_metrics(file)},
        )
    return _result(rows, "query_metrics_ohlc")


@_safe_tool
async def query_metrics_ohlc_abnormal(
    metric_name: str,
    interval: str = "5m",
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    """Query ABNORMAL-period metrics as OHLC (K-line) summary per time bucket.

    Use this to inspect metric behavior during the incident window.
    Returns open/high/low/close/count per bucket.
    If the metric name does not exist, returns all available metric names.
    If no data in the time range, returns the available time range.

    Args:
        metric_name: Metric to query, e.g. "k8s.pod.cpu.utilization".
        interval: Bucket interval — "1m", "5m", or "15m" (default "5m").
        start_time: Optional start time (ISO format, e.g. "2025-08-28T20:45:00").
        end_time: Optional end time (ISO format).
        filters: Optional JSON, e.g. '{"service_name": "ts-order-service"}'.
    """
    return await _query_metrics_ohlc(metric_name, "abnormal", interval, start_time, end_time, filters)


@_safe_tool
async def query_metrics_ohlc_normal(
    metric_name: str,
    interval: str = "5m",
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    """Query NORMAL-period (baseline) metrics as OHLC summary per time bucket.

    Use this to compare baseline metric behavior against the abnormal period.
    Returns open/high/low/close/count per bucket.
    If the metric name does not exist, returns all available metric names.
    If no data in the time range, returns the available time range.

    Args:
        metric_name: Metric to query, e.g. "k8s.pod.cpu.utilization".
        interval: Bucket interval — "1m", "5m", or "15m" (default "5m").
        start_time: Optional start time (ISO format).
        end_time: Optional end time (ISO format).
        filters: Optional JSON, e.g. '{"service_name": "ts-order-service"}'.
    """
    return await _query_metrics_ohlc(metric_name, "normal", interval, start_time, end_time, filters)


# ── Trace Stats ───────────────────────────────────────────────────────


async def _query_trace_stats(
    period: str,
    group_by: str = "service_name",
    interval: str = "5m",
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    file = _resolve_file("traces", period)
    interval_map = {"1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes"}
    db_interval = interval_map.get(interval, "5 minutes")
    if group_by not in ("service_name", "span_name"):
        group_by = "service_name"
    fc = _build_filter_clauses(filters)
    tc = _build_time_clause(start_time, end_time)

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
    """
    rows = _query(sql)
    if not rows:
        return _empty_hint(
            file,
            "query_trace_stats",
            {"available_services": _get_available_services(file)},
        )
    return _result(rows, "query_trace_stats")


@_safe_tool
async def query_trace_stats_abnormal(
    group_by: str = "service_name",
    interval: str = "5m",
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    """Trace latency/error stats during the ABNORMAL (incident) period.

    Returns percentile latencies and error counts per time bucket,
    grouped by service or span name.

    Args:
        group_by: "service_name" or "span_name" (default "service_name").
        interval: Bucket interval — "1m", "5m", or "15m" (default "5m").
        start_time: Optional start time (ISO format).
        end_time: Optional end time (ISO format).
        filters: Optional JSON filters.
    """
    return await _query_trace_stats("abnormal", group_by, interval, start_time, end_time, filters)


@_safe_tool
async def query_trace_stats_normal(
    group_by: str = "service_name",
    interval: str = "5m",
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    """Trace latency/error stats during the NORMAL (baseline) period.

    Use to compare baseline latency against the abnormal period.

    Args:
        group_by: "service_name" or "span_name" (default "service_name").
        interval: Bucket interval — "1m", "5m", or "15m" (default "5m").
        start_time: Optional start time (ISO format).
        end_time: Optional end time (ISO format).
        filters: Optional JSON filters.
    """
    return await _query_trace_stats("normal", group_by, interval, start_time, end_time, filters)


# ── Service Call Graph ────────────────────────────────────────────────


async def _get_service_call_graph(
    period: str,
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    file = _resolve_file("traces", period)
    fc = _build_filter_clauses(filters, "child")
    tc_child = _build_time_clause(start_time, end_time, "child")

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
        WHERE parent.service_name != child.service_name{fc}{tc_child}
        GROUP BY parent.service_name, child.service_name
        ORDER BY call_count DESC
    """
    sql_nodes = f"""
        SELECT service_name AS service, count(*) AS span_count
        FROM read_parquet('{file}')
        GROUP BY service_name ORDER BY span_count DESC
    """
    nodes = _query(sql_nodes)
    edges = _query(sql_edges)
    result = {"nodes": nodes, "edges": edges}
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    return _enforce_token_limit(payload, "get_service_call_graph")


@_safe_tool
async def get_service_call_graph_abnormal(
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    """Service-level call graph from ABNORMAL (incident) trace data.

    Builds service-to-service edges with call counts, latency, and errors.

    Args:
        start_time: Optional start time (ISO format).
        end_time: Optional end time (ISO format).
        filters: Optional JSON filters on child spans.

    Returns:
        JSON {nodes: [{service, span_count}],
        edges: [{source, target, call_count, avg_duration_ms, error_count}]}.
    """
    return await _get_service_call_graph("abnormal", start_time, end_time, filters)


@_safe_tool
async def get_service_call_graph_normal(
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
) -> str:
    """Service-level call graph from NORMAL (baseline) trace data.

    Use to compare baseline call patterns against the abnormal period.

    Args:
        start_time: Optional start time (ISO format).
        end_time: Optional end time (ISO format).
        filters: Optional JSON filters on child spans.

    Returns:
        JSON {nodes: [{service, span_count}],
        edges: [{source, target, call_count, avg_duration_ms, error_count}]}.
    """
    return await _get_service_call_graph("normal", start_time, end_time, filters)


# ── Span Call Graph ───────────────────────────────────────────────────


async def _get_span_call_graph(trace_id: str, period: str) -> str:
    file = _resolve_file("traces", period)
    safe_tid = trace_id.replace("'", "''")
    sql = f"""
        SELECT span_id, parent_span_id, span_name, service_name,
               round(duration / 1000.0, 2) AS duration_ms,
               "attr.status_code" AS status_code
        FROM read_parquet('{file}')
        WHERE trace_id = '{safe_tid}'
        ORDER BY time
    """
    rows = _query(sql)
    if not rows:
        sample_tids = _query(f"SELECT DISTINCT trace_id FROM read_parquet('{file}') LIMIT 5")
        return _empty_hint(
            file,
            "get_span_call_graph",
            {
                "available_services": _get_available_services(file),
                "sample_trace_ids": [r["trace_id"] for r in sample_tids],
            },
        )
    return _result(rows, "get_span_call_graph")


@_safe_tool
async def get_span_call_graph_abnormal(
    trace_id: str,
) -> str:
    """Span-level call tree for a trace in the ABNORMAL (incident) period.

    Returns a flat span list; reconstruct the tree via parent_span_id.

    Args:
        trace_id: The trace ID to look up.

    Returns:
        JSON array of spans with span_id, parent_span_id, span_name,
        service_name, duration_ms, status_code.
    """
    return await _get_span_call_graph(trace_id, "abnormal")


@_safe_tool
async def get_span_call_graph_normal(
    trace_id: str,
) -> str:
    """Span-level call tree for a trace in the NORMAL (baseline) period.

    Args:
        trace_id: The trace ID to look up.

    Returns:
        JSON array of spans.
    """
    return await _get_span_call_graph(trace_id, "normal")


# ── Deployment Graph ──────────────────────────────────────────────────


@_safe_tool
async def get_deployment_graph(
    service_name: str | None = None,
) -> str:
    """Infrastructure deployment topology: which services run on which pods and nodes.

    Extracts service -> deployment -> pod -> node relationships from metrics
    resource attributes during the abnormal period.

    Args:
        service_name: Optional filter to show only this service's deployment.

    Returns:
        JSON list of unique deployment records, each with
        service, deployment, pod, node, namespace.
    """
    file = _resolve_file("metrics", "abnormal")

    svc_filter = "service_name = '{}'".format(service_name.replace("'", "''")) if service_name else ""
    k8s_filter = '"attr.k8s.pod.name" IS NOT NULL'
    where = f" WHERE {k8s_filter}"
    if svc_filter:
        where += f" AND {svc_filter}"

    sql = f"""
        SELECT DISTINCT
            service_name AS service,
            "attr.k8s.deployment.name" AS deployment,
            "attr.k8s.pod.name" AS pod,
            "attr.k8s.node.name" AS node,
            "attr.k8s.namespace.name" AS namespace
        FROM read_parquet('{file}'){where}
        ORDER BY service, deployment, pod
    """
    rows = _query(sql)

    services_with_pods = {r["service"] for r in rows}
    deploy_where = ' WHERE "attr.k8s.deployment.name" IS NOT NULL AND "attr.k8s.pod.name" IS NULL'
    if svc_filter:
        deploy_where += f" AND {svc_filter}"
    deploy_sql = f"""
        SELECT DISTINCT
            service_name AS service,
            "attr.k8s.deployment.name" AS deployment,
            NULL AS pod, NULL AS node,
            "attr.k8s.namespace.name" AS namespace
        FROM read_parquet('{file}'){deploy_where}
    """
    deploy_rows = _query(deploy_sql)
    for r in deploy_rows:
        if r["service"] not in services_with_pods:
            rows.append(r)

    if not rows:
        return json.dumps({"warning": "No Kubernetes deployment info found in metrics data."})
    rows = [{k: v for k, v in r.items() if v is not None and v != ""} for r in rows]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return _enforce_token_limit(payload, "get_deployment_graph")


# ── Search Logs ───────────────────────────────────────────────────────


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
    safe_kw = keyword.replace("'", "''")

    if match_mode == "exact":
        match_clause = f"message = '{safe_kw}'"
    elif match_mode == "regex":
        match_clause = f"regexp_matches(message, '{safe_kw}')"
    else:
        match_clause = f"message ILIKE '%{safe_kw}%'"

    extra = ""
    if level:
        extra += f" AND level = '{level.replace(chr(39), chr(39) * 2)}'"
    if service_name:
        extra += f" AND service_name = '{service_name.replace(chr(39), chr(39) * 2)}'"
    tc = _build_time_clause(start_time, end_time)

    sql = f"""
        SELECT time, level, service_name, message, trace_id, span_id
        FROM read_parquet('{file}')
        WHERE {match_clause}{extra}{tc}
        ORDER BY time
        LIMIT {int(limit)}
    """
    rows = _query(sql)
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
