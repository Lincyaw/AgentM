"""Metrics OHLC query tools."""

from __future__ import annotations

from agentm.tools.observability._builders import (
    _build_filter_clauses,
    _build_time_clause,
    _empty_hint,
    _get_available_metrics,
    _resolve_file,
    _result,
)
from agentm.tools.observability._core import INTERVAL_MAP, _query, _safe_tool


async def _query_metrics_ohlc(
    metric_name: str,
    period: str,
    interval: str = "5m",
    start_time: str | None = None,
    end_time: str | None = None,
    filters: str | None = None,
    limit: int = 200,
) -> str:
    file = _resolve_file("metrics", period)
    db_interval = INTERVAL_MAP.get(interval, "5 minutes")
    fc, fc_params = _build_filter_clauses(filters)
    tc, tc_params = _build_time_clause(start_time, end_time)

    sql = f"""
        SELECT
            time_bucket(INTERVAL '{db_interval}', time) AS time_bucket,
            first(value) AS open, max(value) AS high,
            min(value) AS low, last(value) AS close,
            count(*) AS count
        FROM read_parquet('{file}')
        WHERE metric = ?{fc}{tc}
        GROUP BY time_bucket ORDER BY time_bucket
        LIMIT ?
    """
    params = [metric_name] + fc_params + tc_params + [int(limit)]
    rows = _query(sql, params)
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
    return await _query_metrics_ohlc(
        metric_name, "abnormal", interval, start_time, end_time, filters
    )


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
    return await _query_metrics_ohlc(
        metric_name, "normal", interval, start_time, end_time, filters
    )
