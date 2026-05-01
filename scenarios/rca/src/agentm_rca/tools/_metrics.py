"""Metrics OHLC query tools."""

from __future__ import annotations

from ._builders import (
    _build_filter_clauses,
    _build_time_clause,
    _empty_hint,
    _get_available_metrics,
    _resolve_file,
    _result,
)
from ._core import INTERVAL_MAP, _query, obs_safe_tool

_QUERY_METRICS_OHLC_ABNORMAL_DOC = """Query abnormal-period metrics as OHLC summaries per time bucket."""
_QUERY_METRICS_OHLC_NORMAL_DOC = """Query normal-period metrics as OHLC summaries per time bucket."""


def _make_query_metrics_ohlc(period: str, doc: str):
    @obs_safe_tool
    async def query_metrics_ohlc(
        metric_name: str,
        interval: str = "5m",
        start_time: str | None = None,
        end_time: str | None = None,
        filters: str | None = None,
    ) -> str:
        file = _resolve_file("metrics", period)
        db_interval = INTERVAL_MAP.get(interval, "5 minutes")
        filter_clause, filter_params = _build_filter_clauses(filters)
        time_clause, time_params = _build_time_clause(start_time, end_time)

        sql = f"""
            SELECT
                time_bucket(INTERVAL '{db_interval}', time) AS time_bucket,
                first(value) AS open, max(value) AS high,
                min(value) AS low, last(value) AS close,
                count(*) AS count
            FROM read_parquet('{file}')
            WHERE metric = ?{filter_clause}{time_clause}
            GROUP BY time_bucket ORDER BY time_bucket
            LIMIT 200
        """
        rows = _query(sql, [metric_name] + filter_params + time_params)
        if not rows:
            return _empty_hint(
                file,
                {"available_metrics": _get_available_metrics(file)},
            )
        return _result(rows, "query_metrics_ohlc")

    query_metrics_ohlc.__doc__ = doc
    return query_metrics_ohlc


query_metrics_ohlc_abnormal = _make_query_metrics_ohlc(
    "abnormal", _QUERY_METRICS_OHLC_ABNORMAL_DOC
)
query_metrics_ohlc_normal = _make_query_metrics_ohlc(
    "normal", _QUERY_METRICS_OHLC_NORMAL_DOC
)
