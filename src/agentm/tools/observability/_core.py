"""Core utilities for observability tools: query engine, token limits, ContextVar state."""

from __future__ import annotations

import contextvars
import functools
import json
import math
import os
from datetime import datetime
from typing import Any, cast

import duckdb

TOKEN_LIMIT = 5000

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Thread-safe state via ContextVar, set by builder at startup
_data_dir_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "obs_data_dir", default=None
)

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


class QueryError(Exception):
    """Raised when a DuckDB query fails. Contains a user-friendly error message."""


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


def _query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(sql, params or []).fetchall()
        columns = [desc[0] for desc in conn.description]
        rows = [dict(zip(columns, row, strict=False)) for row in result]
        return cast(list[dict[str, Any]], _serialize_datetime(rows))
    except duckdb.Error as e:
        error_msg = str(e)
        raise QueryError(error_msg) from e
    finally:
        conn.close()


def _safe_tool(func):  # noqa: ANN001, ANN201
    """Decorator that catches tool errors and returns JSON error instead of crashing."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
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


def set_data_directory(directory: str) -> str:
    """Set the working data directory containing observability parquet files.

    Called by the builder/eval pipeline at startup, NOT by the LLM.

    Args:
        directory: Path to the data directory.

    Returns:
        JSON with status and discovered files.
    """
    from pathlib import Path

    p = Path(directory)
    if not p.is_dir():
        return json.dumps({"error": f"Directory not found: {directory}"})
    parquet_files = sorted(str(f.relative_to(p)) for f in p.rglob("*.parquet"))
    if not parquet_files:
        return json.dumps({"error": f"No parquet files found in {directory}"})
    data_dir = str(p)
    _data_dir_var.set(data_dir)
    return json.dumps({"status": "ok", "data_dir": data_dir, "files": parquet_files}, indent=2)
