"""Core utilities for observability tools: query engine, token limits, ContextVar state."""

from __future__ import annotations

import contextvars
import functools
import json
import os
from typing import Any, cast

import duckdb

from agentm.tools._shared import (
    TOKEN_LIMIT,  # noqa: F401  (re-exported via observability/__init__)
    serialize_for_json,
)

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

# Allowed parquet file names for DuckDB table registration.
# Only observability data tables (logs, traces, metrics) are permitted;
# result files like conclusion.parquet are excluded.
ALLOWED_TABLE_FILES: frozenset[str] = frozenset(_FILE_MAP.values())

# Shared interval shorthand -> SQL INTERVAL string, used by _traces and _metrics.
INTERVAL_MAP: dict[str, str] = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
}


class QueryError(Exception):
    """Raised when a DuckDB query fails. Contains a user-friendly error message."""


def _query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(sql, params or []).fetchall()
        columns = [desc[0] for desc in conn.description]
        rows = [dict(zip(columns, row, strict=False)) for row in result]
        return cast(list[dict[str, Any]], serialize_for_json(rows))
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
            return json.dumps(
                {
                    "error": "Query failed",
                    "detail": str(e),
                    "suggestion": "Check your filter column names and parameter values. "
                    "Only use column/metric/service names that appeared in previous tool results.",
                },
                ensure_ascii=False,
                indent=2,
            )
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            return json.dumps(
                {
                    "error": str(e),
                },
                ensure_ascii=False,
                indent=2,
            )

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
    return json.dumps(
        {"status": "ok", "data_dir": data_dir, "files": parquet_files}, indent=2
    )
