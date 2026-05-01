"""Core utilities for RCA observability tools."""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import math
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, ParamSpec, TypeVar, cast, overload

import duckdb  # type: ignore[import-untyped]
import tiktoken  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

TOKEN_LIMIT = 5000

P = ParamSpec("P")
T = TypeVar("T")


def tool_error(msg: str, **extra: Any) -> str:
    """Return a JSON error response for a tool."""
    return json.dumps({"error": msg, **extra}, ensure_ascii=False)


def safe_tool(func: Callable[P, Awaitable[str]]) -> Callable[P, Awaitable[str]]:
    """Decorator that catches exceptions in async tool functions."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool %s failed: %s", func.__name__, exc)
            return tool_error(str(exc))

    return wrapper


def _estimate_tokens_with_tiktoken(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001
        return (len(text) + 2) // 3
    return len(encoding.encode(text))


def estimate_tokens(text: str) -> int:
    """Estimate the token count for *text*."""
    return _estimate_tokens_with_tiktoken(text)


def serialize_for_json(obj: Any) -> Any:
    """Recursively convert values to JSON-safe equivalents."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


@overload
def enforce_token_budget(payload: str, context: str) -> str: ...


@overload
def enforce_token_budget(
    payload: str,
    context: str,
    *,
    parsed: list[Any] | dict[str, Any],
) -> str: ...


def enforce_token_budget(
    payload: str,
    context: str,
    *,
    parsed: list[Any] | dict[str, Any] | None = None,
) -> str:
    """Trim oversized JSON payloads to the shared tool token budget."""
    if estimate_tokens(payload) <= TOKEN_LIMIT:
        return payload

    estimated_tokens = estimate_tokens(payload)
    ratio = TOKEN_LIMIT / estimated_tokens
    data = parsed if parsed is not None else json.loads(payload)

    if isinstance(data, list):
        original_count = len(data)
        keep = max(1, int(original_count * ratio * 0.8))
        truncated = data[:keep]
        result: dict[str, Any] = {
            "_truncated": True,
            "_total_rows": original_count,
            "_rows_returned": keep,
            "_context": context,
            "_suggestion": (
                "Add more filters, a narrower time range, or specify a LIMIT "
                "to get complete data."
            ),
            "data": truncated,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    if isinstance(data, dict):
        truncated_dict: dict[str, Any] = {}
        meta: dict[str, Any] = {"_truncated": True, "_context": context}
        for key, value in data.items():
            if isinstance(value, list):
                keep = max(1, int(len(value) * ratio * 0.8))
                truncated_dict[key] = value[:keep]
                meta[f"_{key}_total"] = len(value)
                meta[f"_{key}_returned"] = keep
            else:
                truncated_dict[key] = value
        meta["_suggestion"] = "Add more filters or a narrower time range."
        truncated_dict.update(meta)
        return json.dumps(truncated_dict, ensure_ascii=False, indent=2)

    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": estimated_tokens,
        "token_limit": TOKEN_LIMIT,
        "suggestion": "Add more filters or reduce the LIMIT.",
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)


def _ensure_tokenizers_parallel_disabled() -> None:
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


_data_dir_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "obs_data_dir", default=None
)
_obs_conn_var: contextvars.ContextVar[duckdb.DuckDBPyConnection | None] = (
    contextvars.ContextVar("obs_duckdb_conn", default=None)
)

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

ALLOWED_TABLE_FILES: frozenset[str] = frozenset(_FILE_MAP.values())

INTERVAL_MAP: dict[str, str] = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
}


class QueryError(Exception):
    """Raised when a DuckDB query fails."""


def _get_obs_conn() -> duckdb.DuckDBPyConnection:
    conn = _obs_conn_var.get()
    if conn is not None:
        return conn
    _ensure_tokenizers_parallel_disabled()
    conn = duckdb.connect(":memory:")
    _obs_conn_var.set(conn)
    return conn


def close_obs_connection() -> None:
    conn = _obs_conn_var.get()
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        _obs_conn_var.set(None)


def _query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    conn = _get_obs_conn()
    try:
        result = conn.execute(sql, params or []).fetchall()
        columns = [desc[0] for desc in conn.description]
        rows = [dict(zip(columns, row, strict=False)) for row in result]
        return cast(list[dict[str, Any]], serialize_for_json(rows))
    except duckdb.Error as exc:
        raise QueryError(str(exc)) from exc


def obs_safe_tool(func: Callable[P, Awaitable[str]]) -> Callable[P, Awaitable[str]]:
    """Add observability-specific query guidance on top of ``safe_tool``."""

    @functools.wraps(func)
    async def _inner(*args: P.args, **kwargs: P.kwargs) -> str:
        try:
            return await func(*args, **kwargs)
        except QueryError as exc:
            return json.dumps(
                {
                    "error": "Query failed",
                    "detail": str(exc),
                    "suggestion": (
                        "Check your filter column names and parameter values. "
                        "Only use column, metric, or service names that appeared "
                        "in previous tool results."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )

    return safe_tool(_inner)


def set_data_directory(directory: str) -> str:
    """Set the working data directory containing observability parquet files."""
    from pathlib import Path

    path = Path(directory)
    if not path.is_dir():
        return json.dumps({"error": f"Directory not found: {directory}"})
    parquet_files = sorted(str(file.relative_to(path)) for file in path.rglob("*.parquet"))
    if not parquet_files:
        return json.dumps({"error": f"No parquet files found in {directory}"})
    _data_dir_var.set(str(path))
    return json.dumps(
        {"status": "ok", "data_dir": str(path), "files": parquet_files},
        indent=2,
    )
