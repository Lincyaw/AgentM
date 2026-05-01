"""Builder helpers for RCA observability queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._core import TOKEN_LIMIT, _FILE_MAP, _data_dir_var, _query, enforce_token_budget


def _resolve_file(category: str, period: str = "abnormal") -> str:
    data_dir = _data_dir_var.get()
    if not data_dir:
        raise RuntimeError("Data directory not set. Configure the RCA tool package first.")
    key = (category, period)
    if key not in _FILE_MAP:
        raise ValueError(f"Unknown file mapping: category={category}, period={period}")
    path = Path(data_dir) / _FILE_MAP[key]
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return str(path)


def _build_filter_clauses(
    filters_json: str | None, table_alias: str = ""
) -> tuple[str, list[Any]]:
    if not filters_json:
        return "", []
    try:
        filters = json.loads(filters_json)
    except (json.JSONDecodeError, TypeError):
        return "", []
    if not isinstance(filters, dict):
        return "", []

    prefix = f"{table_alias}." if table_alias else ""
    parts: list[str] = []
    params: list[Any] = []
    for col, val in filters.items():
        parts.append(f'{prefix}"{col}" = ?')
        params.append(str(val))
    clause = " AND " + " AND ".join(parts) if parts else ""
    return clause, params


def _build_time_clause(
    start_time: str | None,
    end_time: str | None,
    table_alias: str = "",
) -> tuple[str, list[Any]]:
    prefix = f"{table_alias}." if table_alias else ""
    parts: list[str] = []
    params: list[Any] = []
    if start_time:
        parts.append(f"{prefix}time >= ?")
        params.append(start_time)
    if end_time:
        parts.append(f"{prefix}time <= ?")
        params.append(end_time)
    clause = " AND " + " AND ".join(parts) if parts else ""
    return clause, params


def _result(rows: list[dict[str, Any]], context: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    return enforce_token_budget(payload, context)


def _empty_hint(file: str, extra: dict[str, Any] | None = None) -> str:
    hint: dict[str, Any] = {
        "warning": "No data found for the given parameters.",
        "available_time_range": _get_time_range(file),
        "token_limit": TOKEN_LIMIT,
    }
    if extra:
        hint.update(extra)
    return json.dumps(hint, ensure_ascii=False, indent=2)


def _get_time_range(file: str) -> dict[str, str | None]:
    rows = _query(
        f"SELECT min(time) AS min_time, max(time) AS max_time FROM read_parquet('{file}')"
    )
    return rows[0] if rows else {"min_time": None, "max_time": None}


def _get_available_metrics(file: str) -> list[str]:
    rows = _query(f"SELECT DISTINCT metric FROM read_parquet('{file}') ORDER BY metric")
    return [str(row["metric"]) for row in rows]


def _get_available_services(file: str) -> list[str]:
    rows = _query(
        f"SELECT DISTINCT service_name FROM read_parquet('{file}') "
        "WHERE service_name IS NOT NULL ORDER BY service_name"
    )
    return [str(row["service_name"]) for row in rows]
