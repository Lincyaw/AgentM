"""Builder helpers: file resolution, filter/time clause construction, result formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.tools._shared import enforce_token_budget
from agentm.tools.observability._core import (
    _FILE_MAP,
    _data_dir_var,
    _query,
)


def _resolve_file(category: str, period: str = "abnormal") -> str:
    data_dir = _data_dir_var.get()
    if not data_dir:
        raise RuntimeError("Data directory not set. Call set_data_directory first.")
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
        filters: dict[str, str] = json.loads(filters_json)
    except (json.JSONDecodeError, TypeError):
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


def _result(rows: list[dict], context: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    return enforce_token_budget(payload, context)


def _empty_hint(file: str, _context: str, extra: dict | None = None) -> str:
    tr = _get_time_range(file)
    hint: dict[str, Any] = {
        "warning": "No data found for the given parameters.",
        "available_time_range": tr,
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
    return [r["metric"] for r in rows]


def _get_available_services(file: str) -> list[str]:
    rows = _query(
        f"SELECT DISTINCT service_name FROM read_parquet('{file}') "
        "WHERE service_name IS NOT NULL ORDER BY service_name"
    )
    return [r["service_name"] for r in rows]
