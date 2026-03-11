"""DuckDB SQL tool — lets an LLM query tabular data via raw SQL.

Usage pattern
-------------
1. At startup, call ``register_tables(tables)`` to describe the available tables.
   Each entry maps a logical table name to either:
   - a parquet file path (str / Path), or
   - an in-memory list of dicts.

2. Pass ``query_sql`` (and optionally ``describe_tables``) as tools to the LLM.

3. The LLM calls ``describe_tables`` first to learn column names/types, then
   writes SQL with those exact table names and calls ``query_sql``.

Thread/async safety
-------------------
Table registrations are stored in a ``ContextVar`` so each async task (or OS
thread) gets its own isolated view, just like the observability tools do.
"""

from __future__ import annotations

import contextvars
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

# ---------------------------------------------------------------------------
# Token budget (same constant used across observability tools)
# ---------------------------------------------------------------------------
TOKEN_LIMIT = 5000


# ---------------------------------------------------------------------------
# ContextVar: maps table_name -> source (file path string or list[dict])
# ---------------------------------------------------------------------------
_tables_var: contextvars.ContextVar[dict[str, str | list[dict]]] = (
    contextvars.ContextVar("duckdb_sql_tables", default={})
)


# ---------------------------------------------------------------------------
# Public setup API (called by orchestrator / eval harness, NOT by LLM)
# ---------------------------------------------------------------------------

TableSource = str | Path | list[dict[str, Any]]


def register_tables(tables: dict[str, TableSource]) -> None:
    """Register tables available for the LLM to query.

    Args:
        tables: Mapping of logical table name to data source.
            A source can be:
            - ``str`` or ``Path`` — path to a parquet file.
            - ``list[dict]`` — in-memory rows (registered as DuckDB views).

    Example::

        register_tables({
            "orders":    "/data/orders.parquet",
            "customers": "/data/customers.parquet",
            "summary":   [{"region": "us", "count": 42}, ...],
        })
    """
    normalised: dict[str, str | list[dict]] = {}
    for name, src in tables.items():
        if isinstance(src, Path):
            normalised[name] = str(src)
        else:
            normalised[name] = src  # type: ignore[assignment]
    _tables_var.set(normalised)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    """Recursively convert datetime / NaN / Inf to JSON-safe values."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    return obj


def _estimate_tokens(text: str) -> int:
    return (len(text) + 2) // 3


def _enforce_budget(payload: str, context: str) -> str:
    """Truncate oversized results and embed a hint instead of blowing the budget."""
    if _estimate_tokens(payload) <= TOKEN_LIMIT:
        return payload

    estimated = _estimate_tokens(payload)
    ratio = TOKEN_LIMIT / estimated
    parsed = json.loads(payload)

    if isinstance(parsed, list):
        original = len(parsed)
        keep = max(1, int(original * ratio * 0.8))
        return json.dumps(
            {
                "_truncated": True,
                "_total_rows": original,
                "_rows_returned": keep,
                "_context": context,
                "_suggestion": (
                    "Result too large. Narrow with WHERE / LIMIT, "
                    "select fewer columns, or use aggregation."
                ),
                "data": parsed[:keep],
            },
            ensure_ascii=False,
            indent=2,
        )

    # Scalar / unknown shape — return a budget-exceeded error
    return json.dumps(
        {
            "error": "Result exceeds token budget",
            "context": context,
            "estimated_tokens": estimated,
            "token_limit": TOKEN_LIMIT,
            "suggestion": "Add WHERE / LIMIT or use aggregation to reduce output size.",
        },
        ensure_ascii=False,
        indent=2,
    )


def _open_conn() -> tuple[duckdb.DuckDBPyConnection, dict[str, str | list[dict]]]:
    """Open a fresh in-memory DuckDB connection and register all tables.

    Returns the connection and the current table mapping so callers can
    include ``available_tables`` in error messages.
    """
    tables = _tables_var.get()
    conn = duckdb.connect(":memory:")
    for name, src in tables.items():
        if isinstance(src, str):
            # Parquet file — create a view
            safe_path = src.replace("'", "''")
            conn.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{safe_path}')"
            )
        else:
            # In-memory list[dict] — write to a temp JSON file, import, then delete
            if src:
                fd, tmp_path = tempfile.mkstemp(suffix=".json")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(src, f)
                    safe_tmp = tmp_path.replace("'", "''")
                    conn.execute(
                        f"CREATE TABLE {name} AS SELECT * FROM read_json_auto('{safe_tmp}')"
                    )
                finally:
                    # Remove temp file immediately; DuckDB already loaded the data
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    return conn, tables


# ---------------------------------------------------------------------------
# LLM-callable tools
# ---------------------------------------------------------------------------


async def describe_tables() -> str:
    """Return the schema (column names and types) of all registered tables.

    Call this **before** writing a SQL query to discover table names and
    column names/types.

    **Column quoting rule (CRITICAL)**:
    Column names that contain dots or other special characters (e.g.
    ``attr.k8s.pod.name``) **must** be wrapped in double-quotes in SQL.
    Each column descriptor includes a ``sql_ref`` field that is already
    correctly quoted — copy it verbatim into your query.

    Example — selecting a dotted column::

        SELECT service_name, "attr.k8s.pod.name" FROM abnormal_logs LIMIT 10

    Returns:
        JSON string — ``{ "<table>": { "row_count": N, "columns": [{name, type, sql_ref}, ...] }, ... }``
        or ``{ "error": "..." }`` if no tables are registered.
    """
    tables = _tables_var.get()
    if not tables:
        return json.dumps(
            {
                "error": "No tables registered.",
                "suggestion": "The orchestrator must call register_tables() before starting the agent.",
            }
        )

    conn, _ = _open_conn()
    try:
        schema: dict[str, Any] = {}
        for name in tables:
            result = conn.execute(f"SELECT * FROM {name} LIMIT 0")
            columns = [
                {
                    "name": desc[0],
                    "type": str(desc[1]),
                    # sql_ref: safe to paste directly into a SELECT / WHERE clause
                    "sql_ref": f'"{desc[0]}"' if _needs_quoting(desc[0]) else desc[0],
                }
                for desc in result.description
            ]
            row_count_row = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
            row_count = row_count_row[0] if row_count_row else 0
            schema[name] = {"row_count": row_count, "columns": columns}
        return json.dumps(schema, ensure_ascii=False, indent=2)
    except duckdb.Error as e:
        return json.dumps(
            {"error": f"Schema extraction failed: {e}"}, ensure_ascii=False
        )
    finally:
        conn.close()


def _needs_quoting(col: str) -> bool:
    """Return True if a column name requires double-quoting in DuckDB SQL."""
    import re

    # Needs quoting if it contains non-alphanumeric/underscore chars or starts with a digit
    return bool(re.search(r"[^a-zA-Z0-9_]", col)) or (len(col) > 0 and col[0].isdigit())


async def query_sql(sql: str) -> str:
    """Execute a SQL query against the registered tables and return the results.

    **Available tables** — call ``describe_tables`` first to learn what tables
    and columns exist.  Use the exact table names returned by that tool.

    **Column quoting rule (CRITICAL)**:
    Column names that contain dots or other special characters (e.g.
    ``attr.k8s.pod.name``) **must** be wrapped in double-quotes::

        -- WRONG: SELECT attr.k8s.pod.name FROM ...   (parsed as table.col.col)
        -- RIGHT: SELECT "attr.k8s.pod.name" FROM ...

    The ``sql_ref`` field in ``describe_tables`` output is already correctly
    quoted — copy it verbatim into your query.

    **Guidelines for staying within the token budget**:
    - Always include a ``LIMIT`` clause (recommend ≤ 100 rows for exploration).
    - Prefer ``SELECT col1, col2`` over ``SELECT *`` when you only need a few columns.
    - Use aggregations (``COUNT``, ``SUM``, ``AVG``, ``GROUP BY``) instead of
      fetching raw rows when you need summary statistics.
    - Filter with ``WHERE`` to reduce result size.

    **SQL dialect**: DuckDB SQL (PostgreSQL-compatible).  Supported features include
    window functions, ``LIST_AGG``, ``UNNEST``, ``STRUCT``, ``MAP``, regex via
    ``regexp_matches``, and full ``TIMESTAMP``/``INTERVAL`` arithmetic.

    Args:
        sql: A valid DuckDB SQL query string.

    Returns:
        JSON string — a list of row objects on success, or an error object on failure.

    Examples::

        -- Count rows per service
        SELECT service_name, COUNT(*) AS cnt
        FROM abnormal_logs GROUP BY service_name ORDER BY cnt DESC LIMIT 20

        -- Filter by dotted column (must use double-quotes)
        SELECT service_name, "attr.k8s.pod.name"
        FROM abnormal_logs WHERE "attr.k8s.pod.name" IS NOT NULL LIMIT 50

        -- Time-range filter on TIMESTAMP WITH TIME ZONE column
        SELECT * FROM abnormal_traces
        WHERE time >= '2025-09-04T12:37:00+08:00' LIMIT 100
    """
    tables = _tables_var.get()
    if not tables:
        return json.dumps(
            {
                "error": "No tables registered.",
                "suggestion": "The orchestrator must call register_tables() before starting the agent.",
            }
        )

    conn, tables = _open_conn()
    try:
        result = conn.execute(sql).fetchall()
        columns = [desc[0] for desc in conn.description]
        rows = [dict(zip(columns, row, strict=False)) for row in result]
        serialized = _serialize(rows)
        payload = json.dumps(serialized, ensure_ascii=False, indent=2)
        return _enforce_budget(payload, "query_sql")

    except duckdb.Error as e:
        error_msg = str(e)
        available = list(tables.keys())

        if "syntax error" in error_msg.lower() or "parser error" in error_msg.lower():
            return json.dumps(
                {
                    "error": f"SQL syntax error: {error_msg}",
                    "sql": sql,
                    "available_tables": available,
                    "hint": "Call describe_tables() to verify column names before retrying.",
                },
                ensure_ascii=False,
                indent=2,
            )
        if "catalog" in error_msg.lower() or "table" in error_msg.lower():
            return json.dumps(
                {
                    "error": f"Table/column not found: {error_msg}",
                    "sql": sql,
                    "available_tables": available,
                    "hint": (
                        "Table names are case-sensitive. "
                        "If the error mentions a dotted name like 'attr', the column name contains dots "
                        "and must be wrapped in double-quotes: "
                        'e.g. SELECT "attr.k8s.pod.name" FROM ... '
                        "Use the sql_ref field from describe_tables() for correct quoting."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            {
                "error": f"Query failed: {error_msg}",
                "sql": sql,
                "available_tables": available,
            },
            ensure_ascii=False,
            indent=2,
        )
    finally:
        conn.close()
