"""DuckDB SQL tool for the RCA scenario.

Exposes two atoms:

* ``list_tables`` — enumerate the parquet files under ``data_dir`` and
  return their schema + row count. Each file is registered as a DuckDB
  view named after its filename without the ``.parquet`` suffix, so the
  agent can write ``SELECT ... FROM abnormal_traces`` instead of
  ``read_parquet('/abs/path/abnormal_traces.parquet')``.
* ``query_sql`` — execute a single read-only SQL statement against the
  registered views and return rows as JSON, capped by a token budget.

Configuration::

    - module: agentm_rca.tools.duckdb_sql
      config:
        data_dir: /path/to/converted     # required; or AGENTM_RCA_DATA_DIR
        exclude: [conclusion.parquet]    # optional, hides ground truth
        row_limit: 200                   # optional default LIMIT (200)
        token_limit: 5000                # optional response token cap
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb  # type: ignore[import-not-found,import-untyped]

from agentm.core.abi.messages import TextContent
from agentm.core.abi import FunctionTool, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="duckdb_sql",
    description="DuckDB SQL access to parquet files for RCA investigation.",
    registers=("tool:list_tables", "tool:query_sql"),
    config_schema={
        "type": "object",
        "properties": {
            "data_dir": {"type": "string"},
            "exclude": {"type": "array", "items": {"type": "string"}},
            "row_limit": {"type": "integer", "minimum": 1},
            "token_limit": {"type": "integer", "minimum": 100},
        },
        "additionalProperties": False,
    },
)

_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|COPY|"
    r"PRAGMA|EXPORT|IMPORT|INSTALL|LOAD|CALL|SET)\b",
    re.IGNORECASE,
)
_DEFAULT_TOKEN_LIMIT = 5000
_DEFAULT_ROW_LIMIT = 200


def _serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_serialize(item) for item in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def _estimate_tokens(text: str) -> int:
    # Cheap heuristic; close enough for budget enforcement.
    return (len(text) + 2) // 3


def _truncate(payload: str, *, token_limit: int, rows: list[dict[str, Any]]) -> str:
    if _estimate_tokens(payload) <= token_limit:
        return payload
    ratio = token_limit / max(1, _estimate_tokens(payload))
    keep = max(1, int(len(rows) * ratio * 0.8))
    out: dict[str, Any] = {
        "_truncated": True,
        "_total_rows": len(rows),
        "_rows_returned": keep,
        "_suggestion": "Add WHERE filters, narrower time range, or a smaller LIMIT.",
        "rows": rows[:keep],
    }
    return json.dumps(out, ensure_ascii=False, default=str, indent=2)


def _ok(text: str, *, is_error: bool = False) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


def _err(msg: str, **extra: Any) -> ToolResult:
    return _ok(json.dumps({"error": msg, **extra}, ensure_ascii=False), is_error=True)


def _table_name(filename: str) -> str:
    return Path(filename).stem


class _DuckDBState:
    def __init__(
        self,
        *,
        data_dir: Path,
        exclude: set[str],
        row_limit: int,
        token_limit: int,
    ) -> None:
        self.data_dir = data_dir
        self.exclude = exclude
        self.row_limit = row_limit
        self.token_limit = token_limit
        self.conn: duckdb.DuckDBPyConnection | None = None
        self.tables: list[str] = []

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self.conn is not None:
            return self.conn
        conn = duckdb.connect(":memory:")
        files = sorted(
            f.name
            for f in self.data_dir.iterdir()
            if f.is_file() and f.suffix == ".parquet" and f.name not in self.exclude
        )
        for fname in files:
            view = _table_name(fname)
            path = (self.data_dir / fname).as_posix().replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
            self.tables.append(view)
        self.conn = conn
        return conn

    def describe(self) -> list[dict[str, Any]]:
        conn = self.connect()
        result: list[dict[str, Any]] = []
        for view in self.tables:
            try:
                schema = conn.execute(f"DESCRIBE {view}").fetchall()
                count = conn.execute(f"SELECT count(*) FROM {view}").fetchone()
                result.append(
                    {
                        "table": view,
                        "row_count": int(count[0]) if count else 0,
                        "columns": [
                            {"name": col[0], "type": col[1]} for col in schema
                        ],
                    }
                )
            except duckdb.Error as exc:  # noqa: PERF203
                result.append({"table": view, "error": str(exc)})
        return result


def _resolve_data_dir(config: dict[str, Any]) -> Path:
    raw = config.get("data_dir") or os.environ.get("AGENTM_RCA_DATA_DIR")
    if not raw:
        raise ValueError(
            "duckdb_sql requires config.data_dir or AGENTM_RCA_DATA_DIR env var"
        )
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"data_dir is not a directory: {path}")
    return path


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    state = _DuckDBState(
        data_dir=_resolve_data_dir(config),
        exclude=set(config.get("exclude") or []),
        row_limit=int(config.get("row_limit") or _DEFAULT_ROW_LIMIT),
        token_limit=int(config.get("token_limit") or _DEFAULT_TOKEN_LIMIT),
    )

    async def _list(_: dict[str, Any]) -> ToolResult:
        try:
            tables = state.describe()
        except (duckdb.Error, OSError) as exc:
            return _err(f"list_tables failed: {exc}")
        payload = json.dumps(
            {"data_dir": str(state.data_dir), "tables": tables},
            ensure_ascii=False,
            default=str,
            indent=2,
        )
        return _ok(payload)

    async def _query(args: dict[str, Any]) -> ToolResult:
        sql_raw = str(args.get("sql", "")).strip()
        if not sql_raw:
            return _err("sql is required")
        sql = sql_raw.rstrip(";").strip()
        if ";" in sql:
            return _err("only one statement per call (no ';' inside SQL)")
        if _WRITE_KEYWORDS.search(sql):
            return _err("only read-only SELECT/WITH/EXPLAIN/DESCRIBE statements are allowed")
        head = sql.lstrip().split(None, 1)[0].upper() if sql.lstrip() else ""
        if head not in {"SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"}:
            return _err(
                f"unsupported leading keyword: {head!r}; "
                "use SELECT/WITH/EXPLAIN/DESCRIBE/SHOW/SUMMARIZE"
            )

        wrapped = (
            sql
            if head in {"EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"}
            else f"SELECT * FROM ({sql}) LIMIT {state.row_limit}"
        )

        try:
            conn = state.connect()
            cur = conn.execute(wrapped)
            cols = [d[0] for d in cur.description]
            rows = [
                dict(zip(cols, _serialize(list(r)), strict=False))
                for r in cur.fetchall()
            ]
        except duckdb.Error as exc:
            return _err(f"query failed: {exc}", sql=sql)

        body = json.dumps(
            {
                "sql": sql,
                "row_count": len(rows),
                "rows": rows,
                "row_limit_applied": state.row_limit
                if head in {"SELECT", "WITH"}
                else None,
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )
        return _ok(_truncate(body, token_limit=state.token_limit, rows=rows))

    api.register_tool(
        FunctionTool(
            name="list_tables",
            description=(
                "List the parquet files registered as DuckDB views, with row "
                "counts and column schemas. Call this first to discover what "
                "data is available."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            fn=_list,
        )
    )
    api.register_tool(
        FunctionTool(
            name="query_sql",
            description=(
                "Run a single read-only DuckDB SQL statement (SELECT / WITH / "
                "EXPLAIN / DESCRIBE / SHOW / SUMMARIZE) against the parquet "
                "views. SELECTs are auto-wrapped with a LIMIT; results are "
                "JSON-serialised and capped by a token budget."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "DuckDB SQL statement"},
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
            fn=_query,
        )
    )


__all__ = ["MANIFEST", "install"]
