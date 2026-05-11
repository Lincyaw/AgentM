"""ThinkDepthAI-parity tool atom for the rca:baseline scenario.

Mirrors the four-tool surface from
``/home/ddq/AoyangSpace/ThinkDepthAI/src/thinkdepthai/tools_lib/query_parquet_toolkit.py``
plus ``thinkdepthai/tools.py:think_tool`` so the AgentM baseline runs the
same agent contract as the reference implementation:

  * ``query_parquet_files(parquet_files, query, limit=10)``
  * ``get_schema(parquet_file)``
  * ``list_tables_in_directory(directory)``
  * ``think_tool(reasoning)``

Tool names, signatures, behaviors, the token-limit warning, and the
telemetry-only file allow-list (``log|trace|metric``) are kept identical so
the prompt's tool references match what the runtime exposes. The atom does
not register the broader ``query_sql`` / ``list_tables`` /
``hypothesis_*`` tooling — that is the point of baseline parity.
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

from agentm.core.abi import FunctionTool, ToolResult
from agentm.core.abi.messages import TextContent
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="thinkdepth_sql",
    description=(
        "ThinkDepthAI-parity SQL toolkit (query_parquet_files, get_schema, "
        "list_tables_in_directory, think_tool) for the rca:baseline scenario."
    ),
    registers=(
        "tool:query_parquet_files",
        "tool:get_schema",
        "tool:list_tables_in_directory",
        "tool:think_tool",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "data_dir": {"type": "string"},
            "token_limit": {"type": "integer", "minimum": 100},
        },
        "additionalProperties": False,
    },
)


# Match ThinkDepthAI's TOKEN_LIMIT exactly so identical queries return
# identically truncated payloads.
_TOKEN_LIMIT = 5000

# Same allow-list as ThinkDepthAI: only telemetry files (those whose name
# contains 'log', 'trace', or 'metric') are queryable. This blocks
# ``conclusion.parquet`` (ground-truth label) from leaking during eval.
_ALLOWED_PARQUET_RE = re.compile(r"(log|trace|metric)", re.IGNORECASE)


def _is_allowed_parquet(path: str | Path) -> bool:
    return _ALLOWED_PARQUET_RE.search(Path(path).stem) is not None


def _serialize_datetime(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj


def _estimate_token_count(text: str) -> int:
    average_chars_per_token = 3
    return (len(text) + average_chars_per_token - 1) // average_chars_per_token


def _enforce_token_limit(payload: str, context: str, token_limit: int) -> str:
    token_estimate = _estimate_token_count(payload)
    if token_estimate <= token_limit:
        return payload

    current_size: int | None = None
    if payload.startswith("["):
        try:
            current_size = len(json.loads(payload))
        except json.JSONDecodeError:
            current_size = None
    suggested_limit: int | None = None
    if current_size:
        ratio = token_limit / token_estimate
        suggested_limit = max(1, int(current_size * ratio * 0.8))

    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": token_estimate,
        "token_limit": token_limit,
        "rows_returned": current_size,
        "suggested_limit": suggested_limit,
        "suggestion": "Reduce LIMIT, add WHERE clauses, or use aggregations.",
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)


def _ok(text: str, *, is_error: bool = False) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


def _resolve_data_dir(config: dict[str, Any]) -> Path | None:
    raw = config.get("data_dir") or os.environ.get("AGENTM_RCA_DATA_DIR")
    if not raw:
        return None
    return Path(str(raw)).expanduser().resolve()


def _resolve_path(arg: str, default_dir: Path | None) -> str:
    """Accept absolute paths verbatim; resolve bare filenames against the
    case data directory.

    ThinkDepthAI's reference invocations from its prompt use bare file
    names like ``abnormal_traces.parquet``. AgentM's eval harness sets
    ``AGENTM_RCA_DATA_DIR`` per case; resolve relative names against it
    so the same prompt works without modification.
    """

    p = Path(arg)
    if p.is_absolute() or default_dir is None:
        return str(p)
    return str(default_dir / arg)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    token_limit = int(config.get("token_limit") or _TOKEN_LIMIT)
    default_dir = _resolve_data_dir(config)

    async def _query_parquet_files(args: dict[str, Any]) -> ToolResult:
        files_arg = args.get("parquet_files")
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", 10))
        if not query:
            return _ok(json.dumps({"error": "query is required"}), is_error=True)
        if isinstance(files_arg, str):
            parquet_files: list[str] = [files_arg]
        elif isinstance(files_arg, list):
            parquet_files = [str(f) for f in files_arg]
        else:
            return _ok(
                json.dumps({"error": "parquet_files must be a string or list of strings"}),
                is_error=True,
            )

        resolved: list[str] = [_resolve_path(f, default_dir) for f in parquet_files]
        for fp in resolved:
            if not _is_allowed_parquet(fp):
                return _ok(
                    json.dumps(
                        {
                            "error": (
                                f"Access denied: {Path(fp).name} is not a telemetry "
                                "parquet. Only files whose name contains 'log', "
                                "'trace', or 'metric' are queryable."
                            )
                        }
                    ),
                    is_error=True,
                )
            if not Path(fp).exists():
                return _ok(
                    json.dumps({"error": f"Parquet file not found: {fp}"}),
                    is_error=True,
                )

        conn = duckdb.connect(":memory:")
        table_names: set[str] = set()
        try:
            for file_path in resolved:
                base_name = Path(file_path).stem
                table_name = base_name
                counter = 1
                while table_name in table_names:
                    table_name = f"{base_name}_{counter}"
                    counter += 1
                table_names.add(table_name)
                escaped = file_path.replace("'", "''")
                conn.execute(
                    f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{escaped}')"
                )

            cur = conn.execute(query)
            columns = [desc[0] for desc in cur.description]
            rows_raw = cur.fetchall()
            rows = [dict(zip(columns, row, strict=False)) for row in rows_raw]
            serialized_rows = _serialize_datetime(rows)

            if isinstance(serialized_rows, list) and len(serialized_rows) > limit:
                serialized_rows = serialized_rows[:limit]

            result_json = json.dumps(serialized_rows, ensure_ascii=False, indent=2)
            return _ok(_enforce_token_limit(result_json, "query_parquet_files", token_limit))
        except duckdb.Error as exc:
            return _ok(
                json.dumps(
                    {
                        "error": f"Query execution failed: {exc}",
                        "query": query,
                        "available_tables": list(table_names),
                    }
                ),
                is_error=True,
            )
        finally:
            conn.close()

    async def _get_schema(args: dict[str, Any]) -> ToolResult:
        parquet_file = _resolve_path(str(args.get("parquet_file", "")), default_dir)
        if not parquet_file or parquet_file == ".":
            return _ok(json.dumps({"error": "parquet_file is required"}), is_error=True)
        if not _is_allowed_parquet(parquet_file):
            return _ok(
                json.dumps(
                    {
                        "error": (
                            f"Access denied: {Path(parquet_file).name} is not a "
                            "telemetry parquet. Only files whose name contains "
                            "'log', 'trace', or 'metric' are inspectable."
                        )
                    }
                ),
                is_error=True,
            )
        if not Path(parquet_file).exists():
            return _ok(
                json.dumps({"error": f"Parquet file not found: {parquet_file}"}),
                is_error=True,
            )

        conn = duckdb.connect(":memory:")
        try:
            escaped = parquet_file.replace("'", "''")
            cur = conn.execute(f"SELECT * FROM read_parquet('{escaped}') LIMIT 0")
            schema = [{"name": desc[0], "type": str(desc[1])} for desc in cur.description]
            row_count_result = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{escaped}')"
            ).fetchone()
            row_count = int(row_count_result[0]) if row_count_result else 0
            schema_info = {"file": parquet_file, "row_count": row_count, "columns": schema}
            result_json = json.dumps(schema_info, ensure_ascii=False, indent=2)
            return _ok(_enforce_token_limit(result_json, "get_schema", token_limit))
        except duckdb.Error as exc:
            return _ok(
                json.dumps({"error": f"Failed to extract schema: {exc}", "file": parquet_file}),
                is_error=True,
            )
        finally:
            conn.close()

    async def _list_tables_in_directory(args: dict[str, Any]) -> ToolResult:
        directory_arg = str(args.get("directory", "")).strip()
        if not directory_arg and default_dir is not None:
            directory = str(default_dir)
        else:
            directory = directory_arg
        dir_path = Path(directory)
        if not dir_path.exists():
            return _ok(
                json.dumps({"error": f"Directory not found: {directory}"}),
                is_error=True,
            )
        if not dir_path.is_dir():
            return _ok(
                json.dumps({"error": f"Path is not a directory: {directory}"}),
                is_error=True,
            )

        files_info: list[dict[str, Any]] = []
        for file_path in sorted(dir_path.glob("*.parquet")):
            if not _is_allowed_parquet(file_path):
                continue
            try:
                conn = duckdb.connect(":memory:")
                escaped = str(file_path).replace("'", "''")
                row_count_result = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{escaped}')"
                ).fetchone()
                row_count = int(row_count_result[0]) if row_count_result else 0
                cur = conn.execute(f"SELECT * FROM read_parquet('{escaped}') LIMIT 0")
                column_count = len(cur.description)
                conn.close()
                files_info.append(
                    {
                        "filename": file_path.name,
                        "path": str(file_path),
                        "row_count": row_count,
                        "column_count": column_count,
                    }
                )
            except duckdb.Error as exc:
                files_info.append(
                    {"filename": file_path.name, "path": str(file_path), "error": str(exc)}
                )

        result_json = json.dumps(files_info, ensure_ascii=False, indent=2)
        return _ok(_enforce_token_limit(result_json, "list_tables_in_directory", token_limit))

    async def _think_tool(args: dict[str, Any]) -> ToolResult:
        reasoning = str(args.get("reasoning", ""))
        return _ok(
            json.dumps(
                {
                    "status": "recorded",
                    "reasoning": reasoning,
                    "next_action": "Continue with planned investigation",
                }
            )
        )

    api.register_tool(
        FunctionTool(
            name="query_parquet_files",
            description=(
                "Query parquet files using SQL syntax for data analysis and "
                "exploration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "parquet_files": {
                        "description": "Path(s) to parquet file(s)",
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of records to return (default: 10)",
                    },
                },
                "required": ["parquet_files", "query"],
                "additionalProperties": False,
            },
            fn=_query_parquet_files,
        )
    )
    api.register_tool(
        FunctionTool(
            name="get_schema",
            description="Get schema information of a parquet file.",
            parameters={
                "type": "object",
                "properties": {
                    "parquet_file": {
                        "type": "string",
                        "description": "Path to parquet file to inspect",
                    },
                },
                "required": ["parquet_file"],
                "additionalProperties": False,
            },
            fn=_get_schema,
        )
    )
    api.register_tool(
        FunctionTool(
            name="list_tables_in_directory",
            description=(
                "List all parquet files in a directory with metadata."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to search for parquet files",
                    },
                },
                "required": ["directory"],
                "additionalProperties": False,
            },
            fn=_list_tables_in_directory,
        )
    )
    api.register_tool(
        FunctionTool(
            name="think_tool",
            description=(
                "Reflect on findings and plan next investigation steps. Use "
                "this tool to analyze findings so far and decide what to "
                "investigate next."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Your analysis and reasoning about findings and next steps"
                        ),
                    },
                },
                "required": ["reasoning"],
                "additionalProperties": False,
            },
            fn=_think_tool,
        )
    )


__all__ = ["MANIFEST", "install"]
