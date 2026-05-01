"""DuckDB-backed RCA observability tools exposed as AgentM tool atoms."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from agentm.core.kernel import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from ._core import (
    ALLOWED_TABLE_FILES,
    QueryError,
    TOKEN_LIMIT,
    close_obs_connection,
    set_data_directory,
)
from ._deployment import get_deployment_graph
from ._logs import search_logs_abnormal, search_logs_normal
from ._metrics import query_metrics_ohlc_abnormal, query_metrics_ohlc_normal
from ._traces import (
    get_service_call_graph_abnormal,
    get_service_call_graph_normal,
    get_span_call_graph_abnormal,
    get_span_call_graph_normal,
    query_trace_stats_abnormal,
    query_trace_stats_normal,
)

ToolHandler = Callable[..., Awaitable[str]]

_MANIFEST_REGISTERS = (
    "tool:query_metrics_ohlc_abnormal",
    "tool:query_metrics_ohlc_normal",
    "tool:query_trace_stats_abnormal",
    "tool:query_trace_stats_normal",
    "tool:get_service_call_graph_abnormal",
    "tool:get_service_call_graph_normal",
    "tool:get_span_call_graph_abnormal",
    "tool:get_span_call_graph_normal",
    "tool:get_deployment_graph",
    "tool:search_logs_abnormal",
    "tool:search_logs_normal",
)

MANIFEST = ExtensionManifest(
    name="rca_observability",
    description="Register RCA observability tools backed by DuckDB parquet queries.",
    registers=_MANIFEST_REGISTERS,
    config_schema={
        "type": "object",
        "properties": {
            "data_dir": {"type": "string"},
        },
        "required": ["data_dir"],
        "additionalProperties": False,
    },
)

_TOOL_SPECS: tuple[tuple[str, str, dict[str, Any], ToolHandler], ...] = (
    (
        "query_metrics_ohlc_abnormal",
        "Query abnormal-period metrics as OHLC summaries per time bucket.",
        {
            "type": "object",
            "properties": {
                "metric_name": {"type": "string"},
                "interval": {"type": "string", "enum": ["1m", "5m", "15m"], "default": "5m"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "filters": {"type": "string"},
            },
            "required": ["metric_name"],
            "additionalProperties": False,
        },
        query_metrics_ohlc_abnormal,
    ),
    (
        "query_metrics_ohlc_normal",
        "Query normal-period metrics as OHLC summaries per time bucket.",
        {
            "type": "object",
            "properties": {
                "metric_name": {"type": "string"},
                "interval": {"type": "string", "enum": ["1m", "5m", "15m"], "default": "5m"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "filters": {"type": "string"},
            },
            "required": ["metric_name"],
            "additionalProperties": False,
        },
        query_metrics_ohlc_normal,
    ),
    (
        "query_trace_stats_abnormal",
        "Query abnormal-period trace latency and error stats.",
        {
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "group_by": {"type": "string", "enum": ["service_name", "span_name"], "default": "service_name"},
                "interval": {"type": "string", "enum": ["1m", "5m", "15m"], "default": "5m"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "filters": {"type": "string"},
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        query_trace_stats_abnormal,
    ),
    (
        "query_trace_stats_normal",
        "Query normal-period trace latency and error stats.",
        {
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "group_by": {"type": "string", "enum": ["service_name", "span_name"], "default": "service_name"},
                "interval": {"type": "string", "enum": ["1m", "5m", "15m"], "default": "5m"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "filters": {"type": "string"},
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        query_trace_stats_normal,
    ),
    (
        "get_service_call_graph_abnormal",
        "Build the abnormal-period service call graph from trace data.",
        {
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "filters": {"type": "string"},
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        get_service_call_graph_abnormal,
    ),
    (
        "get_service_call_graph_normal",
        "Build the normal-period service call graph from trace data.",
        {
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "filters": {"type": "string"},
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        get_service_call_graph_normal,
    ),
    (
        "get_span_call_graph_abnormal",
        "Return the abnormal-period span tree for one trace.",
        {
            "type": "object",
            "properties": {
                "trace_id": {"type": "string"},
            },
            "required": ["trace_id"],
            "additionalProperties": False,
        },
        get_span_call_graph_abnormal,
    ),
    (
        "get_span_call_graph_normal",
        "Return the normal-period span tree for one trace.",
        {
            "type": "object",
            "properties": {
                "trace_id": {"type": "string"},
            },
            "required": ["trace_id"],
            "additionalProperties": False,
        },
        get_span_call_graph_normal,
    ),
    (
        "get_deployment_graph",
        "Return the deployment topology derived from abnormal metrics.",
        {
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "service_name": {"type": "string"},
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        get_deployment_graph,
    ),
    (
        "search_logs_abnormal",
        "Search abnormal-period logs.",
        {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "match_mode": {"type": "string", "enum": ["contains", "exact", "regex"], "default": "contains"},
                "level": {"type": "string"},
                "service_name": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
        search_logs_abnormal,
    ),
    (
        "search_logs_normal",
        "Search normal-period logs.",
        {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "match_mode": {"type": "string", "enum": ["contains", "exact", "regex"], "default": "contains"},
                "level": {"type": "string"},
                "service_name": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
        search_logs_normal,
    ),
)


def _ok(text: str, *, is_error: bool = False) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


def _payload_is_error(payload: str) -> bool:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and "error" in parsed


def _ensure_data_dir(data_dir: str) -> None:
    response = json.loads(set_data_directory(data_dir))
    if isinstance(response, dict) and "error" in response:
        raise ValueError(str(response["error"]))


def _build_function_tool(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: ToolHandler,
    data_dir: str,
) -> FunctionTool:
    async def _execute(args: dict[str, Any]) -> ToolResult:
        _ensure_data_dir(data_dir)
        payload = await handler(**args)
        return _ok(payload, is_error=_payload_is_error(payload))

    return FunctionTool(
        name=name,
        description=description,
        parameters=parameters,
        fn=_execute,
        metadata={"package": "agentm-rca"},
    )


def build_observability_tools(*, data_dir: str) -> list[FunctionTool]:
    _ensure_data_dir(data_dir)
    return [
        _build_function_tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            data_dir=data_dir,
        )
        for name, description, parameters, handler in _TOOL_SPECS
    ]


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    data_dir = str(config.get("data_dir", ""))
    for tool in build_observability_tools(data_dir=data_dir):
        api.register_tool(tool)


__all__ = [
    "ALLOWED_TABLE_FILES",
    "MANIFEST",
    "QueryError",
    "TOKEN_LIMIT",
    "build_observability_tools",
    "close_obs_connection",
    "get_deployment_graph",
    "get_service_call_graph_abnormal",
    "get_service_call_graph_normal",
    "get_span_call_graph_abnormal",
    "get_span_call_graph_normal",
    "install",
    "query_metrics_ohlc_abnormal",
    "query_metrics_ohlc_normal",
    "query_trace_stats_abnormal",
    "query_trace_stats_normal",
    "search_logs_abnormal",
    "search_logs_normal",
    "set_data_directory",
]
