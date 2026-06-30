"""Analyst trace tools — query any session by session_id.

Unlike the builtin trace_query (scoped to parent), these tools accept
a session_id parameter so the analyst can read arbitrary historical
sessions for comparative analysis.
"""

from __future__ import annotations

import json
from typing import Any, Final

from agentm.core.abi import ExtensionAPI, FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, Field


class TraceToolsConfig(BaseModel):
    pass


MANIFEST = ExtensionManifest(
    name="analyst_trace_tools",
    description="Trace query tools for arbitrary sessions by session_id.",
    registers=(
        "tool:list_turns",
        "tool:read_messages",
        "tool:get_tool_calls",
    ),
    config_schema=TraceToolsConfig,
)


def _text(s: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=s)])


def _get_backend() -> tuple[Any, str] | None:
    try:
        from agentm.core.observability import clickhouse
        url = clickhouse.get_url()
        if url is None:
            return None
        return clickhouse, url
    except Exception:
        return None


def install(api: ExtensionAPI, config: TraceToolsConfig) -> None:
    backend = _get_backend()
    if backend is None:
        logger.warning("analyst_trace_tools: ClickHouse unavailable")
        return

    ch, url = backend

    # -- list_turns --------------------------------------------------------

    class ListTurnsArgs(BaseModel):
        session_id: str = Field(description="The session ID to query")
        start: int = Field(default=0, description="Start turn index (inclusive)")
        limit: int = Field(default=50, description="Max turns to return")

    async def _list_turns(args: dict[str, Any]) -> ToolResult:
        parsed = ListTurnsArgs.model_validate(args)
        records = list(ch.turns(url, parsed.session_id))
        total = len(records)
        sliced = records[parsed.start : parsed.start + parsed.limit]
        lines = [f"Session {parsed.session_id[:12]}…: {total} turns"]
        for rec in sliced:
            idx = rec.get("turn_index", rec.get("turn_id", "?"))
            tools = rec.get("tool_calls", [])
            tool_str = ", ".join(tools) if tools else "—"
            tokens_in = rec.get("input_tokens", "?")
            tokens_out = rec.get("output_tokens", "?")
            lines.append(f"  [{idx}] tools=[{tool_str}] in={tokens_in} out={tokens_out}")
        return _text("\n".join(lines))

    api.register_tool(FunctionTool(
        name="list_turns",
        description="List turn summaries for a session. Shows tool calls and token counts per turn.",
        parameters=ListTurnsArgs,
        fn=_list_turns,
    ))

    # -- read_messages -----------------------------------------------------

    class ReadMessagesArgs(BaseModel):
        session_id: str = Field(description="The session ID to query")
        role: str | None = Field(
            default=None,
            description="Filter by role: user, assistant, tool_result, system",
        )
        limit: int = Field(default=20, description="Max messages to return")
        offset: int = Field(default=0, description="Skip this many messages")

    async def _read_messages(args: dict[str, Any]) -> ToolResult:
        parsed = ReadMessagesArgs.model_validate(args)
        roles = {parsed.role} if parsed.role else None
        records = list(ch.messages(url, parsed.session_id, roles=roles))
        total = len(records)
        sliced = records[parsed.offset : parsed.offset + parsed.limit]
        parts = [
            f"Session {parsed.session_id[:12]}…: {total} messages "
            f"(showing {parsed.offset}-{parsed.offset + len(sliced) - 1})"
        ]
        for rec in sliced:
            payload = rec.get("payload", {})
            role = payload.get("role", "?")
            content = payload.get("content", [])
            blocks: list[str] = []
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    btype = b.get("type", "")
                    if btype == "tool_call":
                        name = b.get("name", "")
                        a = json.dumps(b.get("arguments", {}), ensure_ascii=False)
                        blocks.append(f"[tool_call: {name}({a[:200]})]")
                    elif btype == "tool_result":
                        sub = b.get("content", [])
                        txt = ""
                        if isinstance(sub, list):
                            for s in sub:
                                if isinstance(s, dict):
                                    txt += s.get("text", "")
                        err = " ERROR" if b.get("is_error") else ""
                        blocks.append(f"[tool_result{err}] {txt[:500]}")
                    elif b.get("text"):
                        blocks.append(b["text"][:500])
            body = "\n".join(blocks) if blocks else "(empty)"
            parts.append(f"[{role}]\n{body}\n")
        return _text("\n".join(parts))

    api.register_tool(FunctionTool(
        name="read_messages",
        description="Read messages from a session. Filter by role, paginate with offset/limit.",
        parameters=ReadMessagesArgs,
        fn=_read_messages,
    ))

    # -- get_tool_calls ----------------------------------------------------

    class GetToolCallsArgs(BaseModel):
        session_id: str = Field(description="The session ID to query")
        tool_name: str | None = Field(default=None, description="Filter by tool name (e.g. 'edit', 'bash')")
        limit: int = Field(default=30, description="Max tool calls to return")

    async def _get_tool_calls(args: dict[str, Any]) -> ToolResult:
        parsed = GetToolCallsArgs.model_validate(args)
        records = list(ch.tools(url, parsed.session_id))
        if parsed.tool_name:
            records = [r for r in records if r.get("tool") == parsed.tool_name]
        total = len(records)
        sliced = records[: parsed.limit]
        parts = [
            f"Session {parsed.session_id[:12]}…: {total} tool calls"
            + (f" (filter: {parsed.tool_name})" if parsed.tool_name else "")
        ]
        for rec in sliced:
            name = rec.get("tool", "?")
            args_data = rec.get("args")
            result_data = rec.get("result")
            args_str = json.dumps(args_data, ensure_ascii=False)[:300] if args_data else "{}"
            result_preview = ""
            if isinstance(result_data, dict):
                content = result_data.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("text"):
                            result_preview = c["text"][:500]
                            break
            err_flag = ""
            if isinstance(result_data, dict) and result_data.get("is_error"):
                err_flag = " [ERROR]"
            parts.append(f"[{name}]{err_flag}\n  args: {args_str}\n  result: {result_preview[:300]}\n")
        return _text("\n".join(parts))

    api.register_tool(FunctionTool(
        name="get_tool_calls",
        description="Query tool calls from a session. Optionally filter by tool name.",
        parameters=GetToolCallsArgs,
        fn=_get_tool_calls,
    ))


__all__: Final = ["MANIFEST", "install"]
