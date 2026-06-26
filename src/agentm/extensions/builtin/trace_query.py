"""Builtin ``trace_query`` atom — query the parent session's trajectory.

Exposes read-only tools that let a child session inspect its parent's
conversation trajectory via the ClickHouse trace backend.  Scoped
automatically to ``api.parent_session_id`` — no session id parameter
needed, and no access to other sessions.

Typical consumer: the ``goal`` atom's checker agent, which needs to
verify whether the parent agent's work satisfies a completion condition.

§11: single file; ``MANIFEST`` + ``install(api, config)``; no atom-to-atom
imports; ``core.abi`` only; no ``core.runtime.*`` / ``core._internal``.
"""

from __future__ import annotations

import json
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel, Field

from agentm.core.abi import ExtensionAPI, FunctionTool, TextContent, ToolResult, ToolTerminate
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="trace_query",
    description=(
        "Read-only trajectory query tools scoped to the parent session. "
        "Lets a child agent inspect what its parent did: list turns, "
        "read specific messages, query tool calls."
    ),
    registers=(
        "tool:list_turns",
        "tool:read_turn",
        "tool:get_tool_calls",
        "tool:submit_verdict",
    ),
    requires=(),
)


def _text(s: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=s)])


def _get_backend() -> tuple[Any, str] | None:
    """Return (clickhouse_module, url) or None."""
    try:
        from agentm.core.observability import clickhouse
        url = clickhouse.get_url()
        if url is None:
            return None
        return clickhouse, url
    except Exception:  # noqa: BLE001
        return None


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    parent_sid = api.parent_session_id
    if parent_sid is None:
        logger.debug("trace_query: no parent session — tools will return empty results")
        return

    backend = _get_backend()
    if backend is None:
        logger.warning("trace_query: ClickHouse unavailable — tools disabled")
        return

    ch, url = backend

    # -- list_turns --------------------------------------------------------

    class ListTurnsArgs(BaseModel):
        start: int = Field(default=0, description="Start turn index (inclusive)")
        limit: int = Field(default=50, description="Max turns to return")

    async def _list_turns(args: dict[str, Any]) -> ToolResult:
        parsed = ListTurnsArgs.model_validate(args)
        records = list(ch.turns(url, parent_sid))
        total = len(records)
        sliced = records[parsed.start:parsed.start + parsed.limit]
        lines = [f"Parent session: {total} turns total"]
        for rec in sliced:
            idx = rec.get("turn_index", rec.get("turn_id", "?"))
            tools = rec.get("tool_calls", [])
            tool_str = ", ".join(tools) if tools else "—"
            tokens_in = rec.get("input_tokens", "?")
            tokens_out = rec.get("output_tokens", "?")
            lines.append(
                f"  [{idx}] tools=[{tool_str}] "
                f"in={tokens_in} out={tokens_out}"
            )
        return _text("\n".join(lines))

    api.register_tool(FunctionTool(
        name="list_turns",
        description=(
            "List turn-level summaries of the parent session's trajectory. "
            "Shows which tools were called each turn with token counts. "
            "Use this first to get an overview, then drill into specific "
            "turns with read_turn."
        ),
        parameters=ListTurnsArgs,
        fn=_list_turns,
    ))

    # -- read_turn ---------------------------------------------------------

    class ReadTurnArgs(BaseModel):
        role: str | None = Field(
            default=None,
            description="Filter by message role: user, assistant, tool_result, system",
        )
        limit: int = Field(default=20, description="Max messages to return")
        offset: int = Field(default=0, description="Skip this many messages")

    async def _read_turn(args: dict[str, Any]) -> ToolResult:
        parsed = ReadTurnArgs.model_validate(args)
        roles = {parsed.role} if parsed.role else None
        records = list(ch.messages(url, parent_sid, roles=roles))
        total = len(records)
        sliced = records[parsed.offset:parsed.offset + parsed.limit]
        parts = [f"Messages: {total} total (showing {parsed.offset}-{parsed.offset + len(sliced) - 1})"]
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
                        blocks.append(f"[tool_call: {name}({a})]")
                    elif btype == "tool_result":
                        sub = b.get("content", [])
                        txt = ""
                        if isinstance(sub, list):
                            for s in sub:
                                if isinstance(s, dict):
                                    txt += s.get("text", "")
                        err = " ERROR" if b.get("is_error") else ""
                        blocks.append(f"[tool_result{err}] {txt}")
                    elif b.get("text"):
                        blocks.append(b["text"])
            body = "\n".join(blocks) if blocks else "(empty)"
            parts.append(f"[{role}]\n{body}\n")
        return _text("\n".join(parts))

    api.register_tool(FunctionTool(
        name="read_turn",
        description=(
            "Read messages from the parent session's trajectory. "
            "Filter by role (user/assistant/tool_result/system) and "
            "paginate with offset/limit. Use this to inspect what the "
            "agent actually did and what results it received."
        ),
        parameters=ReadTurnArgs,
        fn=_read_turn,
    ))

    # -- get_tool_calls ----------------------------------------------------

    class GetToolCallsArgs(BaseModel):
        tool_name: str | None = Field(
            default=None,
            description="Filter by tool name (e.g. 'submit_final_report', 'query_sql')",
        )
        limit: int = Field(default=30, description="Max tool calls to return")

    async def _get_tool_calls(args: dict[str, Any]) -> ToolResult:
        parsed = GetToolCallsArgs.model_validate(args)
        records = list(ch.tools(url, parent_sid))
        if parsed.tool_name:
            records = [r for r in records if r.get("tool") == parsed.tool_name]
        total = len(records)
        sliced = records[:parsed.limit]
        parts = [f"Tool calls: {total} total" + (f" (filter: {parsed.tool_name})" if parsed.tool_name else "")]
        for rec in sliced:
            name = rec.get("tool", "?")
            args_data = rec.get("args")
            result_data = rec.get("result")
            args_str = json.dumps(args_data, ensure_ascii=False) if args_data else "{}"
            result_preview = ""
            if isinstance(result_data, dict):
                content = result_data.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("text"):
                            result_preview = c["text"][:500]
                            break
            parts.append(
                f"[{name}]\n  args: {args_str}\n  result: {result_preview}\n"
            )
        return _text("\n".join(parts))

    api.register_tool(FunctionTool(
        name="get_tool_calls",
        description=(
            "Query tool calls from the parent session. "
            "Optionally filter by tool name. Shows arguments and result "
            "previews. Use this to check specific tool outputs like "
            "submit_final_report."
        ),
        parameters=GetToolCallsArgs,
        fn=_get_tool_calls,
    ))

    # -- submit_verdict (terminates checker session) -----------------------

    class SubmitVerdictArgs(BaseModel):
        met: bool = Field(description="True if the goal condition is met, False otherwise")
        reason: str = Field(description="One-line explanation of why the condition is or is not met")
        unexplained: list[str] = Field(
            default_factory=list,
            description="List of specific items not covered (empty if met)",
        )

    async def _submit_verdict(args: dict[str, Any]) -> ToolTerminate:
        parsed = SubmitVerdictArgs.model_validate(args)
        return ToolTerminate(
            result=ToolResult(content=[TextContent(
                type="text",
                text=json.dumps({"met": parsed.met, "reason": parsed.reason,
                                 "unexplained": parsed.unexplained},
                                ensure_ascii=False),
            )]),
            reason="goal_checker:verdict_submitted",
        )

    api.register_tool(FunctionTool(
        name="submit_verdict",
        description=(
            "Submit your final verdict on whether the goal condition is met. "
            "This terminates the checker session. Call this ONLY after you "
            "have completed all verification steps."
        ),
        parameters=SubmitVerdictArgs,
        fn=_submit_verdict,
    ))


__all__: Final = ["MANIFEST", "install"]
