# code-health: ignore-file[AM025] -- atom tools validate untyped tool, config, and service payloads
"""Builtin ``trace_query`` atom -- query the parent session's trajectory.

Exposes read-only tools that let a child session inspect its parent's
conversation trajectory.  Scoped automatically to ``api.ctx.parent_session_id``.

Typical consumer: the ``goal`` atom's checker agent, which needs to
verify whether the parent agent's work satisfies a completion condition.
"""

from __future__ import annotations

import json

from loguru import logger
from pydantic import BaseModel, Field

from agentm.core.abi import (
    AssistantMessage,
    AtomAPI,
    AtomInstallPriority,
    FunctionTool,
    JsonValue,
    TextContent,
    ToolCallBlock,
    ToolResult,
)
from agentm.core.abi.trajectory import Turn
from agentm.core.lib import pydantic_to_tool_schema, text_result
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="trace_query",
    description=(
        "Read-only trajectory query tools scoped to the parent session. "
        "Lets a child agent inspect what its parent did."
    ),
    registers=(
        "tool:list_turns",
        "tool:read_turn",
        "tool:get_tool_calls",
    ),
    requires=(),
    priority=AtomInstallPriority.TOOL,
)


# ---------------------------------------------------------------------------
# Turn loading -- read parent turns from the trajectory store
# ---------------------------------------------------------------------------


def _load_parent_turns(api: AtomAPI) -> list[Turn] | None:
    parent_sid = api.ctx.parent_session_id
    if parent_sid is None:
        return None
    store = api.store
    if store is None:
        return None
    try:
        if not store.session_exists(parent_sid):
            return None
        _, turns = store.load(parent_sid)
        return turns
    except Exception as exc:  # noqa: BLE001
        logger.debug("trace_query: failed to load parent turns: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _turn_summary(turn: Turn) -> dict[str, object]:
    tool_names: list[str] = []
    if turn.response is not None:
        for block in turn.response.content:
            if isinstance(block, ToolCallBlock):
                tool_names.append(block.name)
    return {
        "turn_index": turn.index,
        "run_id": turn.run_id,
        "run_step": turn.run_step,
        "tool_calls": tool_names,
        "input_tokens": turn.meta.total_input_tokens,
        "output_tokens": turn.meta.total_output_tokens,
    }


def _message_records(turns: list[Turn]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for turn in turns:
        if turn.response is not None:
            records.append(_assistant_record(turn.response, turn))
        for tr in turn.tool_results:
            records.append(
                {
                    "role": "tool_result",
                    "turn_index": turn.index,
                    "run_id": turn.run_id,
                    "run_step": turn.run_step,
                    "blocks": [
                        {"text": c.text if isinstance(c, TextContent) else str(c)}
                        for c in (tr.result.content if tr.result else [])
                    ],
                    "is_error": tr.result.is_error if tr.result else False,
                    "tool_name": tr.call.name,
                }
            )
    return records


def _assistant_record(msg: AssistantMessage, turn: Turn) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCallBlock):
            blocks.append(
                {
                    "type": "tool_call",
                    "name": block.name,
                    "arguments": block.arguments,
                }
            )
    return {
        "role": "assistant",
        "turn_index": turn.index,
        "run_id": turn.run_id,
        "run_step": turn.run_step,
        "blocks": blocks,
    }


def _tool_call_records(turns: list[Turn]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for turn in turns:
        for tr in turn.tool_results:
            result_text = ""
            if tr.result:
                for c in tr.result.content:
                    if isinstance(c, TextContent):
                        result_text += c.text
            records.append(
                {
                    "turn_index": turn.index,
                    "run_id": turn.run_id,
                    "run_step": turn.run_step,
                    "tool": tr.call.name,
                    "args": tr.call.arguments,
                    "result_preview": result_text[:500],
                    "is_error": tr.result.is_error if tr.result else False,
                }
            )
    return records


def _clip(text: str, limit: int, full: bool) -> str:
    if full or len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated {len(text) - limit} chars]"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class _ListTurnsArgs(BaseModel):
    start: int = Field(default=0, description="Start turn index (inclusive)")
    limit: int = Field(default=50, description="Max turns to return")


class _ReadTurnArgs(BaseModel):
    role: str | None = Field(
        default=None,
        description="Filter by message role: assistant, tool_result",
    )
    limit: int = Field(default=20, description="Max messages to return")
    offset: int = Field(default=0, description="Skip this many messages")
    full: bool = Field(
        default=False,
        description="false: clipped previews. true: full content.",
    )


class _GetToolCallsArgs(BaseModel):
    tool_name: str | None = Field(
        default=None,
        description="Filter by tool name",
    )
    limit: int = Field(default=30, description="Max tool calls to return")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class _TraceQueryRuntime:
    def __init__(self, api: AtomAPI, turns: list[Turn]) -> None:
        self._api = api
        self._turns = turns

    def install(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="list_turns",
                description=(
                    "List turn-level summaries of the parent session. "
                    "Shows which tools were called each turn with token counts."
                ),
                parameters=pydantic_to_tool_schema(_ListTurnsArgs),
                fn=self.list_turns,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="read_turn",
                description=(
                    "Page through the parent session's messages. "
                    "Paginate with offset/limit, optionally filter by role."
                ),
                parameters=pydantic_to_tool_schema(_ReadTurnArgs),
                fn=self.read_turn,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="get_tool_calls",
                description=(
                    "Query tool calls from the parent session. "
                    "Optionally filter by tool name."
                ),
                parameters=pydantic_to_tool_schema(_GetToolCallsArgs),
                fn=self.get_tool_calls,
            )
        )

    def _refresh_turns(self) -> list[Turn]:
        fresh = _load_parent_turns(self._api)
        if fresh is not None:
            self._turns = fresh
        return self._turns

    async def list_turns(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _ListTurnsArgs.model_validate(args)
        turns = self._refresh_turns()
        summaries = [_turn_summary(t) for t in turns]
        sliced = summaries[parsed.start : parsed.start + parsed.limit]
        lines = [f"Parent session: {len(summaries)} turns total"]
        for s in sliced:
            tool_calls = s["tool_calls"]
            tool_str = (
                ", ".join(tool_calls) if isinstance(tool_calls, list) else ""
            ) or "—"
            lines.append(
                f"  [{s['turn_index']}] run={str(s['run_id'])[:8]} "
                f"step={s['run_step']} tools=[{tool_str}] "
                f"in={s['input_tokens']} out={s['output_tokens']}"
            )
        return text_result("\n".join(lines))

    async def read_turn(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _ReadTurnArgs.model_validate(args)
        turns = self._refresh_turns()
        records = _message_records(turns)
        if parsed.role:
            records = [r for r in records if r["role"] == parsed.role]
        total = len(records)
        sliced = records[parsed.offset : parsed.offset + parsed.limit]
        parts = [
            f"Messages: {total} total (showing {parsed.offset}-{parsed.offset + len(sliced) - 1})"
        ]
        for rec in sliced:
            role = rec["role"]
            blocks = rec.get("blocks", [])
            rendered: list[str] = []
            for b in blocks if isinstance(blocks, list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_call":
                    a = json.dumps(b.get("arguments", {}), ensure_ascii=False)
                    rendered.append(
                        f"[tool_call: {b.get('name', '')}({_clip(a, 500, parsed.full)})]"
                    )
                elif b.get("type") == "tool_result" or "text" in b:
                    rendered.append(_clip(b.get("text", ""), 1500, parsed.full))
            body = "\n".join(rendered) if rendered else "(empty)"
            parts.append(f"[{role}]\n{body}\n")
        return text_result("\n".join(parts))

    async def get_tool_calls(self, args: dict[str, JsonValue]) -> ToolResult:
        parsed = _GetToolCallsArgs.model_validate(args)
        turns = self._refresh_turns()
        records = _tool_call_records(turns)
        if parsed.tool_name:
            records = [r for r in records if r["tool"] == parsed.tool_name]
        total = len(records)
        sliced = records[: parsed.limit]
        parts = [
            f"Tool calls: {total} total"
            + (f" (filter: {parsed.tool_name})" if parsed.tool_name else "")
        ]
        for rec in sliced:
            args_str = json.dumps(rec.get("args", {}), ensure_ascii=False)
            parts.append(
                f"[{rec['tool']}]\n"
                f"  args: {args_str}\n"
                f"  result: {rec.get('result_preview', '')}\n"
            )
        return text_result("\n".join(parts))


def install(api: AtomAPI, config: dict[str, JsonValue] | None = None) -> None:
    del config
    turns = _load_parent_turns(api)
    if turns is None:
        logger.debug("trace_query: no parent trajectory available — tools disabled")
        return
    _TraceQueryRuntime(api, turns).install()
