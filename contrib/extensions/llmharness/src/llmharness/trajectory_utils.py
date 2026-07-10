"""Shared trajectory-processing helpers used by both runtime and eval."""

from __future__ import annotations

from typing import Any

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
)


def serialize_trajectory(
    messages: list[AgentMessage],
    *,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    from agentm.core.lib import to_jsonable

    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages, start=start_index):
        d = to_jsonable(msg)
        if isinstance(d, dict):
            d["index"] = i
            out.append(d)
    return out


def extract_loaded_skills(messages: list[AgentMessage]) -> list[str]:
    """Extract text content from all load_skill tool results in the conversation."""
    skill_call_ids = _tool_call_ids(messages, "load_skill")
    if not skill_call_ids:
        return []
    skills: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolResultMessage):
            for block in msg.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                if block.tool_call_id not in skill_call_ids or block.is_error:
                    continue
                text = " ".join(
                    inner.text
                    for inner in block.content
                    if isinstance(inner, TextContent) and inner.text
                )
                if text:
                    skills.append(text)
    return skills


def _tool_call_ids(messages: list[AgentMessage], tool_name: str) -> set[str]:
    call_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                call_ids.add(block.id)
    return call_ids
