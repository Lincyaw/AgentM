"""Shared headless rendering helpers for presenter surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
)
from agentm.core.abi.tool import TOOL_RESULT_FORMAT_METADATA_KEY, ToolResult

RESULT_FORMAT_METADATA_KEY = TOOL_RESULT_FORMAT_METADATA_KEY


class ToolResultRenderer(Protocol):
    """Optional text renderer registered by an extension for one tool."""

    def __call__(self, result: ToolResult | ToolResultBlock | ToolResultMessage) -> str: ...


@dataclass(frozen=True, slots=True)
class UsageReport:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    assistant_turns: int = 0


@dataclass(frozen=True, slots=True)
class FinalReport:
    text: str
    message_count: int
    tool_calls: int
    usage: UsageReport


def assistant_text(msg: AssistantMessage) -> str:
    return "\n".join(block.text for block in msg.content if isinstance(block, TextContent))


def tool_result_text(
    result: ToolResult | ToolResultBlock | ToolResultMessage,
    *,
    tool_name: str | None = None,
    renderers: Mapping[str, ToolResultRenderer] | None = None,
) -> str:
    if tool_name is not None and renderers is not None:
        renderer = renderers.get(tool_name)
        if renderer is not None:
            return renderer(result)
    parts: list[str] = []
    for block in _tool_result_content(result):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n\n".join(parts)


def usage_summary(messages: Iterable[AssistantMessage]) -> UsageReport:
    input_tokens = output_tokens = cache_read = cache_write = assistant_turns = 0
    for msg in messages:
        usage = msg.usage
        if usage is None:
            continue
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        cache_read += usage.cache_read
        cache_write += usage.cache_write
        assistant_turns += 1
    return UsageReport(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        assistant_turns=assistant_turns,
    )


def final_summary(messages: Iterable[AgentMessage]) -> FinalReport:
    materialized = list(messages)
    assistant_messages = [msg for msg in materialized if isinstance(msg, AssistantMessage)]
    text = "\n\n".join(
        chunk for msg in assistant_messages if (chunk := assistant_text(msg))
    )
    tool_calls = sum(
        1
        for msg in assistant_messages
        for block in msg.content
        if isinstance(block, ToolCallBlock)
    )
    return FinalReport(
        text=text,
        message_count=len(materialized),
        tool_calls=tool_calls,
        usage=usage_summary(assistant_messages),
    )


def _tool_result_content(
    result: ToolResult | ToolResultBlock | ToolResultMessage,
) -> list[Any]:
    if isinstance(result, ToolResultMessage):
        content: list[Any] = []
        for block in result.content:
            content.extend(block.content)
        return content
    raw_content = getattr(result, "content", None)
    if isinstance(raw_content, list):
        return cast(list[Any], raw_content)
    return []


__all__ = [
    "FinalReport",
    "RESULT_FORMAT_METADATA_KEY",
    "ToolResultRenderer",
    "UsageReport",
    "assistant_text",
    "final_summary",
    "tool_result_text",
    "usage_summary",
]
