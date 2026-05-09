"""Shared helpers for provider-internal LLM stream adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentm.core.abi.messages import (
    AssistantContent,
    AssistantMessage,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    Usage,
)
from agentm.core.abi.stream import ToolCallArgsParseError
from agentm.core.abi.termination import TerminationHint
from agentm.core.abi.tool import Tool


def encode_tool_args(args: Mapping[str, Any]) -> str:
    """Encode tool arguments consistently across provider adapters."""

    return json.dumps(dict(args), ensure_ascii=False)


class ToolSpecAdapter(Protocol):
    """Provider-specific conversion from AgentM tools to vendor specs."""

    def vendor_spec(self, tool: Tool) -> dict[str, Any]: ...

    def encode_tool_args(self, args: Mapping[str, Any]) -> str: ...


@dataclass
class _ContentEntry:
    kind: str
    order: int
    index: int | None = None
    text: str = ""
    signature: str | None = None
    tool_call_id: str = ""
    tool_name: str = ""
    args_json: str = ""


@dataclass
class StreamAccumulator:
    """Builds an ``AssistantMessage`` from provider-specific stream events."""

    _entries: list[_ContentEntry] = field(default_factory=list)
    _by_text_index: dict[int | None, _ContentEntry] = field(default_factory=dict)
    _by_thinking_index: dict[int | None, _ContentEntry] = field(default_factory=dict)
    _by_tool_id: dict[str, _ContentEntry] = field(default_factory=dict)
    _next_order: int = 0
    parse_errors: list[ToolCallArgsParseError] = field(default_factory=list)

    def add_text(self, index: int | None, delta: str) -> None:
        entry = self._by_text_index.get(index)
        if entry is None:
            entry = self._append("text", index=index)
            self._by_text_index[index] = entry
        entry.text += delta

    def add_thinking(self, index: int | None, delta: str) -> None:
        entry = self._by_thinking_index.get(index)
        if entry is None:
            entry = self._append("thinking", index=index)
            self._by_thinking_index[index] = entry
        entry.text += delta

    def set_thinking_signature(self, index: int | None, signature: str | None) -> None:
        if signature is None:
            return
        entry = self._by_thinking_index.get(index)
        if entry is None:
            entry = self._append("thinking", index=index)
            self._by_thinking_index[index] = entry
        entry.signature = (entry.signature or "") + signature

    def add_tool_call(
        self, id: str, name: str, args_delta: str, *, index: int | None = None
    ) -> None:
        entry = self._by_tool_id.get(id)
        if entry is None:
            entry = self._append("tool_call", index=index)
            entry.tool_call_id = id
            entry.tool_name = name
            self._by_tool_id[id] = entry
        if name and not entry.tool_name:
            entry.tool_name = name
        if entry.index is None and index is not None:
            entry.index = index
        entry.args_json += args_delta

    def assemble(
        self,
        *,
        stop_reason: str | None,
        termination: TerminationHint | None,
        usage: Usage | None,
        timestamp: float,
    ) -> AssistantMessage:
        self.parse_errors.clear()
        content: list[AssistantContent] = []
        for entry in self._ordered_entries():
            if entry.kind == "text":
                if entry.text:
                    content.append(TextContent(type="text", text=entry.text))
            elif entry.kind == "thinking":
                if entry.text or entry.signature is not None:
                    content.append(
                        ThinkingBlock(
                            type="thinking",
                            text=entry.text,
                            signature=entry.signature,
                        )
                    )
            elif entry.kind == "tool_call":
                content.append(
                    ToolCallBlock(
                        type="tool_call",
                        id=entry.tool_call_id,
                        name=entry.tool_name,
                        arguments=self._parse_tool_args(entry),
                    )
                )
        return AssistantMessage(
            role="assistant",
            content=content,
            timestamp=timestamp,
            stop_reason=stop_reason,
            termination=termination,
            usage=usage,
        )

    def _append(self, kind: str, *, index: int | None = None) -> _ContentEntry:
        entry = _ContentEntry(kind=kind, index=index, order=self._next_order)
        self._next_order += 1
        self._entries.append(entry)
        return entry

    def _ordered_entries(self) -> list[_ContentEntry]:
        return sorted(
            self._entries,
            key=lambda entry: (
                entry.index if entry.index is not None else 1_000_000,
                entry.order,
            ),
        )

    def _parse_tool_args(self, entry: _ContentEntry) -> dict[str, Any]:
        raw = entry.args_json
        if not raw:
            return {}
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.parse_errors.append(
                ToolCallArgsParseError(
                    tool_call_id=entry.tool_call_id,
                    raw=raw,
                    error=str(exc),
                )
            )
            return {}
        if not isinstance(args, dict):
            self.parse_errors.append(
                ToolCallArgsParseError(
                    tool_call_id=entry.tool_call_id,
                    raw=raw,
                    error=f"expected JSON object, got {type(args).__name__}",
                )
            )
            return {}
        return args
