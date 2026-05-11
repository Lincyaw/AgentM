"""LLM stream boundary.

Implements §3.1 (LLM Stream port) of
`.claude/designs/pluggable-architecture.md`. The ``StreamFn`` Protocol is the
single point that touches a real LLM API; the agent loop has zero hard-coded
provider knowledge. Uses Python ``AsyncIterator`` semantics and a
Python-native event taxonomy.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from .messages import AgentMessage, AssistantMessage
from .tool import Tool


# --- Stream events ----------------------------------------------------------


@dataclass(slots=True, frozen=True)
class TextDelta:
    """Incremental text token(s) for the assistant's text output."""

    text: str


@dataclass(slots=True, frozen=True)
class ThinkingDelta:
    """Incremental reasoning text. ``signature`` is provider-specific."""

    text: str
    signature: str | None = None


@dataclass(slots=True, frozen=True)
class ToolCallStart:
    """Marks the beginning of a tool-call block in the stream."""

    id: str
    name: str


@dataclass(slots=True, frozen=True)
class ToolCallArgsDelta:
    """Incremental JSON-encoded fragment of a tool call's arguments."""

    id: str
    args_json_delta: str


@dataclass(slots=True, frozen=True)
class ToolCallEnd:
    """Marks the end of a tool-call block."""

    id: str


@dataclass(slots=True, frozen=True)
class ToolCallArgsParseError:
    """Provider could not parse a streamed tool-call argument JSON object."""

    CHANNEL: ClassVar[Literal["tool_call_args_parse_error"]] = (
        "tool_call_args_parse_error"
    )
    tool_call_id: str
    raw: str
    error: str


@dataclass(slots=True, frozen=True)
class MessageEnd:
    """Terminal event carrying the fully assembled assistant message.

    Provided as a convenience so consumers that don't want to assemble from
    deltas can just take this. The agent loop uses it as the canonical
    assistant-message source for the turn.
    """

    message: AssistantMessage


AssistantStreamEvent = (
    TextDelta
    | ThinkingDelta
    | ToolCallStart
    | ToolCallArgsDelta
    | ToolCallArgsParseError
    | ToolCallEnd
    | MessageEnd
)


# --- Model descriptor -------------------------------------------------------


@dataclass(slots=True)
class Model:
    """Provider-agnostic model descriptor.

    ``metadata`` holds vendor-specific bits, opaque to the kernel.
    """

    id: str
    provider: str
    context_window: int
    max_output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Stream protocol --------------------------------------------------------


@runtime_checkable
class StreamFn(Protocol):
    """The single LLM-boundary callable.

    Implementations turn the given messages/tools into a stream of
    ``AssistantStreamEvent``s. They are the *only* place provider-specific
    code lives.
    """

    def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: asyncio.Event | None = None,
        thinking: Literal["off", "low", "medium", "high"] = "off",
    ) -> AsyncIterator[AssistantStreamEvent]: ...


__all__ = [
    "AssistantStreamEvent",
    "MessageEnd",
    "Model",
    "StreamFn",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallArgsDelta",
    "ToolCallArgsParseError",
    "ToolCallEnd",
    "ToolCallStart",
]
