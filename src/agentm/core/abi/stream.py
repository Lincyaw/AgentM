"""LLM stream boundary.

Implements the LLM stream port in ``docs/refactor-abstract-inventory.md``.
The ``StreamFn`` Protocol is the single point that touches a real LLM API; the
agent loop has zero hard-coded provider knowledge. Uses Python
``AsyncIterator`` semantics and a Python-native event taxonomy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol, TypeAlias, runtime_checkable

from .cancel import CancelSignal
from .messages import (
    AgentMessage,
    AssistantMessage,
    JsonValue,
    freeze_json,
)
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
ThinkingLevel: TypeAlias = Literal["off", "low", "medium", "high"]


# --- Model descriptor -------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Model:
    """Provider-agnostic model descriptor.

    ``metadata`` holds vendor-specific bits, opaque to the kernel.
    """

    id: str
    provider: str
    context_window: int
    max_output_tokens: int
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("model id must be a non-empty string")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("model provider must be a non-empty string")
        for label, value in (
            ("context_window", self.context_window),
            ("max_output_tokens", self.max_output_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"model {label} must be a positive integer")
        if self.max_output_tokens > self.context_window:
            raise ValueError(
                "model max_output_tokens cannot exceed context_window"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("model metadata must be an object")
        frozen = freeze_json(self.metadata)
        if not isinstance(frozen, Mapping):
            raise TypeError("model metadata must be an object")
        object.__setattr__(self, "metadata", frozen)


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
        signal: CancelSignal | None = None,
        thinking: ThinkingLevel = "off",
    ) -> AsyncIterator[AssistantStreamEvent]: ...


__all__ = [
    "AssistantStreamEvent",
    "MessageEnd",
    "Model",
    "StreamFn",
    "TextDelta",
    "ThinkingDelta",
    "ThinkingLevel",
    "ToolCallArgsDelta",
    "ToolCallArgsParseError",
    "ToolCallEnd",
    "ToolCallStart",
]
