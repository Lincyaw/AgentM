"""Kernel message data model.

Implements the message types that flow through the agent loop, per
`.claude/designs/pluggable-architecture.md` §3 (Five Pluggability Axes) — the
data shape that crosses the LLM Stream boundary (§3.1) and the Tool Execution
boundary (§3.2).

Design constraints:
- Plain ``@dataclass(slots=True, frozen=True)`` — no pydantic, no langchain.
- Bytes-based image content (environment-agnostic; works in notebooks, web,
  RPC without filesystem assumptions).
- Tagged unions via ``Literal`` discriminators so consumers can switch on
  ``content.type`` without ``isinstance`` chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .termination import TerminationHint


@dataclass(slots=True, frozen=True)
class TextContent:
    """Plain-text content block carried in user/assistant/tool-result messages."""

    type: Literal["text"]
    text: str


@dataclass(slots=True, frozen=True)
class ImageContent:
    """Raw image bytes plus a MIME type. Environment-agnostic by design."""

    type: Literal["image"]
    data: bytes
    mime_type: str


@dataclass(slots=True, frozen=True)
class ThinkingBlock:
    """An assistant thinking/reasoning block.

    ``signature`` is provider-specific (e.g. Anthropic's redacted-thinking
    signature); ``None`` for providers that don't sign reasoning.
    """

    type: Literal["thinking"]
    text: str
    signature: str | None = None


@dataclass(slots=True, frozen=True)
class ToolCallBlock:
    """A request from the assistant to invoke a tool.

    ``arguments`` is a parsed JSON object (dict). The kernel does not validate
    against the tool's schema — that's the loop's / tool's responsibility.
    """

    type: Literal["tool_call"]
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ToolResultBlock:
    """The result returned from executing a single tool call."""

    type: Literal["tool_result"]
    tool_call_id: str
    content: list[TextContent | ImageContent]
    is_error: bool = False


# Discriminated union of content blocks that can appear inside an assistant
# message. ``ImageContent`` is intentionally NOT included — the assistant
# emits text, thinking, or tool calls.
AssistantContent = TextContent | ToolCallBlock | ThinkingBlock


@dataclass(slots=True, frozen=True)
class Usage:
    """Token usage for one assistant turn."""

    input_tokens: int
    output_tokens: int
    cache_read: int = 0
    cache_write: int = 0


@dataclass(slots=True, frozen=True)
class UserMessage:
    """A message authored by the user (or a synthetic user-side observation)."""

    role: Literal["user"]
    content: list[TextContent | ImageContent]
    timestamp: float


@dataclass(slots=True, frozen=True)
class AssistantMessage:
    """A message authored by the model.

    ``termination`` is the kernel-canonical reason the model stopped streaming,
    expressed as a :class:`TerminationHint` sum-type that providers fill in.
    The kernel dispatches on this value (not on ``stop_reason``).

    ``stop_reason`` is the raw vendor string carried through verbatim for
    observability and debugging (e.g. ``"end_turn"`` from Anthropic, ``"stop"``
    from OpenAI, or any vendor-specific value). It MUST NOT be inspected by
    kernel code — providers translate it into ``termination``.
    """

    role: Literal["assistant"]
    content: list[AssistantContent]
    timestamp: float
    stop_reason: str | None = None
    termination: TerminationHint | None = None
    usage: Usage | None = None


@dataclass(slots=True, frozen=True)
class ToolResultMessage:
    """A message containing one or more tool-result blocks.

    Conceptually a "user-side" message in most provider protocols, but kept as
    its own role for clarity at the kernel layer.
    """

    role: Literal["tool_result"]
    content: list[ToolResultBlock]
    timestamp: float


# Discriminated union of every message kind the agent loop manipulates.
AgentMessage = UserMessage | AssistantMessage | ToolResultMessage


# --- Helper constructors ----------------------------------------------------


def text_message(text: str, *, timestamp: float = 0.0) -> UserMessage:
    """Build a ``UserMessage`` containing a single text block.

    Convenience for tests and simple invocations. Callers wanting an accurate
    timestamp should pass one explicitly (e.g. ``time.time()``).
    """

    return UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=timestamp,
    )


def tool_result(
    tool_call_id: str,
    text: str,
    *,
    is_error: bool = False,
    timestamp: float = 0.0,
) -> ToolResultMessage:
    """Build a ``ToolResultMessage`` wrapping a single text-only result block."""

    block = ToolResultBlock(
        type="tool_result",
        tool_call_id=tool_call_id,
        content=[TextContent(type="text", text=text)],
        is_error=is_error,
    )
    return ToolResultMessage(role="tool_result", content=[block], timestamp=timestamp)


__all__ = [
    "AgentMessage",
    "AssistantContent",
    "AssistantMessage",
    "ImageContent",
    "TerminationHint",
    "TextContent",
    "ThinkingBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "text_message",
    "tool_result",
]
