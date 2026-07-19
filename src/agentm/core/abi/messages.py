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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from .termination import TerminationHint

MessageVisibility = Literal["visible", "hidden", "replay_only"]
MessageTokenAccounting = Literal["normal", "exclude", "metadata_only"]
MessageReplayPolicy = Literal["include", "skip", "metadata_only"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def _require_string(value: object, label: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TypeError(f"{label} must be a string")


def _require_timestamp(value: object, label: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")


def freeze_json(value: object) -> JsonValue:
    """Defensively copy a JSON value into immutable containers."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType(
            {key: freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


@dataclass(slots=True, frozen=True)
class MessageMeta:
    """Control-plane metadata for synthetic and injected messages.

    The provider-facing message roles stay intentionally small. Hosts,
    presenters, replay tools, and context policies use this metadata to
    distinguish ordinary conversation from hidden attachments, no-response
    prompts, permission results, hook output, mode changes, or local command
    output.
    """

    synthetic: bool = False
    synthetic_kind: str | None = None
    origin: str | None = None
    visibility: MessageVisibility = "visible"
    no_response_requested: bool = False
    token_accounting: MessageTokenAccounting = "normal"
    replay: MessageReplayPolicy = "include"
    target_session_id: str | None = None
    target_agent_id: str | None = None
    mode: str | None = None
    tags: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.synthetic, bool):
            raise TypeError("synthetic must be a bool")
        if not isinstance(self.no_response_requested, bool):
            raise TypeError("no_response_requested must be a bool")
        for label, value in (
            ("synthetic_kind", self.synthetic_kind),
            ("origin", self.origin),
            ("target_session_id", self.target_session_id),
            ("target_agent_id", self.target_agent_id),
            ("mode", self.mode),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{label} must be a string or None")
        if self.visibility not in {"visible", "hidden", "replay_only"}:
            raise ValueError(f"invalid message visibility: {self.visibility!r}")
        if self.token_accounting not in {"normal", "exclude", "metadata_only"}:
            raise ValueError(
                f"invalid message token accounting: {self.token_accounting!r}"
            )
        if self.replay not in {"include", "skip", "metadata_only"}:
            raise ValueError(f"invalid message replay policy: {self.replay!r}")
        object.__setattr__(self, "tags", freeze_json(self.tags))


DEFAULT_MESSAGE_META = MessageMeta()


@dataclass(slots=True, frozen=True)
class TextContent:
    """Plain-text content block carried in user/assistant/tool-result messages."""

    type: Literal["text"]
    text: str

    def __post_init__(self) -> None:
        if self.type != "text":
            raise ValueError(f"invalid text content type: {self.type!r}")
        _require_string(self.text, "text content")


@dataclass(slots=True, frozen=True)
class ImageContent:
    """Raw image bytes plus a MIME type. Environment-agnostic by design."""

    type: Literal["image"]
    data: bytes
    mime_type: str

    def __post_init__(self) -> None:
        if self.type != "image":
            raise ValueError(f"invalid image content type: {self.type!r}")
        if not isinstance(self.data, bytes):
            raise TypeError("image content data must be bytes")
        _require_string(self.mime_type, "image MIME type", allow_empty=False)


@dataclass(slots=True, frozen=True)
class ThinkingBlock:
    """An assistant thinking/reasoning block.

    ``signature`` is provider-specific (e.g. Anthropic's redacted-thinking
    signature); ``None`` for providers that don't sign reasoning.
    """

    type: Literal["thinking"]
    text: str
    signature: str | None = None

    def __post_init__(self) -> None:
        if self.type != "thinking":
            raise ValueError(f"invalid thinking block type: {self.type!r}")
        _require_string(self.text, "thinking text")
        if self.signature is not None:
            _require_string(self.signature, "thinking signature")


@dataclass(slots=True, frozen=True)
class ToolCallBlock:
    """A request from the assistant to invoke a tool.

    ``arguments`` is a parsed JSON object (dict). The kernel does not validate
    against the tool's schema — that's the loop's / tool's responsibility.
    """

    type: Literal["tool_call"]
    id: str
    name: str
    arguments: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.type != "tool_call":
            raise ValueError(f"invalid tool call block type: {self.type!r}")
        _require_string(self.id, "tool call id", allow_empty=False)
        _require_string(self.name, "tool call name", allow_empty=False)
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be an object")
        frozen = freeze_json(self.arguments)
        if not isinstance(frozen, Mapping):
            raise TypeError("tool call arguments must be an object")
        object.__setattr__(self, "arguments", frozen)


@dataclass(slots=True, frozen=True)
class ToolResultBlock:
    """The result returned from executing a single tool call."""

    type: Literal["tool_result"]
    tool_call_id: str
    content: Sequence[TextContent | ImageContent]
    is_error: bool = False
    deterministic: bool = True
    extras: JsonValue = None

    def __post_init__(self) -> None:
        if self.type != "tool_result":
            raise ValueError(f"invalid tool result block type: {self.type!r}")
        _require_string(self.tool_call_id, "tool result call id", allow_empty=False)
        if not isinstance(self.is_error, bool):
            raise TypeError("tool result is_error must be a bool")
        if not isinstance(self.deterministic, bool):
            raise TypeError("tool result deterministic must be a bool")
        content = tuple(self.content)
        if not all(isinstance(item, (TextContent, ImageContent)) for item in content):
            raise TypeError("tool result content must contain text or image blocks")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "extras", freeze_json(self.extras))


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

    def __post_init__(self) -> None:
        for label, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("cache_read", self.cache_read),
            ("cache_write", self.cache_write),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")


@dataclass(slots=True, frozen=True)
class UserMessage:
    """A message authored by the user (or a synthetic user-side observation)."""

    role: Literal["user"]
    content: Sequence[TextContent | ImageContent]
    timestamp: float
    meta: MessageMeta = field(default_factory=MessageMeta)

    def __post_init__(self) -> None:
        if self.role != "user":
            raise ValueError(f"invalid user message role: {self.role!r}")
        content = tuple(self.content)
        if not all(isinstance(item, (TextContent, ImageContent)) for item in content):
            raise TypeError("user message content must contain text or image blocks")
        _require_timestamp(self.timestamp, "user message timestamp")
        if not isinstance(self.meta, MessageMeta):
            raise TypeError("user message meta must be MessageMeta")
        object.__setattr__(self, "content", content)


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
    content: Sequence[AssistantContent]
    timestamp: float
    stop_reason: str | None = None
    termination: TerminationHint | None = None
    usage: Usage | None = None
    meta: MessageMeta = field(default_factory=MessageMeta)

    def __post_init__(self) -> None:
        if self.role != "assistant":
            raise ValueError(f"invalid assistant message role: {self.role!r}")
        content = tuple(self.content)
        if not all(
            isinstance(item, (TextContent, ToolCallBlock, ThinkingBlock))
            for item in content
        ):
            raise TypeError(
                "assistant message content must contain text, thinking, or tool calls"
            )
        _require_timestamp(self.timestamp, "assistant message timestamp")
        if self.stop_reason is not None:
            _require_string(self.stop_reason, "assistant stop_reason")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("assistant usage must be Usage or None")
        if not isinstance(self.meta, MessageMeta):
            raise TypeError("assistant message meta must be MessageMeta")
        object.__setattr__(self, "content", content)


@dataclass(slots=True, frozen=True)
class ToolResultMessage:
    """A message containing one or more tool-result blocks.

    Conceptually a "user-side" message in most provider protocols, but kept as
    its own role for clarity at the kernel layer.
    """

    role: Literal["tool_result"]
    content: Sequence[ToolResultBlock]
    timestamp: float
    meta: MessageMeta = field(default_factory=MessageMeta)

    def __post_init__(self) -> None:
        if self.role != "tool_result":
            raise ValueError(f"invalid tool result message role: {self.role!r}")
        content = tuple(self.content)
        if not all(isinstance(item, ToolResultBlock) for item in content):
            raise TypeError("tool result message content must contain tool result blocks")
        _require_timestamp(self.timestamp, "tool result message timestamp")
        if not isinstance(self.meta, MessageMeta):
            raise TypeError("tool result message meta must be MessageMeta")
        object.__setattr__(self, "content", content)


# Discriminated union of every message kind the agent loop manipulates.
AgentMessage = UserMessage | AssistantMessage | ToolResultMessage


@runtime_checkable
class InterruptionMessagePolicy(Protocol):
    """Policy-owned provider messages used to close an interrupted turn."""

    def interruption_message(
        self,
        reason: str,
        *,
        for_tool_use: bool = False,
    ) -> UserMessage | None:
        ...

    def interrupted_tool_result(
        self,
        tool_call_id: str,
        reason: str,
    ) -> ToolResultBlock:
        ...


# --- Helper constructors ----------------------------------------------------


def text_message(
    text: str,
    *,
    timestamp: float = 0.0,
    meta: MessageMeta | None = None,
) -> UserMessage:
    """Build a ``UserMessage`` containing a single text block.

    Convenience for tests and simple invocations. Callers wanting an accurate
    timestamp should pass one explicitly (e.g. ``time.time()``).
    """

    return UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=timestamp,
        meta=meta or MessageMeta(),
    )


def synthetic_user_message(
    text: str,
    *,
    kind: str,
    origin: str | None = "system",
    visibility: MessageVisibility = "hidden",
    no_response_requested: bool = False,
    timestamp: float = 0.0,
    tags: Mapping[str, JsonValue] | None = None,
) -> UserMessage:
    """Build a user-side synthetic message with explicit replay/UI metadata."""

    return text_message(
        text,
        timestamp=timestamp,
        meta=MessageMeta(
            synthetic=True,
            synthetic_kind=kind,
            origin=origin,
            visibility=visibility,
            no_response_requested=no_response_requested,
            tags=tags or {},
        ),
    )


def tool_result(
    tool_call_id: str,
    text: str,
    *,
    is_error: bool = False,
    extras: object = None,
    timestamp: float = 0.0,
) -> ToolResultMessage:
    """Build a ``ToolResultMessage`` wrapping a single text-only result block."""

    block = ToolResultBlock(
        type="tool_result",
        tool_call_id=tool_call_id,
        content=[TextContent(type="text", text=text)],
        is_error=is_error,
        extras=freeze_json(extras),
    )
    return ToolResultMessage(role="tool_result", content=[block], timestamp=timestamp)


__all__ = [
    "AgentMessage",
    "AssistantContent",
    "AssistantMessage",
    "ImageContent",
    "InterruptionMessagePolicy",
    "JsonValue",
    "DEFAULT_MESSAGE_META",
    "TerminationHint",
    "MessageMeta",
    "MessageReplayPolicy",
    "MessageTokenAccounting",
    "MessageVisibility",
    "TextContent",
    "ThinkingBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "synthetic_user_message",
    "text_message",
    "tool_result",
    "freeze_json",
]
