# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Serialization codec for trajectory data.

Every field on Turn must be serializable to a JSON-safe dict so that
TrajectoryStore implementations can write to JSONL, SQLite,
or any other backend.

The main challenge is Trigger: it's an open Protocol, so atoms can
define custom trigger types.  The codec uses a registry of
TriggerCodec instances keyed by ``source`` string.  Built-in triggers
have default codecs.  Unknown triggers on deserialization produce a
``RawTrigger`` that carries the original dict.

Usage::

    registry = CodecRegistry()
    registry.register_trigger_codec("my_source", MyCodec())

    data = registry.serialize_turn(turn)     # -> JSON-safe dict
    turn = registry.deserialize_turn(data)   # -> Turn
"""
# code-health: ignore-file[AM022] -- validates untyped persisted JSON at the wire boundary

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast, runtime_checkable

from agentm.core.abi._codec_v2 import (
    migrate_v2_checkpoint,
    migrate_v2_turn,
)
from agentm.core.abi._resource_codec import (
    deserialize_resource_mutations,
    serialize_resource_mutations,
)
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    MessageMeta,
    MessageReplayPolicy,
    MessageTokenAccounting,
    MessageVisibility,
    OpaqueThinkingBlock,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from agentm.core.abi.termination import (
    BudgetExhausted,
    MaxTurnsExhausted,
    ModelEndTurn,
    PromptRunContinued,
    ProviderRequestFailed,
    ProviderTruncated,
    SignalAborted,
    TerminationCause,
    ToolTerminated,
)
from agentm.core.abi.trajectory import (
    AtomInstall,
    Outcome,
    ToolRecord,
    Turn,
    TurnCheckpoint,
    TurnMeta,
)
from agentm.core.abi.trigger import (
    BackgroundCompletion,
    ContinueTrigger,
    Injection,
    MonitorFire,
    SubagentResult,
    Trigger,
    TriggerMetadata,
    TriggerPriority,
    UserInput,
)
from agentm.core.lib.codec_primitives import (
    expect_array,
    expect_boolean,
    expect_integer,
    expect_literal,
    expect_number,
    expect_object,
    expect_only_fields,
    expect_optional_string,
    expect_string,
)
from agentm.core.lib.json_value import (
    json_restore as _json_restore,
)
from agentm.core.lib.json_value import (
    json_safe as _json_safe,
)

TURN_CODEC_VERSION = 3
TURN_CHECKPOINT_CODEC_VERSION = 3
SESSION_META_CODEC_VERSION = 3
_LEGACY_TRAJECTORY_CODEC_VERSION = 2


@runtime_checkable
class TriggerCodec(Protocol):
    """Serialize/deserialize a custom Trigger type."""

    def serialize(self, trigger: Any) -> dict[str, Any]: ...
    def deserialize(self, data: dict[str, Any]) -> Trigger: ...


@dataclasses.dataclass(frozen=True, slots=True)
class RawTrigger:
    """Deferred trigger record whose atom codec is not registered yet."""

    source: str = "unknown"
    data: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("raw trigger source must be a non-empty string")
        if not isinstance(self.data, Mapping):
            raise TypeError("raw trigger data must be an object")
        safe = _json_safe(self.data)
        if not isinstance(safe, dict):
            raise TypeError("raw trigger data must be an object")
        object.__setattr__(self, "data", MappingProxyType(safe))


def _serialize_injected(injected: tuple[AgentMessage, ...]) -> list[dict[str, Any]]:
    return [serialize_message(message) for message in injected]


def _deserialize_injected(
    data: Any,
    *,
    path: str,
) -> tuple[AgentMessage, ...]:
    raw_injected = expect_array(data, path)
    return tuple(
        deserialize_message(expect_object(message, f"{path}[{index}]"))
        for index, message in enumerate(raw_injected)
    )


def _serialize_trigger_metadata(
    metadata: TriggerMetadata | None,
) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {
        "priority": metadata.priority,
        "target_session_id": metadata.target_session_id,
        "target_agent_id": metadata.target_agent_id,
        "origin": metadata.origin,
        "mode": metadata.mode,
        "is_meta": metadata.is_meta,
        "skip_commands": metadata.skip_commands,
        "meta": _json_safe(metadata.meta),
    }


# --- Message serialization --------------------------------------------------


def _message_meta_is_default(meta: MessageMeta) -> bool:
    return meta == MessageMeta()


def _serialize_message_meta(meta: MessageMeta) -> dict[str, Any]:
    data = {
        "synthetic": meta.synthetic,
        "synthetic_kind": meta.synthetic_kind,
        "origin": meta.origin,
        "visibility": meta.visibility,
        "no_response_requested": meta.no_response_requested,
        "token_accounting": meta.token_accounting,
        "replay": meta.replay,
        "target_session_id": meta.target_session_id,
        "target_agent_id": meta.target_agent_id,
        "mode": meta.mode,
        "tags": _json_safe(meta.tags),
    }
    return {
        k: v
        for k, v in data.items()
        if v not in (None, False, {}, "visible", "normal", "include")
    }


def _deserialize_message_meta(data: dict[str, Any] | None) -> MessageMeta:
    if data is None:
        return MessageMeta()
    data = expect_object(data, "message.meta")
    expect_only_fields(
        data,
        {
            "synthetic",
            "synthetic_kind",
            "origin",
            "visibility",
            "no_response_requested",
            "token_accounting",
            "replay",
            "target_session_id",
            "target_agent_id",
            "mode",
            "tags",
        },
        "message.meta",
    )
    tags = _json_restore(data.get("tags", {}))
    if not isinstance(tags, dict):
        raise ValueError("message.meta.tags must be an object")
    return MessageMeta(
        synthetic=expect_boolean(
            data.get("synthetic", False), "message.meta.synthetic"
        ),
        synthetic_kind=expect_optional_string(
            data.get("synthetic_kind"),
            "message.meta.synthetic_kind",
        ),
        origin=expect_optional_string(data.get("origin"), "message.meta.origin"),
        visibility=cast(
            MessageVisibility,
            expect_literal(
                data.get("visibility", "visible"),
                "message.meta.visibility",
                {"visible", "hidden", "replay_only"},
            ),
        ),
        no_response_requested=expect_boolean(
            data.get("no_response_requested", False),
            "message.meta.no_response_requested",
        ),
        token_accounting=cast(
            MessageTokenAccounting,
            expect_literal(
                data.get("token_accounting", "normal"),
                "message.meta.token_accounting",
                {"normal", "exclude", "metadata_only"},
            ),
        ),
        replay=cast(
            MessageReplayPolicy,
            expect_literal(
                data.get("replay", "include"),
                "message.meta.replay",
                {"include", "skip", "metadata_only"},
            ),
        ),
        target_session_id=expect_optional_string(
            data.get("target_session_id"),
            "message.meta.target_session_id",
        ),
        target_agent_id=expect_optional_string(
            data.get("target_agent_id"),
            "message.meta.target_agent_id",
        ),
        mode=expect_optional_string(data.get("mode"), "message.meta.mode"),
        tags=tags,
    )


def _serialize_content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "data": block.data.hex(),
            "mime_type": block.mime_type,
        }
    if isinstance(block, ThinkingBlock):
        d: dict[str, Any] = {"type": "thinking", "text": block.text}
        if block.signature is not None:
            d["signature"] = block.signature
        return d
    if isinstance(block, OpaqueThinkingBlock):
        return {
            "type": "opaque_thinking",
            "provider": block.provider,
            "payload": _json_safe(block.payload),
        }
    if isinstance(block, ToolCallBlock):
        return {
            "type": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments": _json_safe(block.arguments),
        }
    if isinstance(block, ToolResultBlock):
        d = {
            "type": "tool_result",
            "tool_call_id": block.tool_call_id,
            "content": [_serialize_content_block(c) for c in block.content],
            "is_error": block.is_error,
        }
        if not block.deterministic:
            d["deterministic"] = False
        if block.extras is not None:
            d["extras"] = _json_safe(block.extras)
        return d
    raise TypeError(f"unsupported content block: {type(block).__name__}")


def _deserialize_content_block(data: dict[str, Any]) -> Any:
    data = expect_object(data, "content block")
    t = data.get("type")
    if t == "text":
        expect_only_fields(data, {"type", "text"}, "text content")
        return TextContent(
            type="text",
            text=expect_string(data.get("text"), "text content.text"),
        )
    if t == "image":
        expect_only_fields(data, {"type", "data", "mime_type"}, "image content")
        encoded = expect_string(data.get("data"), "image content.data")
        try:
            image_data = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("image content.data is not valid hex") from exc
        return ImageContent(
            type="image",
            data=image_data,
            mime_type=expect_string(
                data.get("mime_type"),
                "image content.mime_type",
                allow_empty=False,
            ),
        )
    if t == "thinking":
        expect_only_fields(
            data,
            {"type", "text", "signature"},
            "thinking content",
        )
        return ThinkingBlock(
            type="thinking",
            text=expect_string(data.get("text"), "thinking content.text"),
            signature=expect_optional_string(
                data.get("signature"),
                "thinking content.signature",
            ),
        )
    if t == "opaque_thinking":
        expect_only_fields(
            data,
            {"type", "provider", "payload"},
            "opaque thinking content",
        )
        payload = _json_restore(data.get("payload"))
        if not isinstance(payload, dict):
            raise ValueError("opaque thinking content.payload must be an object")
        return OpaqueThinkingBlock(
            type="opaque_thinking",
            provider=expect_string(
                data.get("provider"),
                "opaque thinking content.provider",
                allow_empty=False,
            ),
            payload=payload,
        )
    if t == "tool_call":
        expect_only_fields(
            data,
            {"type", "id", "name", "arguments"},
            "tool_call content",
        )
        arguments = _json_restore(data.get("arguments"))
        if not isinstance(arguments, dict):
            raise ValueError("tool_call content.arguments must be an object")
        return ToolCallBlock(
            type="tool_call",
            id=expect_string(
                data.get("id"),
                "tool_call content.id",
                allow_empty=False,
            ),
            name=expect_string(
                data.get("name"),
                "tool_call content.name",
                allow_empty=False,
            ),
            arguments=arguments,
        )
    if t == "tool_result":
        expect_only_fields(
            data,
            {
                "type",
                "tool_call_id",
                "content",
                "is_error",
                "deterministic",
                "extras",
            },
            "tool_result content",
        )
        content = expect_array(data.get("content", []), "tool_result content.content")
        return ToolResultBlock(
            type="tool_result",
            tool_call_id=expect_string(
                data.get("tool_call_id"),
                "tool_result content.tool_call_id",
                allow_empty=False,
            ),
            content=[
                _deserialize_content_block(
                    expect_object(item, f"tool_result content.content[{index}]")
                )
                for index, item in enumerate(content)
            ],
            is_error=expect_boolean(
                data.get("is_error", False),
                "tool_result content.is_error",
            ),
            deterministic=expect_boolean(
                data.get("deterministic", True),
                "tool_result content.deterministic",
            ),
            extras=_json_restore(data.get("extras")),
        )
    raise ValueError(f"unknown content block type: {t!r}")


def _serialize_atom_install(install: AtomInstall) -> dict[str, Any]:
    data: dict[str, Any] = {
        "atom_name": install.atom_name,
        "source_kind": install.source_kind,
        "location": install.location,
    }
    if install.digest is not None:
        data["digest"] = install.digest
    if install.config:
        data["config"] = _json_safe(install.config)
    # Written only when true, so a trajectory recorded before atoms could be
    # retired reads back identically and the common record stays the short one.
    if install.retired:
        data["retired"] = True
    return data


def _deserialize_atom_install(data: Mapping[str, Any]) -> AtomInstall:
    expect_only_fields(
        data,
        {"atom_name", "source_kind", "location", "digest", "config", "retired"},
        "atom install",
    )
    config = _json_restore(data.get("config", {}))
    if not isinstance(config, dict):
        raise ValueError("atom install config must be an object")
    source_kind = expect_literal(
        data.get("source_kind"), "atom install.source_kind", {"module", "file"}
    )
    return AtomInstall(
        atom_name=expect_string(
            data.get("atom_name"), "atom install.atom_name", allow_empty=False
        ),
        source_kind=cast(Literal["module", "file"], source_kind),
        location=expect_string(
            data.get("location"), "atom install.location", allow_empty=False
        ),
        digest=expect_optional_string(data.get("digest"), "atom install.digest"),
        config=config,
        retired=expect_boolean(data.get("retired", False), "atom install.retired"),
    )


def serialize_message(msg: AgentMessage) -> dict[str, Any]:
    """Convert an AgentMessage to a JSON-safe dict."""
    if isinstance(msg, UserMessage):
        data = {
            "role": "user",
            "content": [_serialize_content_block(b) for b in msg.content],
            "timestamp": msg.timestamp,
        }
        if not _message_meta_is_default(msg.meta):
            data["meta"] = _serialize_message_meta(msg.meta)
        return data
    if isinstance(msg, AssistantMessage):
        d: dict[str, Any] = {
            "role": "assistant",
            "content": [_serialize_content_block(b) for b in msg.content],
            "timestamp": msg.timestamp,
        }
        if msg.stop_reason is not None:
            d["stop_reason"] = msg.stop_reason
        if msg.usage is not None:
            d["usage"] = asdict(msg.usage)
        if msg.termination is not None:
            d["termination"] = {
                "__type__": type(msg.termination).__qualname__,
                **asdict(msg.termination),
            }
        if not _message_meta_is_default(msg.meta):
            d["meta"] = _serialize_message_meta(msg.meta)
        return d
    if isinstance(msg, ToolResultMessage):
        data = {
            "role": "tool_result",
            "content": [_serialize_content_block(b) for b in msg.content],
            "timestamp": msg.timestamp,
        }
        if not _message_meta_is_default(msg.meta):
            data["meta"] = _serialize_message_meta(msg.meta)
        return data
    raise TypeError(f"unsupported message type: {type(msg).__name__}")


def deserialize_message(data: dict[str, Any]) -> AgentMessage:
    """Reconstruct an AgentMessage from a dict."""
    data = expect_object(data, "message")
    role = data.get("role")
    ts = expect_number(data.get("timestamp"), "message.timestamp")
    if role == "user":
        expect_only_fields(
            data, {"role", "content", "timestamp", "meta"}, "user message"
        )
        content = expect_array(data.get("content"), "user message.content")
        return UserMessage(
            role="user",
            content=[
                _deserialize_content_block(
                    expect_object(item, f"user message.content[{index}]")
                )
                for index, item in enumerate(content)
            ],
            timestamp=ts,
            meta=_deserialize_message_meta(data.get("meta")),
        )
    if role == "assistant":
        expect_only_fields(
            data,
            {
                "role",
                "content",
                "timestamp",
                "stop_reason",
                "usage",
                "termination",
                "meta",
            },
            "assistant message",
        )
        content = expect_array(data.get("content"), "assistant message.content")
        usage = None
        if "usage" in data and data["usage"] is not None:
            u = expect_object(data["usage"], "assistant message.usage")
            expect_only_fields(
                u,
                {"input_tokens", "output_tokens", "cache_read", "cache_write"},
                "assistant message.usage",
            )
            usage = Usage(
                input_tokens=expect_integer(
                    u.get("input_tokens"),
                    "assistant message.usage.input_tokens",
                    minimum=0,
                ),
                output_tokens=expect_integer(
                    u.get("output_tokens"),
                    "assistant message.usage.output_tokens",
                    minimum=0,
                ),
                cache_read=expect_integer(
                    u.get("cache_read"),
                    "assistant message.usage.cache_read",
                    minimum=0,
                ),
                cache_write=expect_integer(
                    u.get("cache_write"),
                    "assistant message.usage.cache_write",
                    minimum=0,
                ),
            )
        termination = None
        term_data = data.get("termination")
        if term_data is not None:
            from agentm.core.abi.termination import (
                Aborted,
                EndTurn,
                MaxTokens,
                PauseTurn,
                ProviderError,
                ToolUseExpected,
                VendorSpecific,
            )

            _term_types: dict[str, type] = {
                cls.__qualname__: cls
                for cls in [
                    EndTurn,
                    ToolUseExpected,
                    MaxTokens,
                    PauseTurn,
                    ProviderError,
                    Aborted,
                    VendorSpecific,
                ]
            }
            if not isinstance(term_data, dict):
                raise ValueError("assistant termination must be an object")
            term_type_name = term_data.get("__type__")
            if not isinstance(term_type_name, str):
                raise ValueError("assistant termination is missing __type__")
            term_cls = _term_types.get(term_type_name)
            if term_cls is None:
                raise ValueError(
                    f"unknown assistant termination type: {term_type_name}"
                )
            allowed_fields = {field.name for field in dataclasses.fields(term_cls)} | {
                "__type__"
            }
            expect_only_fields(
                term_data,
                allowed_fields,
                "assistant message.termination",
            )
            try:
                termination = term_cls(
                    **{
                        key: value
                        for key, value in term_data.items()
                        if key != "__type__"
                    }
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid assistant termination: {term_type_name}"
                ) from exc
        return AssistantMessage(
            role="assistant",
            content=[
                _deserialize_content_block(
                    expect_object(item, f"assistant message.content[{index}]")
                )
                for index, item in enumerate(content)
            ],
            timestamp=ts,
            stop_reason=expect_optional_string(
                data.get("stop_reason"),
                "assistant message.stop_reason",
            ),
            usage=usage,
            termination=termination,
            meta=_deserialize_message_meta(data.get("meta")),
        )
    if role == "tool_result":
        expect_only_fields(
            data,
            {"role", "content", "timestamp", "meta"},
            "tool result message",
        )
        content = expect_array(data.get("content"), "tool result message.content")
        return ToolResultMessage(
            role="tool_result",
            content=[
                _deserialize_content_block(
                    expect_object(item, f"tool result message.content[{index}]")
                )
                for index, item in enumerate(content)
            ],
            timestamp=ts,
            meta=_deserialize_message_meta(data.get("meta")),
        )
    raise ValueError(f"unknown message role: {role!r}")


# --- Built-in trigger codecs -----------------------------------------------


class _DataclassTriggerCodec:
    """Generic codec for frozen dataclass triggers."""

    def __init__(self, cls: type, source: str) -> None:
        self._cls = cls
        self._source = source

    def serialize(self, trigger: Any) -> dict[str, Any]:
        d = asdict(trigger)
        d["__source__"] = trigger.source
        return d

    def deserialize(self, data: dict[str, Any]) -> Any:
        data = dict(data)
        encoded_source = data.pop("__source__", None)
        if encoded_source != self._source:
            raise ValueError(f"{self._cls.__name__} source must be {self._source!r}")
        field_source = data.pop("source", self._source)
        if field_source != self._source:
            raise ValueError(
                f"{self._cls.__name__} source field must be {self._source!r}"
            )
        fields = {f.name for f in dataclasses.fields(self._cls)}
        fields.discard("source")
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown {self._cls.__name__} fields: {sorted(unknown)}")
        return self._cls(**data)


class _UserInputCodec:
    def serialize(self, trigger: UserInput) -> dict[str, Any]:
        return {
            "__source__": "user",
            "content": [_serialize_content_block(b) for b in trigger.content],
        }

    def deserialize(self, data: dict[str, Any]) -> UserInput:
        expect_only_fields(data, {"__source__", "content"}, "user trigger")
        if data.get("__source__") != "user":
            raise ValueError("user trigger source must be 'user'")
        raw_content = expect_array(data.get("content"), "user trigger.content")
        content = tuple(
            _deserialize_content_block(
                expect_object(item, f"user trigger.content[{index}]")
            )
            for index, item in enumerate(raw_content)
        )
        return UserInput(content=content)


class _InjectionCodec:
    def serialize(self, trigger: Injection) -> dict[str, Any]:
        return {
            "__source__": "injection",
            "messages": [serialize_message(m) for m in trigger.messages],
        }

    def deserialize(self, data: dict[str, Any]) -> Injection:
        expect_only_fields(data, {"__source__", "messages"}, "injection trigger")
        if data.get("__source__") != "injection":
            raise ValueError("injection trigger source must be 'injection'")
        raw_messages = expect_array(data.get("messages"), "injection trigger.messages")
        messages = tuple(
            deserialize_message(
                expect_object(item, f"injection trigger.messages[{index}]")
            )
            for index, item in enumerate(raw_messages)
        )
        return Injection(messages=messages)


_BUILTIN_CODECS: Final[dict[str, TriggerCodec]] = {
    "user": _UserInputCodec(),
    "background": _DataclassTriggerCodec(BackgroundCompletion, "background"),
    "monitor": _DataclassTriggerCodec(MonitorFire, "monitor"),
    "subagent": _DataclassTriggerCodec(SubagentResult, "subagent"),
    "continue": _DataclassTriggerCodec(ContinueTrigger, "continue"),
    "injection": _InjectionCodec(),
}


# --- CodecRegistry ----------------------------------------------------------


_BUILTIN_CAUSES: Final[dict[str, type[TerminationCause]]] = {
    cls.__qualname__: cls
    for cls in (
        ModelEndTurn,
        PromptRunContinued,
        ToolTerminated,
        MaxTurnsExhausted,
        SignalAborted,
        ProviderRequestFailed,
        ProviderTruncated,
        BudgetExhausted,
    )
}


class CodecRegistry:
    """Central registry for trigger codecs + Turn serialization."""

    def __init__(self) -> None:
        self._trigger_codecs: dict[str, TriggerCodec] = dict(_BUILTIN_CODECS)
        self._cause_types: dict[str, type[TerminationCause]] = dict(_BUILTIN_CAUSES)

    def register_trigger_codec(
        self,
        source: str,
        codec: TriggerCodec,
        *,
        replace: bool = False,
    ) -> None:
        """Register a codec for a custom trigger source.

        ``replace`` lets a newer version of the same atom take over a source it
        already owns. The source stays decodable across the swap, which is what
        a committed turn naming that source depends on.
        """

        if not isinstance(source, str) or not source:
            raise ValueError("trigger codec source must be a non-empty string")
        if source in self._trigger_codecs and not replace:
            raise ValueError(f"trigger codec already registered for {source!r}")
        self._trigger_codecs[source] = codec

    def trigger_codec(self, source: str) -> TriggerCodec | None:
        """The codec registered for ``source``, or ``None`` if there is none."""

        return self._trigger_codecs.get(source)

    def forget_trigger_codec(self, source: str) -> None:
        """Drop a source's codec; safe to repeat.

        Not part of detaching an atom, which deliberately leaves a codec
        registered so committed turns stay decodable. This is for an
        installation that never landed: its registration was never a fact
        anything could name.
        """

        self._trigger_codecs.pop(source, None)

    def register_cause_type(self, cls: type[TerminationCause]) -> None:
        """Register a custom TerminationCause subclass for deserialization."""
        if not issubclass(cls, TerminationCause):
            raise TypeError("cause type must subclass TerminationCause")
        name = cls.__qualname__
        if name in self._cause_types:
            raise ValueError(f"termination cause type already registered: {name}")
        self._cause_types[name] = cls

    def copy(self) -> CodecRegistry:
        """Return an independent registry with the same codec registrations.

        For a caller that wants a *derived* registry -- a child's, with an
        atom's sources left out.  Not for a rollback: a failed installation
        takes its own codec registrations back through their inverses, so
        nothing puts a picture of this registry back over whatever else was
        registered while the install was awaiting.
        """

        copied = CodecRegistry()
        copied._trigger_codecs = dict(self._trigger_codecs)
        copied._cause_types = dict(self._cause_types)
        return copied

    def copy_without_trigger_sources(
        self,
        sources: set[str],
    ) -> CodecRegistry:
        """Copy the registry while leaving selected atom-owned codecs out."""

        copied = self.copy()
        for source in sources:
            if source not in _BUILTIN_CODECS:
                copied._trigger_codecs.pop(source, None)
        return copied

    # --- Trigger ---

    def serialize_trigger(self, trigger: Any) -> dict[str, Any]:
        if isinstance(trigger, RawTrigger):
            result = dict(trigger.data)
            result["__source__"] = trigger.source
            return _json_safe(result)
        if not isinstance(trigger, Trigger):
            raise TypeError("trigger must implement the Trigger protocol")
        source = expect_string(trigger.source, "trigger.source", allow_empty=False)
        codec = self._trigger_codecs.get(source)
        if codec is not None:
            encoded = codec.serialize(trigger)
            if not isinstance(encoded, dict):
                raise TypeError("TriggerCodec.serialize must return an object")
            encoded = _json_safe(encoded)
            if encoded.get("__source__") != source:
                raise ValueError(
                    f"TriggerCodec for {source!r} emitted a different source"
                )
            return encoded
        raise ValueError(f"trigger source {source!r} has no registered TriggerCodec")

    def deserialize_trigger(self, data: dict[str, Any]) -> Any:
        data = expect_object(data, "trigger")
        source = expect_string(
            data.get("__source__"),
            "trigger.__source__",
            allow_empty=False,
        )
        codec = self._trigger_codecs.get(source)
        if codec is not None:
            return codec.deserialize(data)
        return RawTrigger(source=source, data=dict(data))

    # --- ToolRecord ---

    def _serialize_tool_record(self, tr: ToolRecord) -> dict[str, Any]:
        return {
            "call": _serialize_content_block(tr.call),
            "result": _serialize_content_block(tr.result),
            "backgrounded": tr.backgrounded,
        }

    def _deserialize_tool_record(self, data: dict[str, Any]) -> ToolRecord:
        data = expect_object(data, "tool record")
        expect_only_fields(
            data,
            {"call", "result", "backgrounded"},
            "tool record",
        )
        call = _deserialize_content_block(
            expect_object(data.get("call"), "tool record.call")
        )
        result = _deserialize_content_block(
            expect_object(data.get("result"), "tool record.result")
        )
        if not isinstance(call, ToolCallBlock):
            raise ValueError("tool record.call must be a tool_call block")
        if not isinstance(result, ToolResultBlock):
            raise ValueError("tool record.result must be a tool_result block")
        return ToolRecord(
            call=call,
            result=result,
            backgrounded=expect_boolean(
                data.get("backgrounded", False),
                "tool record.backgrounded",
            ),
        )

    # --- Outcome ---

    def _serialize_outcome(self, outcome: Outcome) -> dict[str, Any]:
        if not isinstance(outcome.cause, TerminationCause):
            raise TypeError("Outcome.cause must be a TerminationCause")
        cause_type = type(outcome.cause)
        if cause_type.__qualname__ not in self._cause_types:
            raise ValueError(
                f"unregistered termination cause type: {cause_type.__qualname__}"
            )
        d: dict[str, Any] = {
            "cause": {
                "__type__": cause_type.__qualname__,
                **asdict(outcome.cause),
            }
        }
        if outcome.injected:
            d["injected"] = _serialize_injected(outcome.injected)
        return d

    def _deserialize_outcome(self, data: dict[str, Any]) -> Outcome:
        data = expect_object(data, "outcome")
        expect_only_fields(data, {"cause", "injected"}, "outcome")
        injected = _deserialize_injected(
            data.get("injected", []),
            path="outcome.injected",
        )
        if "cause" not in data:
            raise ValueError("serialized Outcome is missing cause")
        raw_cause = data["cause"]
        if not isinstance(raw_cause, dict):
            raise ValueError("serialized Outcome.cause must be an object")
        else:
            type_name = raw_cause.get("__type__")
            if not isinstance(type_name, str):
                raise ValueError("serialized Outcome.cause is missing __type__")
            cls = self._cause_types.get(type_name)
            if cls is None:
                raise ValueError(f"unknown termination cause type: {type_name}")
            fields = {field.name for field in dataclasses.fields(cls)}
            expect_only_fields(
                raw_cause,
                fields | {"__type__"},
                "outcome.cause",
            )
            cause_fields = {
                key: value for key, value in raw_cause.items() if key != "__type__"
            }
            try:
                cause = cls(**cause_fields)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid serialized termination cause: {type_name}"
                ) from exc
        return Outcome(
            cause=cause,
            injected=injected,
        )

    # --- TurnMeta ---

    @staticmethod
    def _serialize_meta(meta: TurnMeta) -> dict[str, Any]:
        data: dict[str, Any] = {
            "total_input_tokens": meta.total_input_tokens,
            "total_output_tokens": meta.total_output_tokens,
            "cache_read_tokens": meta.cache_read_tokens,
            "cache_write_tokens": meta.cache_write_tokens,
            "duration_ns": meta.duration_ns,
            "model_id": meta.model_id,
            "resource_mutations": serialize_resource_mutations(meta.resource_mutations),
        }
        if meta.atom_installs:
            data["atom_installs"] = [
                _serialize_atom_install(item) for item in meta.atom_installs
            ]
        if meta.model_context_window is not None:
            data["model_context_window"] = meta.model_context_window
        if meta.system_prompt is not None:
            data["system_prompt"] = meta.system_prompt
        if meta.tool_schema_digest is not None:
            data["tool_schema_digest"] = meta.tool_schema_digest
        return data

    @staticmethod
    def _deserialize_meta(data: dict[str, Any]) -> TurnMeta:
        data = expect_object(data, "turn.meta")
        expect_only_fields(
            data,
            {
                "total_input_tokens",
                "total_output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "duration_ns",
                "model_id",
                "model_context_window",
                "resource_mutations",
                "atom_installs",
                "system_prompt",
                "tool_schema_digest",
            },
            "turn.meta",
        )
        atom_installs = tuple(
            _deserialize_atom_install(expect_object(item, "turn.meta.atom_installs[]"))
            for item in expect_array(
                data.get("atom_installs", []),
                "turn.meta.atom_installs",
            )
        )
        raw_tool_digest = data.get("tool_schema_digest")
        tool_schema_digest = (
            raw_tool_digest if isinstance(raw_tool_digest, str) else None
        )
        raw_system_prompt = data.get("system_prompt")
        system_prompt: str | None = None
        if isinstance(raw_system_prompt, str):
            system_prompt = raw_system_prompt
        return TurnMeta(
            atom_installs=atom_installs,
            total_input_tokens=expect_integer(
                data.get("total_input_tokens"),
                "turn.meta.total_input_tokens",
                minimum=0,
            ),
            total_output_tokens=expect_integer(
                data.get("total_output_tokens"),
                "turn.meta.total_output_tokens",
                minimum=0,
            ),
            cache_read_tokens=expect_integer(
                data.get("cache_read_tokens"),
                "turn.meta.cache_read_tokens",
                minimum=0,
            ),
            cache_write_tokens=expect_integer(
                data.get("cache_write_tokens"),
                "turn.meta.cache_write_tokens",
                minimum=0,
            ),
            duration_ns=expect_integer(
                data.get("duration_ns"),
                "turn.meta.duration_ns",
                minimum=0,
            ),
            model_id=expect_optional_string(data.get("model_id"), "turn.meta.model_id"),
            model_context_window=(
                expect_integer(
                    data.get("model_context_window"),
                    "turn.meta.model_context_window",
                    minimum=1,
                )
                if data.get("model_context_window") is not None
                else None
            ),
            resource_mutations=deserialize_resource_mutations(
                data.get("resource_mutations", [])
            ),
            system_prompt=system_prompt,
            tool_schema_digest=tool_schema_digest,
        )

    # --- Turn ---

    def serialize_turn_checkpoint(
        self,
        checkpoint: TurnCheckpoint,
    ) -> dict[str, Any]:
        """Convert an incomplete turn checkpoint to a JSON-safe dict."""
        return {
            "schema_version": TURN_CHECKPOINT_CODEC_VERSION,
            "index": checkpoint.index,
            "id": checkpoint.id,
            "run_id": checkpoint.run_id,
            "run_step": checkpoint.run_step,
            "trigger": self.serialize_trigger(checkpoint.trigger),
            "response": (
                serialize_message(checkpoint.response)
                if checkpoint.response is not None
                else None
            ),
            "tool_results": [
                self._serialize_tool_record(record)
                for record in checkpoint.tool_results
            ],
            "updated_at": checkpoint.updated_at,
            "meta": self._serialize_meta(checkpoint.meta),
            "injected": _serialize_injected(checkpoint.injected),
            "trigger_metadata": _serialize_trigger_metadata(
                checkpoint.trigger_metadata
            ),
        }

    def deserialize_turn_checkpoint(
        self,
        data: dict[str, Any],
    ) -> TurnCheckpoint:
        """Reconstruct an incomplete turn checkpoint from a dict."""
        data = expect_object(data, "turn checkpoint")
        version = expect_integer(
            data.get("schema_version"),
            "turn checkpoint.schema_version",
        )
        if version == _LEGACY_TRAJECTORY_CODEC_VERSION:
            data = migrate_v2_checkpoint(
                data,
                target_version=TURN_CHECKPOINT_CODEC_VERSION,
            )
        elif version != TURN_CHECKPOINT_CODEC_VERSION:
            raise ValueError(f"unsupported turn checkpoint schema version: {version}")
        expect_only_fields(
            data,
            {
                "schema_version",
                "index",
                "id",
                "run_id",
                "run_step",
                "trigger",
                "response",
                "tool_results",
                "updated_at",
                "meta",
                "injected",
                "trigger_metadata",
            },
            "turn checkpoint",
        )
        raw_response = data.get("response")
        response = (
            None
            if raw_response is None
            else deserialize_message(
                expect_object(raw_response, "turn checkpoint.response")
            )
        )
        if response is not None and not isinstance(response, AssistantMessage):
            raise ValueError("turn checkpoint response must be an assistant message")
        raw_results = expect_array(
            data.get("tool_results"),
            "turn checkpoint.tool_results",
        )
        return TurnCheckpoint(
            index=expect_integer(
                data.get("index"),
                "turn checkpoint.index",
                minimum=0,
            ),
            id=expect_string(
                data.get("id"),
                "turn checkpoint.id",
                allow_empty=False,
            ),
            run_id=expect_string(
                data.get("run_id"),
                "turn checkpoint.run_id",
                allow_empty=False,
            ),
            run_step=expect_integer(
                data.get("run_step"),
                "turn checkpoint.run_step",
                minimum=0,
            ),
            trigger=self.deserialize_trigger(
                expect_object(data.get("trigger"), "turn checkpoint.trigger")
            ),
            response=response,
            tool_results=tuple(
                self._deserialize_tool_record(
                    expect_object(item, f"turn checkpoint.tool_results[{index}]")
                )
                for index, item in enumerate(raw_results)
            ),
            updated_at=expect_number(
                data.get("updated_at"),
                "turn checkpoint.updated_at",
            ),
            meta=self._deserialize_meta(
                expect_object(data.get("meta"), "turn checkpoint.meta")
            ),
            injected=_deserialize_injected(
                data.get("injected", []),
                path="turn checkpoint.injected",
            ),
            trigger_metadata=self._deserialize_trigger_metadata(
                data.get("trigger_metadata")
            ),
        )

    def serialize_turn(self, turn: Turn) -> dict[str, Any]:
        """Convert a Turn to a JSON-safe dict for storage."""
        return {
            "schema_version": TURN_CODEC_VERSION,
            "index": turn.index,
            "id": turn.id,
            "run_id": turn.run_id,
            "run_step": turn.run_step,
            "trigger": self.serialize_trigger(turn.trigger),
            "response": (
                serialize_message(turn.response) if turn.response is not None else None
            ),
            "tool_results": [
                self._serialize_tool_record(record) for record in turn.tool_results
            ],
            "outcome": self._serialize_outcome(turn.outcome),
            "timestamp": turn.timestamp,
            "meta": self._serialize_meta(turn.meta),
            "trigger_metadata": _serialize_trigger_metadata(turn.trigger_metadata),
            "request_appended": _serialize_injected(turn.request_appended),
        }

    def deserialize_turn(self, data: dict[str, Any]) -> Turn:
        """Reconstruct a Turn from a dict."""
        data = expect_object(data, "turn")
        version = expect_integer(data.get("schema_version"), "turn.schema_version")
        if version == _LEGACY_TRAJECTORY_CODEC_VERSION:
            data = migrate_v2_turn(data, target_version=TURN_CODEC_VERSION)
        elif version != TURN_CODEC_VERSION:
            raise ValueError(f"unsupported turn schema version: {version}")
        expect_only_fields(
            data,
            {
                "schema_version",
                "index",
                "id",
                "run_id",
                "run_step",
                "trigger",
                "response",
                "tool_results",
                "outcome",
                "timestamp",
                "meta",
                "trigger_metadata",
                "request_appended",
            },
            "turn",
        )
        raw_response = data.get("response")
        response = (
            None
            if raw_response is None
            else deserialize_message(expect_object(raw_response, "turn.response"))
        )
        if response is not None and not isinstance(response, AssistantMessage):
            raise ValueError("turn response must be an assistant message")
        raw_results = expect_array(data.get("tool_results"), "turn.tool_results")
        return Turn(
            index=expect_integer(data.get("index"), "turn.index", minimum=0),
            id=expect_string(data.get("id"), "turn.id", allow_empty=False),
            run_id=expect_string(data.get("run_id"), "turn.run_id", allow_empty=False),
            run_step=expect_integer(data.get("run_step"), "turn.run_step", minimum=0),
            trigger=self.deserialize_trigger(
                expect_object(data.get("trigger"), "turn.trigger")
            ),
            response=response,
            tool_results=tuple(
                self._deserialize_tool_record(
                    expect_object(item, f"turn.tool_results[{index}]")
                )
                for index, item in enumerate(raw_results)
            ),
            outcome=self._deserialize_outcome(
                expect_object(data.get("outcome"), "turn.outcome")
            ),
            timestamp=expect_number(data.get("timestamp"), "turn.timestamp"),
            meta=self._deserialize_meta(expect_object(data.get("meta"), "turn.meta")),
            request_appended=_deserialize_injected(
                data.get("request_appended", []),
                path="turn.request_appended",
            ),
            trigger_metadata=self._deserialize_trigger_metadata(
                data.get("trigger_metadata")
            ),
        )

    @staticmethod
    def _deserialize_trigger_metadata(data: object) -> TriggerMetadata | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("turn trigger_metadata must be an object")
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError("turn trigger_metadata.meta must be an object")
        expect_only_fields(
            data,
            {
                "priority",
                "target_session_id",
                "target_agent_id",
                "origin",
                "mode",
                "is_meta",
                "skip_commands",
                "meta",
            },
            "turn trigger_metadata",
        )
        return TriggerMetadata(
            priority=cast(
                TriggerPriority,
                expect_literal(
                    data.get("priority"),
                    "turn trigger_metadata.priority",
                    {"now", "next", "later"},
                ),
            ),
            target_session_id=expect_optional_string(
                data.get("target_session_id"),
                "turn trigger_metadata.target_session_id",
            ),
            target_agent_id=expect_optional_string(
                data.get("target_agent_id"),
                "turn trigger_metadata.target_agent_id",
            ),
            origin=expect_optional_string(
                data.get("origin"),
                "turn trigger_metadata.origin",
            ),
            mode=expect_string(
                data.get("mode"),
                "turn trigger_metadata.mode",
                allow_empty=False,
            ),
            is_meta=expect_boolean(
                data.get("is_meta"),
                "turn trigger_metadata.is_meta",
            ),
            skip_commands=expect_boolean(
                data.get("skip_commands"),
                "turn trigger_metadata.skip_commands",
            ),
            meta=meta,
        )

    # --- SessionMeta ---

    @staticmethod
    def serialize_session_meta(meta: Any) -> dict[str, Any]:
        return {
            "schema_version": SESSION_META_CODEC_VERSION,
            "id": meta.id,
            "parent_id": meta.parent_id,
            "fork_point": meta.fork_point,
            "purpose": meta.purpose,
            "cwd": meta.cwd,
            "created_at": meta.created_at,
            "config": dict(meta.config),
        }

    @staticmethod
    def deserialize_session_meta(data: dict[str, Any]) -> Any:
        from agentm.core.abi.store import SessionMeta

        data = expect_object(data, "session metadata")
        expect_only_fields(
            data,
            {
                "schema_version",
                "id",
                "parent_id",
                "fork_point",
                "purpose",
                "cwd",
                "created_at",
                "config",
            },
            "session metadata",
        )
        version = expect_integer(
            data.get("schema_version"),
            "session metadata.schema_version",
        )
        if version not in {
            SESSION_META_CODEC_VERSION,
            _LEGACY_TRAJECTORY_CODEC_VERSION,
        }:
            raise ValueError(f"unsupported session metadata schema version: {version}")
        fork_point = data.get("fork_point")
        if fork_point is not None and (
            not isinstance(fork_point, (str, int)) or isinstance(fork_point, bool)
        ):
            raise ValueError("session metadata.fork_point must be a string or integer")
        config = expect_object(data.get("config"), "session metadata.config")
        if not all(
            isinstance(key, str)
            and (value is None or isinstance(value, (str, int, float, bool)))
            and (not isinstance(value, float) or math.isfinite(value))
            for key, value in config.items()
        ):
            raise ValueError("session metadata.config is invalid")
        return SessionMeta(
            id=expect_string(data.get("id"), "session metadata.id", allow_empty=False),
            parent_id=expect_optional_string(
                data.get("parent_id"),
                "session metadata.parent_id",
            ),
            fork_point=fork_point,
            purpose=expect_string(
                data.get("purpose"),
                "session metadata.purpose",
                allow_empty=False,
            ),
            cwd=expect_string(data.get("cwd"), "session metadata.cwd"),
            created_at=expect_number(
                data.get("created_at"),
                "session metadata.created_at",
            ),
            config=dict(config),
        )


@runtime_checkable
class CodecBackedTrajectoryStore(Protocol):
    """Optional store capability exposing its authoritative turn codec."""

    @property
    def codec(self) -> CodecRegistry: ...


# Module-level default registry
DEFAULT_CODEC = CodecRegistry()


__all__ = [
    "DEFAULT_CODEC",
    "CodecBackedTrajectoryStore",
    "CodecRegistry",
    "RawTrigger",
    "TriggerCodec",
    "deserialize_message",
    "serialize_message",
]
