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

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import asdict
import math
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    MessageMeta,
    MessageReplayPolicy,
    MessageTokenAccounting,
    MessageVisibility,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from agentm.core.abi.resource import ResourceMutation, ResourceRef
from agentm.core.abi.termination import (
    BudgetExhausted,
    MaxTurnsExhausted,
    ModelEndTurn,
    ProviderTruncated,
    SignalAborted,
    TerminationCause,
    ToolTerminated,
)
from agentm.core.abi.trajectory import (
    InjectedMessages,
    Outcome,
    Round,
    ToolRecord,
    Turn,
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


TRAJECTORY_CODEC_VERSION = 1


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


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _string(value: Any, path: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{path} must be a string")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a bool")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    return value


def _number(value: Any, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def _literal(value: Any, path: str, allowed: set[str]) -> str:
    text = _string(value, path)
    if text not in allowed:
        raise ValueError(f"{path} has invalid value {text!r}")
    return text


def _only_fields(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


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
    return {k: v for k, v in data.items() if v not in (None, False, {}, "visible", "normal", "include")}


def _deserialize_message_meta(data: dict[str, Any] | None) -> MessageMeta:
    if data is None:
        return MessageMeta()
    data = _object(data, "message.meta")
    _only_fields(
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
        synthetic=_boolean(data.get("synthetic", False), "message.meta.synthetic"),
        synthetic_kind=_optional_string(
            data.get("synthetic_kind"),
            "message.meta.synthetic_kind",
        ),
        origin=_optional_string(data.get("origin"), "message.meta.origin"),
        visibility=cast(
            MessageVisibility,
            _literal(
                data.get("visibility", "visible"),
                "message.meta.visibility",
                {"visible", "hidden", "replay_only"},
            ),
        ),
        no_response_requested=_boolean(
            data.get("no_response_requested", False),
            "message.meta.no_response_requested",
        ),
        token_accounting=cast(
            MessageTokenAccounting,
            _literal(
                data.get("token_accounting", "normal"),
                "message.meta.token_accounting",
                {"normal", "exclude", "metadata_only"},
            ),
        ),
        replay=cast(
            MessageReplayPolicy,
            _literal(
                data.get("replay", "include"),
                "message.meta.replay",
                {"include", "skip", "metadata_only"},
            ),
        ),
        target_session_id=_optional_string(
            data.get("target_session_id"),
            "message.meta.target_session_id",
        ),
        target_agent_id=_optional_string(
            data.get("target_agent_id"),
            "message.meta.target_agent_id",
        ),
        mode=_optional_string(data.get("mode"), "message.meta.mode"),
        tags=tags,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def _json_restore(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__bytes_hex__"} and isinstance(value["__bytes_hex__"], str):
            try:
                return bytes.fromhex(value["__bytes_hex__"])
            except ValueError as exc:
                raise ValueError("invalid encoded bytes value") from exc
        if not all(isinstance(key, str) for key in value):
            raise ValueError("encoded JSON object keys must be strings")
        return {key: _json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_restore(v) for v in value]
    return value


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
    data = _object(data, "content block")
    t = data.get("type")
    if t == "text":
        _only_fields(data, {"type", "text"}, "text content")
        return TextContent(
            type="text",
            text=_string(data.get("text"), "text content.text"),
        )
    if t == "image":
        _only_fields(data, {"type", "data", "mime_type"}, "image content")
        encoded = _string(data.get("data"), "image content.data")
        try:
            image_data = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("image content.data is not valid hex") from exc
        return ImageContent(
            type="image",
            data=image_data,
            mime_type=_string(
                data.get("mime_type"),
                "image content.mime_type",
                allow_empty=False,
            ),
        )
    if t == "thinking":
        _only_fields(
            data,
            {"type", "text", "signature"},
            "thinking content",
        )
        return ThinkingBlock(
            type="thinking",
            text=_string(data.get("text"), "thinking content.text"),
            signature=_optional_string(
                data.get("signature"),
                "thinking content.signature",
            ),
        )
    if t == "tool_call":
        _only_fields(
            data,
            {"type", "id", "name", "arguments"},
            "tool_call content",
        )
        arguments = _json_restore(data.get("arguments"))
        if not isinstance(arguments, dict):
            raise ValueError("tool_call content.arguments must be an object")
        return ToolCallBlock(
            type="tool_call",
            id=_string(
                data.get("id"),
                "tool_call content.id",
                allow_empty=False,
            ),
            name=_string(
                data.get("name"),
                "tool_call content.name",
                allow_empty=False,
            ),
            arguments=arguments,
        )
    if t == "tool_result":
        _only_fields(
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
        content = _array(data.get("content", []), "tool_result content.content")
        return ToolResultBlock(
            type="tool_result",
            tool_call_id=_string(
                data.get("tool_call_id"),
                "tool_result content.tool_call_id",
                allow_empty=False,
            ),
            content=[
                _deserialize_content_block(
                    _object(item, f"tool_result content.content[{index}]")
                )
                for index, item in enumerate(content)
            ],
            is_error=_boolean(
                data.get("is_error", False),
                "tool_result content.is_error",
            ),
            deterministic=_boolean(
                data.get("deterministic", True),
                "tool_result content.deterministic",
            ),
            extras=_json_restore(data.get("extras")),
        )
    raise ValueError(f"unknown content block type: {t!r}")


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
            from dataclasses import fields as dc_fields
            d["termination"] = {
                "__type__": type(msg.termination).__qualname__,
                **{f.name: getattr(msg.termination, f.name) for f in dc_fields(msg.termination) if not f.name.startswith("_")},
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
    data = _object(data, "message")
    role = data.get("role")
    ts = _number(data.get("timestamp"), "message.timestamp")
    if role == "user":
        _only_fields(data, {"role", "content", "timestamp", "meta"}, "user message")
        content = _array(data.get("content"), "user message.content")
        return UserMessage(
            role="user",
            content=[
                _deserialize_content_block(
                    _object(item, f"user message.content[{index}]")
                )
                for index, item in enumerate(content)
            ],
            timestamp=ts,
            meta=_deserialize_message_meta(data.get("meta")),
        )
    if role == "assistant":
        _only_fields(
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
        content = _array(data.get("content"), "assistant message.content")
        usage = None
        if "usage" in data and data["usage"] is not None:
            u = _object(data["usage"], "assistant message.usage")
            _only_fields(
                u,
                {"input_tokens", "output_tokens", "cache_read", "cache_write"},
                "assistant message.usage",
            )
            usage = Usage(
                input_tokens=_integer(
                    u.get("input_tokens"),
                    "assistant message.usage.input_tokens",
                    minimum=0,
                ),
                output_tokens=_integer(
                    u.get("output_tokens"),
                    "assistant message.usage.output_tokens",
                    minimum=0,
                ),
                cache_read=_integer(
                    u.get("cache_read"),
                    "assistant message.usage.cache_read",
                    minimum=0,
                ),
                cache_write=_integer(
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
                for cls in [EndTurn, ToolUseExpected, MaxTokens, PauseTurn, ProviderError, Aborted, VendorSpecific]
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
            allowed_fields = {
                field.name for field in dataclasses.fields(term_cls)
            } | {"__type__"}
            _only_fields(
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
                    _object(item, f"assistant message.content[{index}]")
                )
                for index, item in enumerate(content)
            ],
            timestamp=ts,
            stop_reason=_optional_string(
                data.get("stop_reason"),
                "assistant message.stop_reason",
            ),
            usage=usage,
            termination=termination,
            meta=_deserialize_message_meta(data.get("meta")),
        )
    if role == "tool_result":
        _only_fields(
            data,
            {"role", "content", "timestamp", "meta"},
            "tool result message",
        )
        content = _array(data.get("content"), "tool result message.content")
        return ToolResultMessage(
            role="tool_result",
            content=[
                _deserialize_content_block(
                    _object(item, f"tool result message.content[{index}]")
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
            raise ValueError(
                f"{self._cls.__name__} source must be {self._source!r}"
            )
        field_source = data.pop("source", self._source)
        if field_source != self._source:
            raise ValueError(
                f"{self._cls.__name__} source field must be {self._source!r}"
            )
        fields = {f.name for f in dataclasses.fields(self._cls)}
        fields.discard("source")
        unknown = set(data) - fields
        if unknown:
            raise ValueError(
                f"unknown {self._cls.__name__} fields: {sorted(unknown)}"
            )
        return self._cls(**data)


class _UserInputCodec:
    def serialize(self, trigger: UserInput) -> dict[str, Any]:
        return {
            "__source__": "user",
            "content": [_serialize_content_block(b) for b in trigger.content],
        }

    def deserialize(self, data: dict[str, Any]) -> UserInput:
        _only_fields(data, {"__source__", "content"}, "user trigger")
        if data.get("__source__") != "user":
            raise ValueError("user trigger source must be 'user'")
        raw_content = _array(data.get("content"), "user trigger.content")
        content = tuple(
            _deserialize_content_block(
                _object(item, f"user trigger.content[{index}]")
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
        _only_fields(data, {"__source__", "messages"}, "injection trigger")
        if data.get("__source__") != "injection":
            raise ValueError("injection trigger source must be 'injection'")
        raw_messages = _array(data.get("messages"), "injection trigger.messages")
        messages = tuple(
            deserialize_message(
                _object(item, f"injection trigger.messages[{index}]")
            )
            for index, item in enumerate(raw_messages)
        )
        return Injection(messages=messages)


_BUILTIN_CODECS: dict[str, TriggerCodec] = {
    "user": _UserInputCodec(),
    "background": _DataclassTriggerCodec(BackgroundCompletion, "background"),
    "monitor": _DataclassTriggerCodec(MonitorFire, "monitor"),
    "subagent": _DataclassTriggerCodec(SubagentResult, "subagent"),
    "continue": _DataclassTriggerCodec(ContinueTrigger, "continue"),
    "injection": _InjectionCodec(),
}


# --- CodecRegistry ----------------------------------------------------------


_BUILTIN_CAUSES: dict[str, type[TerminationCause]] = {
    cls.__qualname__: cls
    for cls in (
        ModelEndTurn,
        ToolTerminated,
        MaxTurnsExhausted,
        SignalAborted,
        ProviderTruncated,
        BudgetExhausted,
    )
}


class CodecRegistry:
    """Central registry for trigger codecs + Turn serialization."""

    def __init__(self) -> None:
        self._trigger_codecs: dict[str, TriggerCodec] = dict(_BUILTIN_CODECS)
        self._cause_types: dict[str, type[TerminationCause]] = dict(_BUILTIN_CAUSES)

    def register_trigger_codec(self, source: str, codec: TriggerCodec) -> None:
        """Register a codec for a custom trigger source."""
        if not isinstance(source, str) or not source:
            raise ValueError("trigger codec source must be a non-empty string")
        if source in self._trigger_codecs:
            raise ValueError(f"trigger codec already registered for {source!r}")
        self._trigger_codecs[source] = codec

    def register_cause_type(self, cls: type[TerminationCause]) -> None:
        """Register a custom TerminationCause subclass for deserialization."""
        if not issubclass(cls, TerminationCause):
            raise TypeError("cause type must subclass TerminationCause")
        name = cls.__qualname__
        if name in self._cause_types:
            raise ValueError(f"termination cause type already registered: {name}")
        self._cause_types[name] = cls

    def copy(self) -> "CodecRegistry":
        """Return an independent registry with the same codec registrations."""

        copied = CodecRegistry()
        copied._trigger_codecs = dict(self._trigger_codecs)
        copied._cause_types = dict(self._cause_types)
        return copied

    def copy_without_trigger_sources(
        self,
        sources: set[str],
    ) -> "CodecRegistry":
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
        source = _string(trigger.source, "trigger.source", allow_empty=False)
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
        raise ValueError(
            f"trigger source {source!r} has no registered TriggerCodec"
        )

    def deserialize_trigger(self, data: dict[str, Any]) -> Any:
        data = _object(data, "trigger")
        source = _string(
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
        data = _object(data, "tool record")
        _only_fields(
            data,
            {"call", "result", "backgrounded"},
            "tool record",
        )
        call = _deserialize_content_block(
            _object(data.get("call"), "tool record.call")
        )
        result = _deserialize_content_block(
            _object(data.get("result"), "tool record.result")
        )
        if not isinstance(call, ToolCallBlock):
            raise ValueError("tool record.call must be a tool_call block")
        if not isinstance(result, ToolResultBlock):
            raise ValueError("tool record.result must be a tool_result block")
        return ToolRecord(
            call=call,
            result=result,
            backgrounded=_boolean(
                data.get("backgrounded", False),
                "tool record.backgrounded",
            ),
        )

    # --- Round ---

    def _serialize_round(self, rnd: Round) -> dict[str, Any]:
        return {
            "response": serialize_message(rnd.response),
            "tool_results": [self._serialize_tool_record(tr) for tr in rnd.tool_results],
        }

    def _deserialize_round(self, data: dict[str, Any]) -> Round:
        data = _object(data, "round")
        _only_fields(data, {"response", "tool_results"}, "round")
        response = deserialize_message(_object(data.get("response"), "round.response"))
        if not isinstance(response, AssistantMessage):
            raise ValueError(
                "serialized round response must be an assistant message"
            )
        raw_results = _array(data.get("tool_results"), "round.tool_results")
        tool_results = tuple(
            self._deserialize_tool_record(
                _object(item, f"round.tool_results[{index}]")
            )
            for index, item in enumerate(raw_results)
        )
        return Round(response=response, tool_results=tool_results)

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
            d["injected"] = [
                {
                    "after_round": injection.after_round,
                    "messages": [
                        serialize_message(message) for message in injection.messages
                    ],
                }
                for injection in outcome.injected
            ]
        return d

    def _deserialize_outcome(self, data: dict[str, Any]) -> Outcome:
        data = _object(data, "outcome")
        _only_fields(data, {"cause", "injected"}, "outcome")
        raw_injected = _array(data.get("injected", []), "outcome.injected")
        injected_records: list[InjectedMessages] = []
        for index, raw_entry in enumerate(raw_injected):
            entry = _object(raw_entry, f"outcome.injected[{index}]")
            _only_fields(
                entry,
                {"after_round", "messages"},
                f"outcome.injected[{index}]",
            )
            raw_messages = _array(
                entry.get("messages"),
                f"outcome.injected[{index}].messages",
            )
            injected_records.append(
                InjectedMessages(
                    after_round=_integer(
                        entry.get("after_round"),
                        f"outcome.injected[{index}].after_round",
                        minimum=0,
                    ),
                    messages=tuple(
                        deserialize_message(
                            _object(
                                message,
                                f"outcome.injected[{index}].messages"
                                f"[{message_index}]",
                            )
                        )
                        for message_index, message in enumerate(raw_messages)
                    ),
                )
            )
        injected = tuple(injected_records)
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
            _only_fields(
                raw_cause,
                fields | {"__type__"},
                "outcome.cause",
            )
            cause_fields = {
                key: value
                for key, value in raw_cause.items()
                if key != "__type__"
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
        return {
            "total_input_tokens": meta.total_input_tokens,
            "total_output_tokens": meta.total_output_tokens,
            "cache_read_tokens": meta.cache_read_tokens,
            "cache_write_tokens": meta.cache_write_tokens,
            "duration_ns": meta.duration_ns,
            "model_id": meta.model_id,
            "resource_mutations": [
                {
                    "ref": {
                        "namespace": mutation.ref.namespace,
                        "path": mutation.ref.path,
                    },
                    "op": mutation.op,
                    "transaction_id": mutation.transaction_id,
                    "before_version": mutation.before_version,
                    "after_version": mutation.after_version,
                    "metadata": _json_safe(mutation.metadata),
                }
                for mutation in meta.resource_mutations
            ],
        }

    @staticmethod
    def _deserialize_resource_mutations(
        data: Any,
    ) -> tuple[ResourceMutation, ...]:
        if not isinstance(data, list):
            raise ValueError("resource_mutations must be a list")
        mutations: list[ResourceMutation] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"resource_mutations[{index}] must be an object"
                )
            ref_data = item.get("ref")
            if not isinstance(ref_data, dict):
                raise ValueError(
                    f"resource_mutations[{index}].ref must be an object"
                )
            namespace = ref_data.get("namespace")
            path = ref_data.get("path")
            op = item.get("op")
            if not isinstance(namespace, str) or not isinstance(path, str):
                raise ValueError(
                    f"resource_mutations[{index}].ref is invalid"
                )
            if op not in {"create", "write", "replace", "delete"}:
                raise ValueError(
                    f"resource_mutations[{index}].op is invalid"
                )
            metadata = _json_restore(item.get("metadata", {}))
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"resource_mutations[{index}].metadata must be an object"
                )
            transaction_id = _optional_string(
                item.get("transaction_id"),
                f"resource_mutations[{index}].transaction_id",
            )
            before_version = _optional_string(
                item.get("before_version"),
                f"resource_mutations[{index}].before_version",
            )
            after_version = _optional_string(
                item.get("after_version"),
                f"resource_mutations[{index}].after_version",
            )
            _only_fields(
                item,
                {
                    "ref",
                    "op",
                    "transaction_id",
                    "before_version",
                    "after_version",
                    "metadata",
                },
                f"resource_mutations[{index}]",
            )
            _only_fields(
                ref_data,
                {"namespace", "path"},
                f"resource_mutations[{index}].ref",
            )
            mutations.append(
                ResourceMutation(
                    ref=ResourceRef(namespace=namespace, path=path),
                    op=op,
                    transaction_id=transaction_id,
                    before_version=before_version,
                    after_version=after_version,
                    metadata=metadata,
                )
            )
        return tuple(mutations)

    @staticmethod
    def _deserialize_meta(data: dict[str, Any]) -> TurnMeta:
        data = _object(data, "turn.meta")
        _only_fields(
            data,
            {
                "total_input_tokens",
                "total_output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "duration_ns",
                "model_id",
                "resource_mutations",
            },
            "turn.meta",
        )
        return TurnMeta(
            total_input_tokens=_integer(
                data.get("total_input_tokens"),
                "turn.meta.total_input_tokens",
                minimum=0,
            ),
            total_output_tokens=_integer(
                data.get("total_output_tokens"),
                "turn.meta.total_output_tokens",
                minimum=0,
            ),
            cache_read_tokens=_integer(
                data.get("cache_read_tokens"),
                "turn.meta.cache_read_tokens",
                minimum=0,
            ),
            cache_write_tokens=_integer(
                data.get("cache_write_tokens"),
                "turn.meta.cache_write_tokens",
                minimum=0,
            ),
            duration_ns=_integer(
                data.get("duration_ns"),
                "turn.meta.duration_ns",
                minimum=0,
            ),
            model_id=_optional_string(data.get("model_id"), "turn.meta.model_id"),
            resource_mutations=CodecRegistry._deserialize_resource_mutations(
                data.get("resource_mutations", [])
            ),
        )

    # --- Turn ---

    def serialize_turn(self, turn: Turn) -> dict[str, Any]:
        """Convert a Turn to a JSON-safe dict for storage."""
        return {
            "schema_version": TRAJECTORY_CODEC_VERSION,
            "index": turn.index,
            "id": turn.id,
            "trigger": self.serialize_trigger(turn.trigger),
            "rounds": [self._serialize_round(r) for r in turn.rounds],
            "outcome": self._serialize_outcome(turn.outcome),
            "timestamp": turn.timestamp,
            "meta": self._serialize_meta(turn.meta),
            "trigger_metadata": (
                {
                    "priority": turn.trigger_metadata.priority,
                    "target_session_id": turn.trigger_metadata.target_session_id,
                    "target_agent_id": turn.trigger_metadata.target_agent_id,
                    "origin": turn.trigger_metadata.origin,
                    "mode": turn.trigger_metadata.mode,
                    "is_meta": turn.trigger_metadata.is_meta,
                    "skip_commands": turn.trigger_metadata.skip_commands,
                    "meta": _json_safe(turn.trigger_metadata.meta),
                }
                if turn.trigger_metadata is not None
                else None
            ),
        }

    def deserialize_turn(self, data: dict[str, Any]) -> Turn:
        """Reconstruct a Turn from a dict."""
        data = _object(data, "turn")
        _only_fields(
            data,
            {
                "schema_version",
                "index",
                "id",
                "trigger",
                "rounds",
                "outcome",
                "timestamp",
                "meta",
                "trigger_metadata",
            },
            "turn",
        )
        version = _integer(data.get("schema_version"), "turn.schema_version")
        if version != TRAJECTORY_CODEC_VERSION:
            raise ValueError(f"unsupported turn schema version: {version}")
        raw_rounds = _array(data.get("rounds"), "turn.rounds")
        return Turn(
            index=_integer(data.get("index"), "turn.index", minimum=0),
            id=_string(data.get("id"), "turn.id", allow_empty=False),
            trigger=self.deserialize_trigger(
                _object(data.get("trigger"), "turn.trigger")
            ),
            rounds=tuple(
                self._deserialize_round(
                    _object(item, f"turn.rounds[{index}]")
                )
                for index, item in enumerate(raw_rounds)
            ),
            outcome=self._deserialize_outcome(
                _object(data.get("outcome"), "turn.outcome")
            ),
            timestamp=_number(data.get("timestamp"), "turn.timestamp"),
            meta=self._deserialize_meta(
                _object(data.get("meta"), "turn.meta")
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
        _only_fields(
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
                _literal(
                    data.get("priority"),
                    "turn trigger_metadata.priority",
                    {"now", "next", "later"},
                ),
            ),
            target_session_id=_optional_string(
                data.get("target_session_id"),
                "turn trigger_metadata.target_session_id",
            ),
            target_agent_id=_optional_string(
                data.get("target_agent_id"),
                "turn trigger_metadata.target_agent_id",
            ),
            origin=_optional_string(
                data.get("origin"),
                "turn trigger_metadata.origin",
            ),
            mode=_string(
                data.get("mode"),
                "turn trigger_metadata.mode",
                allow_empty=False,
            ),
            is_meta=_boolean(
                data.get("is_meta"),
                "turn trigger_metadata.is_meta",
            ),
            skip_commands=_boolean(
                data.get("skip_commands"),
                "turn trigger_metadata.skip_commands",
            ),
            meta=meta,
        )

    # --- SessionMeta ---

    @staticmethod
    def serialize_session_meta(meta: Any) -> dict[str, Any]:
        return {
            "schema_version": TRAJECTORY_CODEC_VERSION,
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
        data = _object(data, "session metadata")
        _only_fields(
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
        version = _integer(
            data.get("schema_version"),
            "session metadata.schema_version",
        )
        if version != TRAJECTORY_CODEC_VERSION:
            raise ValueError(
                f"unsupported session metadata schema version: {version}"
            )
        fork_point = data.get("fork_point")
        if fork_point is not None and (
            not isinstance(fork_point, (str, int)) or isinstance(fork_point, bool)
        ):
            raise ValueError("session metadata.fork_point must be a string or integer")
        config = _object(data.get("config"), "session metadata.config")
        if not all(
            isinstance(key, str)
            and (
                value is None
                or isinstance(value, (str, int, float, bool))
            )
            and (not isinstance(value, float) or math.isfinite(value))
            for key, value in config.items()
        ):
            raise ValueError("session metadata.config is invalid")
        return SessionMeta(
            id=_string(data.get("id"), "session metadata.id", allow_empty=False),
            parent_id=_optional_string(
                data.get("parent_id"),
                "session metadata.parent_id",
            ),
            fork_point=fork_point,
            purpose=_string(
                data.get("purpose"),
                "session metadata.purpose",
                allow_empty=False,
            ),
            cwd=_string(data.get("cwd"), "session metadata.cwd"),
            created_at=_number(
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
    "CodecBackedTrajectoryStore",
    "CodecRegistry",
    "DEFAULT_CODEC",
    "RawTrigger",
    "TriggerCodec",
    "deserialize_message",
    "serialize_message",
]
