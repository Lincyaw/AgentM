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
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    MessageMeta,
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
    UserInput,
)


@runtime_checkable
class TriggerCodec(Protocol):
    """Serialize/deserialize a custom Trigger type."""

    def serialize(self, trigger: Any) -> dict[str, Any]: ...
    def deserialize(self, data: dict[str, Any]) -> Trigger: ...


@dataclasses.dataclass(frozen=True, slots=True)
class RawTrigger:
    """Fallback for triggers whose codec is not registered at load time."""

    source: str = "unknown"
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


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
    if not isinstance(data, dict):
        return MessageMeta()
    tags = _json_restore(data.get("tags", {}))
    if not isinstance(tags, dict):
        tags = {}
    return MessageMeta(
        synthetic=bool(data.get("synthetic", False)),
        synthetic_kind=data.get("synthetic_kind"),
        origin=data.get("origin"),
        visibility=data.get("visibility", "visible"),
        no_response_requested=bool(data.get("no_response_requested", False)),
        token_accounting=data.get("token_accounting", "normal"),
        replay=data.get("replay", "include"),
        target_session_id=data.get("target_session_id"),
        target_agent_id=data.get("target_agent_id"),
        mode=data.get("mode"),
        tags=tags,
    )

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
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
        return {k: _json_restore(v) for k, v in value.items()}
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
    t = data.get("type")
    if t == "text":
        return TextContent(type="text", text=data["text"])
    if t == "image":
        return ImageContent(
            type="image",
            data=bytes.fromhex(data["data"]),
            mime_type=data["mime_type"],
        )
    if t == "thinking":
        return ThinkingBlock(
            type="thinking",
            text=data["text"],
            signature=data.get("signature"),
        )
    if t == "tool_call":
        return ToolCallBlock(
            type="tool_call",
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
        )
    if t == "tool_result":
        return ToolResultBlock(
            type="tool_result",
            tool_call_id=data["tool_call_id"],
            content=[_deserialize_content_block(c) for c in data.get("content", [])],
            is_error=data.get("is_error", False),
            deterministic=data.get("deterministic", True),
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
    role = data.get("role")
    ts = data.get("timestamp", 0.0)
    if role == "user":
        return UserMessage(
            role="user",
            content=[_deserialize_content_block(b) for b in data.get("content", [])],
            timestamp=ts,
            meta=_deserialize_message_meta(data.get("meta")),
        )
    if role == "assistant":
        usage = None
        if "usage" in data and data["usage"] is not None:
            u = data["usage"]
            usage = Usage(
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                cache_read=u.get("cache_read", 0),
                cache_write=u.get("cache_write", 0),
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
            content=[_deserialize_content_block(b) for b in data.get("content", [])],
            timestamp=ts,
            stop_reason=data.get("stop_reason"),
            usage=usage,
            termination=termination,
            meta=_deserialize_message_meta(data.get("meta")),
        )
    if role == "tool_result":
        return ToolResultMessage(
            role="tool_result",
            content=[_deserialize_content_block(b) for b in data.get("content", [])],
            timestamp=ts,
            meta=_deserialize_message_meta(data.get("meta")),
        )
    raise ValueError(f"unknown message role: {role!r}")


# --- Built-in trigger codecs -----------------------------------------------

class _DataclassTriggerCodec:
    """Generic codec for frozen dataclass triggers."""

    def __init__(self, cls: type) -> None:
        self._cls = cls

    def serialize(self, trigger: Any) -> dict[str, Any]:
        d = asdict(trigger)
        d["__source__"] = trigger.source
        return d

    def deserialize(self, data: dict[str, Any]) -> Any:
        data = dict(data)
        data.pop("__source__", None)
        fields = {f.name for f in dataclasses.fields(self._cls)}
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
        content = tuple(
            _deserialize_content_block(b) for b in data.get("content", [])
        )
        return UserInput(content=content)


class _InjectionCodec:
    def serialize(self, trigger: Injection) -> dict[str, Any]:
        return {
            "__source__": "injection",
            "messages": [serialize_message(m) for m in trigger.messages],
        }

    def deserialize(self, data: dict[str, Any]) -> Injection:
        messages = tuple(
            deserialize_message(m) for m in data.get("messages", [])
        )
        return Injection(messages=messages)


_BUILTIN_CODECS: dict[str, TriggerCodec] = {
    "user": _UserInputCodec(),
    "background": _DataclassTriggerCodec(BackgroundCompletion),
    "monitor": _DataclassTriggerCodec(MonitorFire),
    "subagent": _DataclassTriggerCodec(SubagentResult),
    "continue": _DataclassTriggerCodec(ContinueTrigger),
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
        self._trigger_codecs[source] = codec

    def register_cause_type(self, cls: type[TerminationCause]) -> None:
        """Register a custom TerminationCause subclass for deserialization."""
        if not issubclass(cls, TerminationCause):
            raise TypeError("cause type must subclass TerminationCause")
        self._cause_types[cls.__qualname__] = cls

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
            result = dict(trigger.data) if isinstance(trigger.data, dict) else {"data": trigger.data}
            result["__source__"] = trigger.source
            return result
        source = getattr(trigger, "source", "unknown")
        codec = self._trigger_codecs.get(source)
        if codec is not None:
            return codec.serialize(trigger)
        raise ValueError(
            f"trigger source {source!r} has no registered TriggerCodec"
        )

    def deserialize_trigger(self, data: dict[str, Any]) -> Any:
        source = data.get("__source__", data.get("source", "unknown"))
        codec = self._trigger_codecs.get(source)
        if codec is not None:
            return codec.deserialize(data)
        return RawTrigger(source=source, data=data)

    # --- ToolRecord ---

    def _serialize_tool_record(self, tr: ToolRecord) -> dict[str, Any]:
        return {
            "call": _serialize_content_block(tr.call),
            "result": _serialize_content_block(tr.result),
            "backgrounded": tr.backgrounded,
        }

    def _deserialize_tool_record(self, data: dict[str, Any]) -> ToolRecord:
        call = _deserialize_content_block(data["call"])
        result = _deserialize_content_block(data["result"])
        return ToolRecord(
            call=call,
            result=result,
            backgrounded=data.get("backgrounded", False),
        )

    # --- Round ---

    def _serialize_round(self, rnd: Round) -> dict[str, Any]:
        return {
            "response": serialize_message(rnd.response),
            "tool_results": [self._serialize_tool_record(tr) for tr in rnd.tool_results],
        }

    def _deserialize_round(self, data: dict[str, Any]) -> Round:
        response = deserialize_message(data["response"])
        if not isinstance(response, AssistantMessage):
            raise ValueError(
                "serialized round response must be an assistant message"
            )
        tool_results = tuple(
            self._deserialize_tool_record(tr)
            for tr in data.get("tool_results", [])
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
        injected = tuple(
            InjectedMessages(
                after_round=entry["after_round"],
                messages=tuple(
                    deserialize_message(message)
                    for message in entry.get("messages", [])
                ),
            )
            for entry in data.get("injected", [])
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
            filtered = {key: value for key, value in raw_cause.items() if key in fields}
            try:
                cause = cls(**filtered)
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
        data = asdict(meta)
        data["resource_mutations"] = [
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
        ]
        return data

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
            mutations.append(
                ResourceMutation(
                    ref=ResourceRef(namespace=namespace, path=path),
                    op=op,
                    transaction_id=(
                        item.get("transaction_id")
                        if isinstance(item.get("transaction_id"), str)
                        else None
                    ),
                    before_version=item.get("before_version"),
                    after_version=item.get("after_version"),
                    metadata=metadata,
                )
            )
        return tuple(mutations)

    @staticmethod
    def _deserialize_meta(data: dict[str, Any]) -> TurnMeta:
        return TurnMeta(
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            duration_ns=data.get("duration_ns", 0),
            model_id=data.get("model_id"),
            resource_mutations=CodecRegistry._deserialize_resource_mutations(
                data.get("resource_mutations", [])
            ),
        )

    # --- Turn ---

    def serialize_turn(self, turn: Turn) -> dict[str, Any]:
        """Convert a Turn to a JSON-safe dict for storage."""
        return {
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
        return Turn(
            index=data["index"],
            id=data["id"],
            trigger=self.deserialize_trigger(data.get("trigger", {})),
            rounds=tuple(
                self._deserialize_round(r) for r in data.get("rounds", [])
            ),
            outcome=self._deserialize_outcome(data.get("outcome", {})),
            timestamp=data.get("timestamp", 0.0),
            meta=self._deserialize_meta(data.get("meta", {})),
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
        return TriggerMetadata(
            priority=data.get("priority", "next"),
            target_session_id=data.get("target_session_id"),
            target_agent_id=data.get("target_agent_id"),
            origin=data.get("origin"),
            mode=data.get("mode", "prompt"),
            is_meta=bool(data.get("is_meta", False)),
            skip_commands=bool(data.get("skip_commands", False)),
            meta=meta,
        )

    # --- SessionMeta ---

    @staticmethod
    def serialize_session_meta(meta: Any) -> dict[str, Any]:
        return asdict(meta)

    @staticmethod
    def deserialize_session_meta(data: dict[str, Any]) -> Any:
        from agentm.core.abi.store import SessionMeta
        return SessionMeta(
            id=data["id"],
            parent_id=data.get("parent_id"),
            fork_point=data.get("fork_point"),
            purpose=data.get("purpose", "root"),
            cwd=data.get("cwd", ""),
            created_at=data.get("created_at", 0.0),
            config=data.get("config", {}),
        )


# Module-level default registry
DEFAULT_CODEC = CodecRegistry()


__all__ = [
    "CodecRegistry",
    "DEFAULT_CODEC",
    "RawTrigger",
    "TriggerCodec",
    "deserialize_message",
    "serialize_message",
]
