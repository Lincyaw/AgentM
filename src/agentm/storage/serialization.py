"""JSON-safe serialization helpers for optional storage backends."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    CatalogActiveSetRecord,
    ResourceVersion,
)
from agentm.core.abi.codec import deserialize_message, serialize_message
from agentm.core.abi.messages import AgentMessage, MessageVisibility
from agentm.core.abi.trajectory import (
    ContentReplacementState,
    PromptCacheState,
    TrajectoryHead,
    TrajectoryHeadStatus,
    TrajectoryNode,
    TrajectoryNodeKind,
    TrajectoryNodeRole,
    TrajectoryProjectionState,
    TrajectoryProjectionStatus,
)


JsonObject = dict[str, Any]


def json_safe(value: Any) -> Any:
    """Return a JSON-safe value without leaking mutable implementation objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def json_restore(value: Any) -> Any:
    """Restore values encoded by :func:`json_safe` where the type is known."""

    if isinstance(value, dict):
        if set(value) == {"__bytes_hex__"} and isinstance(value["__bytes_hex__"], str):
            try:
                return bytes.fromhex(value["__bytes_hex__"])
            except ValueError as exc:
                raise ValueError("invalid encoded bytes value") from exc
        return {str(key): json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_restore(item) for item in value]
    return value


def serialize_node(node: TrajectoryNode) -> JsonObject:
    data: JsonObject = {
        "id": node.id,
        "session_id": node.session_id,
        "seq": node.seq,
        "kind": node.kind,
        "root_session_id": node.root_session_id,
        "parent_session_id": node.parent_session_id,
        "branch_id": node.branch_id,
        "head_id": node.head_id,
        "role": node.role,
        "parent_id": node.parent_id,
        "logical_parent_id": node.logical_parent_id,
        "turn_id": node.turn_id,
        "turn_index": node.turn_index,
        "round_index": node.round_index,
        "message_index": node.message_index,
        "agent_id": node.agent_id,
        "is_sidechain": node.is_sidechain,
        "tool_call_ids": list(node.tool_call_ids),
        "tool_names": list(node.tool_names),
        "cache_key": node.cache_key,
        "content_ref": node.content_ref,
        "visibility": node.visibility,
        "payload": json_safe(node.payload),
        "removed_node_ids": list(node.removed_node_ids),
        "timestamp": node.timestamp,
    }
    if node.message is not None:
        data["message"] = serialize_message(node.message)
    return data


def deserialize_node(data: Mapping[str, Any]) -> TrajectoryNode:
    message_data = data.get("message")
    if message_data is not None and not isinstance(message_data, Mapping):
        raise ValueError("trajectory node message must be an object")
    message: AgentMessage | None = (
        deserialize_message(dict(message_data))
        if isinstance(message_data, Mapping)
        else None
    )
    payload = json_restore(data.get("payload", {}))
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory node payload must be an object")
    return TrajectoryNode(
        id=str(data["id"]),
        session_id=str(data["session_id"]),
        seq=int(data["seq"]),
        kind=cast(TrajectoryNodeKind, data.get("kind", "message")),
        root_session_id=_optional_str(data.get("root_session_id")),
        parent_session_id=_optional_str(data.get("parent_session_id")),
        branch_id=str(data.get("branch_id", "main")),
        head_id=str(data.get("head_id", "main")),
        role=cast(TrajectoryNodeRole, data.get("role", "control")),
        parent_id=_optional_str(data.get("parent_id")),
        logical_parent_id=_optional_str(data.get("logical_parent_id")),
        turn_id=_optional_str(data.get("turn_id")),
        turn_index=_optional_int(data.get("turn_index")),
        round_index=_optional_int(data.get("round_index")),
        message_index=_optional_int(data.get("message_index")),
        agent_id=_optional_str(data.get("agent_id")),
        is_sidechain=bool(data.get("is_sidechain", False)),
        tool_call_ids=_string_tuple(data, "tool_call_ids"),
        tool_names=_string_tuple(data, "tool_names"),
        cache_key=_optional_str(data.get("cache_key")),
        content_ref=_optional_str(data.get("content_ref")),
        visibility=cast(MessageVisibility, data.get("visibility", "visible")),
        message=message,
        payload=dict(payload),
        removed_node_ids=_string_tuple(data, "removed_node_ids"),
        timestamp=float(data.get("timestamp", 0.0)),
    )


def serialize_head(head: TrajectoryHead) -> JsonObject:
    return {
        "session_id": head.session_id,
        "head_id": head.head_id,
        "branch_id": head.branch_id,
        "node_id": head.node_id,
        "seq": head.seq,
        "root_session_id": head.root_session_id,
        "parent_session_id": head.parent_session_id,
        "logical_parent_id": head.logical_parent_id,
        "agent_id": head.agent_id,
        "is_sidechain": head.is_sidechain,
        "status": head.status,
        "updated_at": head.updated_at,
        "metadata": json_safe(head.metadata),
    }


def deserialize_head(data: Mapping[str, Any]) -> TrajectoryHead:
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("trajectory head metadata must be an object")
    return TrajectoryHead(
        session_id=str(data["session_id"]),
        head_id=str(data.get("head_id", "main")),
        branch_id=str(data.get("branch_id", "main")),
        node_id=_optional_str(data.get("node_id")),
        seq=_optional_int(data.get("seq")),
        root_session_id=_optional_str(data.get("root_session_id")),
        parent_session_id=_optional_str(data.get("parent_session_id")),
        logical_parent_id=_optional_str(data.get("logical_parent_id")),
        agent_id=_optional_str(data.get("agent_id")),
        is_sidechain=bool(data.get("is_sidechain", False)),
        status=cast(TrajectoryHeadStatus, data.get("status", "active")),
        updated_at=float(data.get("updated_at", 0.0)),
        metadata=dict(metadata),
    )


def serialize_projection_status(status: TrajectoryProjectionStatus) -> JsonObject:
    return {
        "session_id": status.session_id,
        "state": status.state,
        "high_water_turn_id": status.high_water_turn_id,
        "high_water_turn_index": status.high_water_turn_index,
        "node_count": status.node_count,
        "updated_at": status.updated_at,
        "error": status.error,
        "metadata": json_safe(status.metadata),
    }


def deserialize_projection_status(data: Mapping[str, Any]) -> TrajectoryProjectionStatus:
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("projection status metadata must be an object")
    return TrajectoryProjectionStatus(
        session_id=str(data["session_id"]),
        state=cast(TrajectoryProjectionState, data.get("state", "current")),
        high_water_turn_id=_optional_str(data.get("high_water_turn_id")),
        high_water_turn_index=_optional_int(data.get("high_water_turn_index")),
        node_count=int(data.get("node_count", 0)),
        updated_at=float(data.get("updated_at", 0.0)),
        error=_optional_str(data.get("error")),
        metadata=dict(metadata),
    )


def serialize_content_state(state: ContentReplacementState) -> JsonObject:
    return {
        "state_key": state.state_key,
        "seen_tool_call_ids": list(state.seen_tool_call_ids),
        "replacements": json_safe(state.replacements),
        "source_session_id": state.source_session_id,
        "source_leaf_id": state.source_leaf_id,
        "leaf_node_id": state.leaf_node_id,
        "branch_id": state.branch_id,
        "head_id": state.head_id,
        "metadata": json_safe(state.metadata),
    }


def deserialize_content_state(data: Mapping[str, Any]) -> ContentReplacementState:
    replacements = json_restore(data.get("replacements", {}))
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(replacements, Mapping):
        raise ValueError("content replacement state must contain an object")
    if not isinstance(metadata, Mapping):
        raise ValueError("content replacement metadata must be an object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in replacements.items()):
        raise ValueError("content replacements must map strings to strings")
    return ContentReplacementState(
        state_key=str(data["state_key"]),
        seen_tool_call_ids=_string_tuple(data, "seen_tool_call_ids"),
        replacements=dict(replacements),
        source_session_id=_optional_str(data.get("source_session_id")),
        source_leaf_id=_optional_str(data.get("source_leaf_id")),
        leaf_node_id=_optional_str(data.get("leaf_node_id")),
        branch_id=str(data.get("branch_id", "main")),
        head_id=str(data.get("head_id", "main")),
        metadata=dict(metadata),
    )


def serialize_prompt_cache_state(state: PromptCacheState) -> JsonObject:
    return {
        "cache_key": state.cache_key,
        "leaf_node_id": state.leaf_node_id,
        "content_replacement_state_key": state.content_replacement_state_key,
        "branch_id": state.branch_id,
        "head_id": state.head_id,
        "provider": state.provider,
        "metadata": json_safe(state.metadata),
    }


def deserialize_prompt_cache_state(data: Mapping[str, Any]) -> PromptCacheState:
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("prompt cache metadata must be an object")
    return PromptCacheState(
        cache_key=str(data["cache_key"]),
        leaf_node_id=_optional_str(data.get("leaf_node_id")),
        content_replacement_state_key=_optional_str(
            data.get("content_replacement_state_key")
        ),
        branch_id=str(data.get("branch_id", "main")),
        head_id=str(data.get("head_id", "main")),
        provider=_optional_str(data.get("provider")),
        metadata=dict(metadata),
    )


def serialize_resource_version(version: ResourceVersion) -> JsonObject:
    return {
        "resource_id": version.resource_id,
        "version_id": version.version_id,
        "digest": version.digest,
        "media_type": version.media_type,
        "size_bytes": version.size_bytes,
        "metadata": json_safe(version.metadata),
    }


def deserialize_resource_version(data: Mapping[str, Any]) -> ResourceVersion:
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("resource version metadata must be an object")
    return ResourceVersion(
        resource_id=str(data["resource_id"]),
        version_id=str(data["version_id"]),
        digest=str(data["digest"]),
        media_type=_optional_str(data.get("media_type")),
        size_bytes=int(data.get("size_bytes", 0)),
        metadata=dict(metadata),
    )


def serialize_atom_activation(atom: AtomActivation) -> JsonObject:
    return {
        "name": atom.name,
        "module_path": atom.module_path,
        "version": (
            serialize_resource_version(atom.version) if atom.version is not None else None
        ),
        "priority": atom.priority,
        "requires": list(atom.requires),
        "registers": list(atom.registers),
        "required_capabilities": list(atom.required_capabilities),
        "provided_capabilities": list(atom.provided_capabilities),
        "config_fingerprint": atom.config_fingerprint,
    }


def deserialize_atom_activation(data: Mapping[str, Any]) -> AtomActivation:
    version_data = data.get("version")
    if version_data is not None and not isinstance(version_data, Mapping):
        raise ValueError("atom activation version must be an object")
    return AtomActivation(
        name=str(data["name"]),
        module_path=str(data["module_path"]),
        version=(
            deserialize_resource_version(version_data)
            if isinstance(version_data, Mapping)
            else None
        ),
        priority=int(data.get("priority", 500)),
        requires=_string_tuple(data, "requires"),
        registers=_string_tuple(data, "registers"),
        required_capabilities=_string_tuple(data, "required_capabilities"),
        provided_capabilities=_string_tuple(data, "provided_capabilities"),
        config_fingerprint=_optional_str(data.get("config_fingerprint")),
    )


def serialize_active_set_fingerprint(
    fingerprint: ActiveSetFingerprint,
) -> JsonObject:
    return {
        "algorithm": fingerprint.algorithm,
        "digest": fingerprint.digest,
        "atoms": [serialize_atom_activation(atom) for atom in fingerprint.atoms],
        "metadata": json_safe(fingerprint.metadata),
    }


def deserialize_active_set_fingerprint(
    data: Mapping[str, Any],
) -> ActiveSetFingerprint:
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("active-set metadata must be an object")
    atoms = data.get("atoms", ())
    if not isinstance(atoms, (list, tuple)) or not all(
        isinstance(item, Mapping) for item in atoms
    ):
        raise ValueError("active-set atoms must be a list of objects")
    return ActiveSetFingerprint(
        algorithm=str(data["algorithm"]),
        digest=str(data["digest"]),
        atoms=tuple(
            deserialize_atom_activation(item)
            for item in atoms
        ),
        metadata=dict(metadata),
    )


def serialize_catalog_record(record: CatalogActiveSetRecord) -> JsonObject:
    return {
        "session_id": record.session_id,
        "fingerprint": serialize_active_set_fingerprint(record.fingerprint),
        "root_session_id": record.root_session_id,
        "parent_session_id": record.parent_session_id,
        "scenario": record.scenario,
        "provider": record.provider,
        "created_at": record.created_at,
        "metadata": json_safe(record.metadata),
    }


def deserialize_catalog_record(data: Mapping[str, Any]) -> CatalogActiveSetRecord:
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("catalog record metadata must be an object")
    fingerprint_data = data.get("fingerprint")
    if not isinstance(fingerprint_data, Mapping):
        raise ValueError("catalog record is missing fingerprint")
    return CatalogActiveSetRecord(
        session_id=str(data["session_id"]),
        fingerprint=deserialize_active_set_fingerprint(fingerprint_data),
        root_session_id=_optional_str(data.get("root_session_id")),
        parent_session_id=_optional_str(data.get("parent_session_id")),
        scenario=_optional_str(data.get("scenario")),
        provider=_optional_str(data.get("provider")),
        created_at=float(data.get("created_at", 0.0)),
        metadata=dict(metadata),
    )


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _string_tuple(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key, ())
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(value)


__all__ = [
    "JsonObject",
    "deserialize_active_set_fingerprint",
    "deserialize_atom_activation",
    "deserialize_catalog_record",
    "deserialize_content_state",
    "deserialize_head",
    "deserialize_node",
    "deserialize_projection_status",
    "deserialize_prompt_cache_state",
    "deserialize_resource_version",
    "json_restore",
    "json_safe",
    "serialize_active_set_fingerprint",
    "serialize_atom_activation",
    "serialize_catalog_record",
    "serialize_content_state",
    "serialize_head",
    "serialize_node",
    "serialize_projection_status",
    "serialize_prompt_cache_state",
    "serialize_resource_version",
]
