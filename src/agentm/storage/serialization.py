"""JSON-safe serialization helpers for optional storage backends."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
import math
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
)


JsonObject = dict[str, Any]
STORAGE_RECORD_VERSION = 1


def json_safe(value: Any) -> Any:
    """Return a JSON-safe value without leaking mutable implementation objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
        if not all(isinstance(key, str) for key in value):
            raise ValueError("encoded JSON object keys must be strings")
        return {key: json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_restore(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("encoded JSON numbers must be finite")
        return value
    raise ValueError(f"encoded value is not JSON-safe: {type(value).__name__}")


def _validate_version(data: Mapping[str, Any], path: str) -> None:
    version = _required_int(data, "schema_version", path=path, minimum=1)
    if version != STORAGE_RECORD_VERSION:
        raise ValueError(f"unsupported {path} schema version: {version}")


def _required_str(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    allow_empty: bool = False,
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{path}.{key} must be a string")
    return value


def _optional_str(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string or null")
    return value


def _required_int(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    minimum: int | None = None,
) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path}.{key} must be >= {minimum}")
    return value


def _optional_int(value: Any, *, path: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer or null")
    return value


def _required_number(data: Mapping[str, Any], key: str, *, path: str) -> float:
    value = data.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path}.{key} must be a finite number")
    return float(value)


def _required_bool(data: Mapping[str, Any], key: str, *, path: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a bool")
    return value


def _enum(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    allowed: set[str],
) -> str:
    value = _required_str(data, key, path=path)
    if value not in allowed:
        raise ValueError(f"{path}.{key} has invalid value {value!r}")
    return value


def _only_fields(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


def serialize_node(node: TrajectoryNode) -> JsonObject:
    data: JsonObject = {
        "schema_version": STORAGE_RECORD_VERSION,
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
    _only_fields(
        data,
        {
            "schema_version",
            "id",
            "session_id",
            "seq",
            "kind",
            "root_session_id",
            "parent_session_id",
            "branch_id",
            "head_id",
            "role",
            "parent_id",
            "logical_parent_id",
            "turn_id",
            "turn_index",
            "round_index",
            "message_index",
            "agent_id",
            "is_sidechain",
            "tool_call_ids",
            "tool_names",
            "cache_key",
            "content_ref",
            "visibility",
            "payload",
            "removed_node_ids",
            "timestamp",
            "message",
        },
        "trajectory node",
    )
    _validate_version(data, "trajectory node")
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
        id=_required_str(data, "id", path="trajectory node"),
        session_id=_required_str(data, "session_id", path="trajectory node"),
        seq=_required_int(data, "seq", path="trajectory node", minimum=0),
        kind=cast(
            TrajectoryNodeKind,
            _enum(
                data,
                "kind",
                path="trajectory node",
                allowed={
                    "message",
                    "compact_boundary",
                    "content_replacement",
                    "snip",
                    "checkpoint",
                },
            ),
        ),
        root_session_id=_optional_str(
            data.get("root_session_id"),
            path="trajectory node.root_session_id",
        ),
        parent_session_id=_optional_str(
            data.get("parent_session_id"),
            path="trajectory node.parent_session_id",
        ),
        branch_id=_required_str(data, "branch_id", path="trajectory node"),
        head_id=_required_str(data, "head_id", path="trajectory node"),
        role=cast(
            TrajectoryNodeRole,
            _enum(
                data,
                "role",
                path="trajectory node",
                allowed={"user", "assistant", "tool_result", "control"},
            ),
        ),
        parent_id=_optional_str(
            data.get("parent_id"),
            path="trajectory node.parent_id",
        ),
        logical_parent_id=_optional_str(
            data.get("logical_parent_id"),
            path="trajectory node.logical_parent_id",
        ),
        turn_id=_optional_str(
            data.get("turn_id"),
            path="trajectory node.turn_id",
        ),
        turn_index=_optional_int(
            data.get("turn_index"),
            path="trajectory node.turn_index",
        ),
        round_index=_optional_int(
            data.get("round_index"),
            path="trajectory node.round_index",
        ),
        message_index=_optional_int(
            data.get("message_index"),
            path="trajectory node.message_index",
        ),
        agent_id=_optional_str(
            data.get("agent_id"),
            path="trajectory node.agent_id",
        ),
        is_sidechain=_required_bool(
            data,
            "is_sidechain",
            path="trajectory node",
        ),
        tool_call_ids=_string_tuple(data, "tool_call_ids"),
        tool_names=_string_tuple(data, "tool_names"),
        cache_key=_optional_str(
            data.get("cache_key"),
            path="trajectory node.cache_key",
        ),
        content_ref=_optional_str(
            data.get("content_ref"),
            path="trajectory node.content_ref",
        ),
        visibility=cast(
            MessageVisibility,
            _enum(
                data,
                "visibility",
                path="trajectory node",
                allowed={"visible", "hidden", "replay_only"},
            ),
        ),
        message=message,
        payload=dict(payload),
        removed_node_ids=_string_tuple(data, "removed_node_ids"),
        timestamp=_required_number(data, "timestamp", path="trajectory node"),
    )


def serialize_head(head: TrajectoryHead) -> JsonObject:
    return {
        "schema_version": STORAGE_RECORD_VERSION,
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
    _only_fields(
        data,
        {
            "schema_version",
            "session_id",
            "head_id",
            "branch_id",
            "node_id",
            "seq",
            "root_session_id",
            "parent_session_id",
            "logical_parent_id",
            "agent_id",
            "is_sidechain",
            "status",
            "updated_at",
            "metadata",
        },
        "trajectory head",
    )
    _validate_version(data, "trajectory head")
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("trajectory head metadata must be an object")
    return TrajectoryHead(
        session_id=_required_str(data, "session_id", path="trajectory head"),
        head_id=_required_str(data, "head_id", path="trajectory head"),
        branch_id=_required_str(data, "branch_id", path="trajectory head"),
        node_id=_optional_str(
            data.get("node_id"),
            path="trajectory head.node_id",
        ),
        seq=_optional_int(data.get("seq"), path="trajectory head.seq"),
        root_session_id=_optional_str(
            data.get("root_session_id"),
            path="trajectory head.root_session_id",
        ),
        parent_session_id=_optional_str(
            data.get("parent_session_id"),
            path="trajectory head.parent_session_id",
        ),
        logical_parent_id=_optional_str(
            data.get("logical_parent_id"),
            path="trajectory head.logical_parent_id",
        ),
        agent_id=_optional_str(
            data.get("agent_id"),
            path="trajectory head.agent_id",
        ),
        is_sidechain=_required_bool(
            data,
            "is_sidechain",
            path="trajectory head",
        ),
        status=cast(
            TrajectoryHeadStatus,
            _enum(
                data,
                "status",
                path="trajectory head",
                allowed={"active", "dead", "archived"},
            ),
        ),
        updated_at=_required_number(data, "updated_at", path="trajectory head"),
        metadata=dict(metadata),
    )


def serialize_content_state(state: ContentReplacementState) -> JsonObject:
    return {
        "schema_version": STORAGE_RECORD_VERSION,
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
    _only_fields(
        data,
        {
            "schema_version",
            "state_key",
            "seen_tool_call_ids",
            "replacements",
            "source_session_id",
            "source_leaf_id",
            "leaf_node_id",
            "branch_id",
            "head_id",
            "metadata",
        },
        "content replacement state",
    )
    _validate_version(data, "content replacement state")
    replacements = json_restore(data.get("replacements", {}))
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(replacements, Mapping):
        raise ValueError("content replacement state must contain an object")
    if not isinstance(metadata, Mapping):
        raise ValueError("content replacement metadata must be an object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in replacements.items()):
        raise ValueError("content replacements must map strings to strings")
    return ContentReplacementState(
        state_key=_required_str(
            data,
            "state_key",
            path="content replacement state",
        ),
        seen_tool_call_ids=_string_tuple(data, "seen_tool_call_ids"),
        replacements=dict(replacements),
        source_session_id=_optional_str(
            data.get("source_session_id"),
            path="content replacement state.source_session_id",
        ),
        source_leaf_id=_optional_str(
            data.get("source_leaf_id"),
            path="content replacement state.source_leaf_id",
        ),
        leaf_node_id=_optional_str(
            data.get("leaf_node_id"),
            path="content replacement state.leaf_node_id",
        ),
        branch_id=_required_str(
            data,
            "branch_id",
            path="content replacement state",
        ),
        head_id=_required_str(
            data,
            "head_id",
            path="content replacement state",
        ),
        metadata=dict(metadata),
    )


def serialize_prompt_cache_state(state: PromptCacheState) -> JsonObject:
    return {
        "schema_version": STORAGE_RECORD_VERSION,
        "cache_key": state.cache_key,
        "leaf_node_id": state.leaf_node_id,
        "content_replacement_state_key": state.content_replacement_state_key,
        "branch_id": state.branch_id,
        "head_id": state.head_id,
        "provider": state.provider,
        "metadata": json_safe(state.metadata),
    }


def deserialize_prompt_cache_state(data: Mapping[str, Any]) -> PromptCacheState:
    _only_fields(
        data,
        {
            "schema_version",
            "cache_key",
            "leaf_node_id",
            "content_replacement_state_key",
            "branch_id",
            "head_id",
            "provider",
            "metadata",
        },
        "prompt cache state",
    )
    _validate_version(data, "prompt cache state")
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("prompt cache metadata must be an object")
    return PromptCacheState(
        cache_key=_required_str(data, "cache_key", path="prompt cache state"),
        leaf_node_id=_optional_str(
            data.get("leaf_node_id"),
            path="prompt cache state.leaf_node_id",
        ),
        content_replacement_state_key=_optional_str(
            data.get("content_replacement_state_key"),
            path="prompt cache state.content_replacement_state_key",
        ),
        branch_id=_required_str(
            data,
            "branch_id",
            path="prompt cache state",
        ),
        head_id=_required_str(
            data,
            "head_id",
            path="prompt cache state",
        ),
        provider=_optional_str(
            data.get("provider"),
            path="prompt cache state.provider",
        ),
        metadata=dict(metadata),
    )


def serialize_resource_version(version: ResourceVersion) -> JsonObject:
    return {
        "schema_version": STORAGE_RECORD_VERSION,
        "resource_id": version.resource_id,
        "version_id": version.version_id,
        "digest": version.digest,
        "media_type": version.media_type,
        "size_bytes": version.size_bytes,
        "metadata": json_safe(version.metadata),
    }


def deserialize_resource_version(data: Mapping[str, Any]) -> ResourceVersion:
    _only_fields(
        data,
        {
            "schema_version",
            "resource_id",
            "version_id",
            "digest",
            "media_type",
            "size_bytes",
            "metadata",
        },
        "resource version",
    )
    _validate_version(data, "resource version")
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("resource version metadata must be an object")
    return ResourceVersion(
        resource_id=_required_str(data, "resource_id", path="resource version"),
        version_id=_required_str(data, "version_id", path="resource version"),
        digest=_required_str(data, "digest", path="resource version"),
        media_type=_optional_str(
            data.get("media_type"),
            path="resource version.media_type",
        ),
        size_bytes=_required_int(
            data,
            "size_bytes",
            path="resource version",
            minimum=0,
        ),
        metadata=dict(metadata),
    )


def serialize_atom_activation(atom: AtomActivation) -> JsonObject:
    return {
        "schema_version": STORAGE_RECORD_VERSION,
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
    _only_fields(
        data,
        {
            "schema_version",
            "name",
            "module_path",
            "version",
            "priority",
            "requires",
            "registers",
            "required_capabilities",
            "provided_capabilities",
            "config_fingerprint",
        },
        "atom activation",
    )
    _validate_version(data, "atom activation")
    version_data = data.get("version")
    if version_data is not None and not isinstance(version_data, Mapping):
        raise ValueError("atom activation version must be an object")
    return AtomActivation(
        name=_required_str(data, "name", path="atom activation"),
        module_path=_required_str(data, "module_path", path="atom activation"),
        version=(
            deserialize_resource_version(version_data)
            if isinstance(version_data, Mapping)
            else None
        ),
        priority=_required_int(data, "priority", path="atom activation"),
        requires=_string_tuple(data, "requires"),
        registers=_string_tuple(data, "registers"),
        required_capabilities=_string_tuple(data, "required_capabilities"),
        provided_capabilities=_string_tuple(data, "provided_capabilities"),
        config_fingerprint=_optional_str(
            data.get("config_fingerprint"),
            path="atom activation.config_fingerprint",
        ),
    )


def serialize_active_set_fingerprint(
    fingerprint: ActiveSetFingerprint,
) -> JsonObject:
    return {
        "schema_version": STORAGE_RECORD_VERSION,
        "algorithm": fingerprint.algorithm,
        "digest": fingerprint.digest,
        "atoms": [serialize_atom_activation(atom) for atom in fingerprint.atoms],
        "metadata": json_safe(fingerprint.metadata),
    }


def deserialize_active_set_fingerprint(
    data: Mapping[str, Any],
) -> ActiveSetFingerprint:
    _only_fields(
        data,
        {"schema_version", "algorithm", "digest", "atoms", "metadata"},
        "active-set fingerprint",
    )
    _validate_version(data, "active-set fingerprint")
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("active-set metadata must be an object")
    atoms = data.get("atoms", ())
    if not isinstance(atoms, (list, tuple)) or not all(
        isinstance(item, Mapping) for item in atoms
    ):
        raise ValueError("active-set atoms must be a list of objects")
    return ActiveSetFingerprint(
        algorithm=_required_str(
            data,
            "algorithm",
            path="active-set fingerprint",
        ),
        digest=_required_str(data, "digest", path="active-set fingerprint"),
        atoms=tuple(
            deserialize_atom_activation(item)
            for item in atoms
        ),
        metadata=dict(metadata),
    )


def serialize_catalog_record(record: CatalogActiveSetRecord) -> JsonObject:
    return {
        "schema_version": STORAGE_RECORD_VERSION,
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
    _only_fields(
        data,
        {
            "schema_version",
            "session_id",
            "fingerprint",
            "root_session_id",
            "parent_session_id",
            "scenario",
            "provider",
            "created_at",
            "metadata",
        },
        "catalog record",
    )
    _validate_version(data, "catalog record")
    metadata = json_restore(data.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ValueError("catalog record metadata must be an object")
    fingerprint_data = data.get("fingerprint")
    if not isinstance(fingerprint_data, Mapping):
        raise ValueError("catalog record is missing fingerprint")
    return CatalogActiveSetRecord(
        session_id=_required_str(data, "session_id", path="catalog record"),
        fingerprint=deserialize_active_set_fingerprint(fingerprint_data),
        root_session_id=_optional_str(
            data.get("root_session_id"),
            path="catalog record.root_session_id",
        ),
        parent_session_id=_optional_str(
            data.get("parent_session_id"),
            path="catalog record.parent_session_id",
        ),
        scenario=_optional_str(
            data.get("scenario"),
            path="catalog record.scenario",
        ),
        provider=_optional_str(
            data.get("provider"),
            path="catalog record.provider",
        ),
        created_at=_required_number(data, "created_at", path="catalog record"),
        metadata=dict(metadata),
    )


def _string_tuple(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
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
    "serialize_prompt_cache_state",
    "serialize_resource_version",
]
