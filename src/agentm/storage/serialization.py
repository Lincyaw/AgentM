# code-health: ignore-file[AM025] -- storage adapters normalize persisted JSON and database rows
"""JSON-safe serialization helpers for optional storage backends."""

# code-health: ignore-file[AM022] -- validates untyped persisted trajectory JSON

from __future__ import annotations

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
from agentm.core.abi.store import DiagnosticLevel, TrajectoryDiagnostic
from agentm.core.abi.trajectory import (
    ContentReplacementState,
    TrajectoryHead,
    TrajectoryHeadStatus,
    TrajectoryNode,
    TrajectoryNodeKind,
    TrajectoryNodeRole,
)
from agentm.core.lib.codec_primitives import (
    expect_only_fields,
    expect_optional_integer,
    expect_optional_string,
    expect_string_tuple,
    field_boolean,
    field_integer,
    field_literal,
    field_number,
    field_string,
)
from agentm.core.lib.json_value import json_restore, json_safe

JsonObject = dict[str, Any]
STORAGE_RECORD_VERSION = 2


def _validate_version(data: Mapping[str, Any], path: str) -> None:
    version = field_integer(data, "schema_version", path=path, minimum=1)
    if version != STORAGE_RECORD_VERSION:
        raise ValueError(f"unsupported {path} schema version: {version}")


def serialize_diagnostic(diagnostic: TrajectoryDiagnostic) -> JsonObject:
    return {
        "schema_version": STORAGE_RECORD_VERSION,
        "id": diagnostic.id,
        "session_id": diagnostic.session_id,
        "timestamp": diagnostic.timestamp,
        "level": diagnostic.level,
        "source": diagnostic.source,
        "phase": diagnostic.phase,
        "message": diagnostic.message,
        "error_type": diagnostic.error_type,
        "error_detail": diagnostic.error_detail,
        "turn_id": diagnostic.turn_id,
        "turn_index": diagnostic.turn_index,
        "checkpoint_id": diagnostic.checkpoint_id,
    }


def deserialize_diagnostic(data: Mapping[str, Any]) -> TrajectoryDiagnostic:
    expect_only_fields(
        data,
        {
            "schema_version",
            "id",
            "session_id",
            "timestamp",
            "level",
            "source",
            "phase",
            "message",
            "error_type",
            "error_detail",
            "turn_id",
            "turn_index",
            "checkpoint_id",
        },
        "trajectory diagnostic",
    )
    _validate_version(data, "trajectory diagnostic")
    level = field_literal(
        data,
        "level",
        path="trajectory diagnostic",
        allowed={"info", "warning", "error"},
    )
    return TrajectoryDiagnostic(
        id=field_string(data, "id", path="trajectory diagnostic"),
        session_id=field_string(
            data,
            "session_id",
            path="trajectory diagnostic",
        ),
        timestamp=field_number(
            data,
            "timestamp",
            path="trajectory diagnostic",
        ),
        level=cast(DiagnosticLevel, level),
        source=field_string(data, "source", path="trajectory diagnostic"),
        phase=field_string(data, "phase", path="trajectory diagnostic"),
        message=field_string(data, "message", path="trajectory diagnostic"),
        error_type=expect_optional_string(
            data.get("error_type"),
            path="trajectory diagnostic.error_type",
        ),
        error_detail=expect_optional_string(
            data.get("error_detail"),
            path="trajectory diagnostic.error_detail",
        ),
        turn_id=expect_optional_string(
            data.get("turn_id"),
            path="trajectory diagnostic.turn_id",
        ),
        turn_index=expect_optional_integer(
            data.get("turn_index"),
            path="trajectory diagnostic.turn_index",
        ),
        checkpoint_id=expect_optional_string(
            data.get("checkpoint_id"),
            path="trajectory diagnostic.checkpoint_id",
        ),
    )


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
        "run_id": node.run_id,
        "run_step": node.run_step,
        "message_index": node.message_index,
        "agent_id": node.agent_id,
        "is_sidechain": node.is_sidechain,
        "tool_call_ids": list(node.tool_call_ids),
        "tool_names": list(node.tool_names),
        "content_ref": node.content_ref,
        "visibility": node.visibility,
        "payload": json_safe(node.payload),
        "timestamp": node.timestamp,
    }
    if node.message is not None:
        data["message"] = serialize_message(node.message)
    return data


def deserialize_node(data: Mapping[str, Any]) -> TrajectoryNode:
    expect_only_fields(
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
            "run_id",
            "run_step",
            "message_index",
            "agent_id",
            "is_sidechain",
            "tool_call_ids",
            "tool_names",
            "cache_key",
            "content_ref",
            "visibility",
            "payload",
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
        id=field_string(data, "id", path="trajectory node"),
        session_id=field_string(data, "session_id", path="trajectory node"),
        seq=field_integer(data, "seq", path="trajectory node", minimum=0),
        kind=cast(
            TrajectoryNodeKind,
            field_literal(
                data,
                "kind",
                path="trajectory node",
                allowed={
                    "message",
                    "compact_boundary",
                    "system_prompt",
                },
            ),
        ),
        root_session_id=expect_optional_string(
            data.get("root_session_id"),
            path="trajectory node.root_session_id",
        ),
        parent_session_id=expect_optional_string(
            data.get("parent_session_id"),
            path="trajectory node.parent_session_id",
        ),
        branch_id=field_string(data, "branch_id", path="trajectory node"),
        head_id=field_string(data, "head_id", path="trajectory node"),
        role=cast(
            TrajectoryNodeRole,
            field_literal(
                data,
                "role",
                path="trajectory node",
                allowed={"user", "assistant", "tool_result", "control"},
            ),
        ),
        parent_id=expect_optional_string(
            data.get("parent_id"),
            path="trajectory node.parent_id",
        ),
        logical_parent_id=expect_optional_string(
            data.get("logical_parent_id"),
            path="trajectory node.logical_parent_id",
        ),
        turn_id=expect_optional_string(
            data.get("turn_id"),
            path="trajectory node.turn_id",
        ),
        turn_index=expect_optional_integer(
            data.get("turn_index"),
            path="trajectory node.turn_index",
        ),
        run_id=expect_optional_string(
            data.get("run_id"),
            path="trajectory node.run_id",
        ),
        run_step=expect_optional_integer(
            data.get("run_step"),
            path="trajectory node.run_step",
        ),
        message_index=expect_optional_integer(
            data.get("message_index"),
            path="trajectory node.message_index",
        ),
        agent_id=expect_optional_string(
            data.get("agent_id"),
            path="trajectory node.agent_id",
        ),
        is_sidechain=field_boolean(
            data,
            "is_sidechain",
            path="trajectory node",
        ),
        tool_call_ids=expect_string_tuple(data.get("tool_call_ids"), "tool_call_ids"),
        tool_names=expect_string_tuple(data.get("tool_names"), "tool_names"),
        content_ref=expect_optional_string(
            data.get("content_ref"),
            path="trajectory node.content_ref",
        ),
        visibility=cast(
            MessageVisibility,
            field_literal(
                data,
                "visibility",
                path="trajectory node",
                allowed={"visible", "hidden", "replay_only"},
            ),
        ),
        message=message,
        payload=dict(payload),
        timestamp=field_number(data, "timestamp", path="trajectory node"),
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
    expect_only_fields(
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
        session_id=field_string(data, "session_id", path="trajectory head"),
        head_id=field_string(data, "head_id", path="trajectory head"),
        branch_id=field_string(data, "branch_id", path="trajectory head"),
        node_id=expect_optional_string(
            data.get("node_id"),
            path="trajectory head.node_id",
        ),
        seq=expect_optional_integer(data.get("seq"), path="trajectory head.seq"),
        root_session_id=expect_optional_string(
            data.get("root_session_id"),
            path="trajectory head.root_session_id",
        ),
        parent_session_id=expect_optional_string(
            data.get("parent_session_id"),
            path="trajectory head.parent_session_id",
        ),
        logical_parent_id=expect_optional_string(
            data.get("logical_parent_id"),
            path="trajectory head.logical_parent_id",
        ),
        agent_id=expect_optional_string(
            data.get("agent_id"),
            path="trajectory head.agent_id",
        ),
        is_sidechain=field_boolean(
            data,
            "is_sidechain",
            path="trajectory head",
        ),
        status=cast(
            TrajectoryHeadStatus,
            field_literal(
                data,
                "status",
                path="trajectory head",
                allowed={"active", "dead", "archived"},
            ),
        ),
        updated_at=field_number(data, "updated_at", path="trajectory head"),
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
    expect_only_fields(
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
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in replacements.items()
    ):
        raise ValueError("content replacements must map strings to strings")
    return ContentReplacementState(
        state_key=field_string(
            data,
            "state_key",
            path="content replacement state",
        ),
        seen_tool_call_ids=expect_string_tuple(
            data.get("seen_tool_call_ids"), "seen_tool_call_ids"
        ),
        replacements=dict(replacements),
        source_session_id=expect_optional_string(
            data.get("source_session_id"),
            path="content replacement state.source_session_id",
        ),
        source_leaf_id=expect_optional_string(
            data.get("source_leaf_id"),
            path="content replacement state.source_leaf_id",
        ),
        leaf_node_id=expect_optional_string(
            data.get("leaf_node_id"),
            path="content replacement state.leaf_node_id",
        ),
        branch_id=field_string(
            data,
            "branch_id",
            path="content replacement state",
        ),
        head_id=field_string(
            data,
            "head_id",
            path="content replacement state",
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
    expect_only_fields(
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
        resource_id=field_string(data, "resource_id", path="resource version"),
        version_id=field_string(data, "version_id", path="resource version"),
        digest=field_string(data, "digest", path="resource version"),
        media_type=expect_optional_string(
            data.get("media_type"),
            path="resource version.media_type",
        ),
        size_bytes=field_integer(
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
            serialize_resource_version(atom.version)
            if atom.version is not None
            else None
        ),
        "priority": atom.priority,
        "requires": list(atom.requires),
        "registers": list(atom.registers),
        "required_capabilities": list(atom.required_capabilities),
        "provided_capabilities": list(atom.provided_capabilities),
        "config_fingerprint": atom.config_fingerprint,
    }


def deserialize_atom_activation(data: Mapping[str, Any]) -> AtomActivation:
    expect_only_fields(
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
        name=field_string(data, "name", path="atom activation"),
        module_path=field_string(data, "module_path", path="atom activation"),
        version=(
            deserialize_resource_version(version_data)
            if isinstance(version_data, Mapping)
            else None
        ),
        priority=field_integer(data, "priority", path="atom activation"),
        requires=expect_string_tuple(data.get("requires"), "requires"),
        registers=expect_string_tuple(data.get("registers"), "registers"),
        required_capabilities=expect_string_tuple(
            data.get("required_capabilities"), "required_capabilities"
        ),
        provided_capabilities=expect_string_tuple(
            data.get("provided_capabilities"), "provided_capabilities"
        ),
        config_fingerprint=expect_optional_string(
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
    expect_only_fields(
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
        algorithm=field_string(
            data,
            "algorithm",
            path="active-set fingerprint",
        ),
        digest=field_string(data, "digest", path="active-set fingerprint"),
        atoms=tuple(deserialize_atom_activation(item) for item in atoms),
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
    expect_only_fields(
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
        session_id=field_string(data, "session_id", path="catalog record"),
        fingerprint=deserialize_active_set_fingerprint(fingerprint_data),
        root_session_id=expect_optional_string(
            data.get("root_session_id"),
            path="catalog record.root_session_id",
        ),
        parent_session_id=expect_optional_string(
            data.get("parent_session_id"),
            path="catalog record.parent_session_id",
        ),
        scenario=expect_optional_string(
            data.get("scenario"),
            path="catalog record.scenario",
        ),
        provider=expect_optional_string(
            data.get("provider"),
            path="catalog record.provider",
        ),
        created_at=field_number(data, "created_at", path="catalog record"),
        metadata=dict(metadata),
    )


__all__ = [
    "JsonObject",
    "deserialize_active_set_fingerprint",
    "deserialize_atom_activation",
    "deserialize_catalog_record",
    "deserialize_content_state",
    "deserialize_diagnostic",
    "deserialize_head",
    "deserialize_node",
    "deserialize_resource_version",
    "json_restore",
    "json_safe",
    "serialize_active_set_fingerprint",
    "serialize_atom_activation",
    "serialize_catalog_record",
    "serialize_content_state",
    "serialize_diagnostic",
    "serialize_head",
    "serialize_node",
    "serialize_resource_version",
]
