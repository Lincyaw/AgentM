"""Default runtime catalog implementations for SDK composition identity."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    CatalogActiveSetRecord,
    CatalogActiveSetInput,
    CatalogMeta,
    CatalogQuery,
    ResourceVersion,
)
from agentm.core.abi.manifest import ExtensionManifest


class InMemoryVersionedResourceStore:
    """Content-addressed in-memory store for immutable SDK resources."""

    __slots__ = ("_aliases", "_content", "_latest", "_versions")

    def __init__(self) -> None:
        self._content: dict[tuple[str, str], bytes] = {}
        self._versions: dict[str, list[ResourceVersion]] = {}
        self._latest: dict[str, ResourceVersion] = {}
        self._aliases: dict[str, ResourceVersion] = {}

    async def put(
        self,
        *,
        resource_id: str,
        content: bytes,
        media_type: str | None = None,
        metadata: CatalogMeta | None = None,
    ) -> ResourceVersion:
        digest = _digest_bytes(content)
        version = ResourceVersion(
            resource_id=resource_id,
            version_id=digest,
            digest=digest,
            media_type=media_type,
            size_bytes=len(content),
            metadata=dict(metadata or {}),
        )
        key = (resource_id, version.version_id)
        if key not in self._content:
            self._content[key] = bytes(content)
            self._versions.setdefault(resource_id, []).append(version)
        self._latest[resource_id] = version
        return version

    async def resolve(
        self,
        resource_id: str,
        *,
        version_id: str | None = None,
    ) -> ResourceVersion | None:
        if version_id is None:
            return self._latest.get(resource_id)
        for version in self._versions.get(resource_id, ()):
            if version.version_id == version_id:
                return version
        return None

    async def read(self, version: ResourceVersion) -> bytes:
        return self._content[(version.resource_id, version.version_id)]

    async def alias(self, alias: str, version: ResourceVersion) -> None:
        self._aliases[alias] = version

    async def resolve_alias(self, alias: str) -> ResourceVersion | None:
        return self._aliases.get(alias)

    async def list_versions(self, resource_id: str) -> list[ResourceVersion]:
        return list(self._versions.get(resource_id, ()))


class InMemoryAtomCatalog:
    """In-memory active-set catalog with deterministic fingerprints."""

    __slots__ = ("_active_sets", "_records")

    def __init__(self) -> None:
        self._active_sets: dict[str, ActiveSetFingerprint] = {}
        self._records: dict[str, CatalogActiveSetRecord] = {}

    async def record_active_set(
        self,
        active_set: CatalogActiveSetInput,
    ) -> ActiveSetFingerprint:
        captured = tuple(active_set.atoms)
        payload = [_activation_record(atom) for atom in captured]
        fingerprint = ActiveSetFingerprint(
            algorithm="sha256",
            digest=_digest_json(payload),
            atoms=captured,
            metadata={"atom_count": len(captured)},
        )
        self._active_sets[active_set.session_id] = fingerprint
        self._records[active_set.session_id] = CatalogActiveSetRecord(
            session_id=active_set.session_id,
            fingerprint=fingerprint,
            root_session_id=active_set.root_session_id,
            parent_session_id=active_set.parent_session_id,
            scenario=active_set.scenario,
            provider=active_set.provider,
            created_at=active_set.created_at,
            metadata=active_set.metadata,
        )
        return fingerprint

    async def get_active_set(self, session_id: str) -> ActiveSetFingerprint | None:
        return self._active_sets.get(session_id)

    async def query_active_sets(
        self,
        query: CatalogQuery,
    ) -> list[CatalogActiveSetRecord]:
        if query.limit is not None and query.limit < 0:
            raise ValueError("catalog query limit cannot be negative")
        records = list(self._records.values())
        if query.session_id is not None:
            records = [r for r in records if r.session_id == query.session_id]
        if query.root_session_id is not None:
            records = [r for r in records if r.root_session_id == query.root_session_id]
        if query.parent_session_id is not None:
            records = [
                r for r in records if r.parent_session_id == query.parent_session_id
            ]
        if query.scenario is not None:
            records = [r for r in records if r.scenario == query.scenario]
        if query.provider is not None:
            records = [r for r in records if r.provider == query.provider]
        if query.digest is not None:
            records = [r for r in records if r.fingerprint.digest == query.digest]
        if query.atom_name is not None:
            records = [
                r
                for r in records
                if any(atom.name == query.atom_name for atom in r.fingerprint.atoms)
            ]
        if query.module_path is not None:
            records = [
                r
                for r in records
                if any(
                    atom.module_path == query.module_path
                    for atom in r.fingerprint.atoms
                )
            ]
        if query.register is not None:
            records = [
                r
                for r in records
                if any(
                    query.register in atom.registers
                    or query.register in atom.provided_capabilities
                    for atom in r.fingerprint.atoms
                )
            ]
        if query.require is not None:
            records = [
                r
                for r in records
                if any(
                    query.require in atom.requires
                    or query.require in atom.required_capabilities
                    for atom in r.fingerprint.atoms
                )
            ]
        if query.version_id is not None:
            records = [
                r
                for r in records
                if any(
                    atom.version is not None
                    and atom.version.version_id == query.version_id
                    for atom in r.fingerprint.atoms
                )
            ]
        records.sort(
            key=lambda record: (record.created_at, record.session_id),
            reverse=query.sort == "desc",
        )
        if query.limit is not None:
            records = records[: query.limit]
        return records


def build_atom_identity_payload(
    *,
    module_path: str,
    manifest: ExtensionManifest | None,
    config: dict[str, Any],
) -> tuple[bytes, CatalogMeta]:
    """Build an auditable atom identity payload and flat metadata."""

    files = _module_source_files(module_path)
    source_record = [
        {
            "path": path,
            "sha256": _digest_bytes(content),
            "content_b64": base64.b64encode(content).decode("ascii"),
        }
        for path, content in files
    ]
    manifest_record = _manifest_record(manifest)
    config_record = normalize_atom_config(manifest, config)
    payload = {
        "module_path": module_path,
        "manifest": manifest_record,
        "config": config_record,
        "source": source_record,
    }
    content = _stable_json(payload).encode("utf-8")
    return content, {
        "module_path": module_path,
        "atom_name": _atom_name(module_path, manifest),
        "source_digest": _digest_json(source_record),
        "manifest_digest": _digest_json(manifest_record),
        "config_digest": _digest_json(config_record),
        "source_file_count": len(source_record),
    }


def _module_source_files(module_path: str) -> list[tuple[str, bytes]]:
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        raise ModuleNotFoundError(module_path)
    if spec.submodule_search_locations:
        files: list[tuple[str, bytes]] = []
        for root_text in spec.submodule_search_locations:
            root = Path(root_text)
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                files.append((str(path.relative_to(root)), path.read_bytes()))
        if not files:
            raise RuntimeError(f"atom package {module_path!r} has no Python source files")
        return files
    if spec.origin is None:
        raise RuntimeError(f"atom module {module_path!r} has no source origin")
    path = Path(spec.origin)
    if path.suffix != ".py":
        raise RuntimeError(
            f"atom module {module_path!r} is not backed by Python source"
        )
    return [(path.name, path.read_bytes())]


def _manifest_record(manifest: ExtensionManifest | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    schema = manifest.config_schema
    schema_name = None if schema is None else f"{schema.__module__}.{schema.__qualname__}"
    return {
        "name": manifest.name,
        "description": manifest.description,
        "registers": list(manifest.registers),
        "requires": list(manifest.requires),
        "priority": manifest.priority,
        "config_schema": schema_name,
        "sensitive_config_fields": list(manifest.sensitive_config_fields),
    }


def _normalized_config(
    manifest: ExtensionManifest | None,
    config: dict[str, Any],
) -> Any:
    schema = None if manifest is None else manifest.config_schema
    if (
        schema is None
        or not isinstance(schema, type)
        or not issubclass(schema, BaseModel)
    ):
        return config
    return schema.model_validate(config).model_dump(mode="json")


def normalize_atom_config(
    manifest: ExtensionManifest | None,
    config: dict[str, Any],
) -> Any:
    """Return the config shape used for atom identity fingerprints."""

    normalized = _json_value(_normalized_config(manifest, config), path="config")
    if manifest is None or not manifest.sensitive_config_fields:
        return normalized
    return _redact_config_fields(normalized, manifest.sensitive_config_fields)


def _activation_record(atom: AtomActivation) -> dict[str, Any]:
    version = atom.version
    return {
        "name": atom.name,
        "module_path": atom.module_path,
        "version": (
            None
            if version is None
            else {
                "resource_id": version.resource_id,
                "version_id": version.version_id,
                "digest": version.digest,
            }
        ),
        "priority": atom.priority,
        "requires": list(atom.requires),
        "registers": list(atom.registers),
        "required_capabilities": list(atom.required_capabilities),
        "provided_capabilities": list(atom.provided_capabilities),
        "config_fingerprint": atom.config_fingerprint,
    }


def _atom_name(module_path: str, manifest: ExtensionManifest | None) -> str:
    return manifest.name if manifest is not None else module_path


def _digest_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _digest_json(value: Any) -> str:
    return _digest_bytes(_stable_json(value).encode("utf-8"))


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} is not JSON-safe: {type(value).__name__}; "
        "atom identity config must be deterministic"
    )


def _redact_config_fields(value: Any, fields: tuple[str, ...]) -> Any:
    if not isinstance(value, dict):
        return value
    redacted = dict(value)
    for field_path in fields:
        parts = tuple(part for part in field_path.split(".") if part)
        if not parts:
            continue
        current: dict[str, Any] = redacted
        for part in parts[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                break
            copied = dict(child)
            current[part] = copied
            current = copied
        else:
            if parts[-1] in current:
                current[parts[-1]] = "<redacted>"
    return redacted


__all__ = [
    "InMemoryAtomCatalog",
    "InMemoryVersionedResourceStore",
    "build_atom_identity_payload",
    "normalize_atom_config",
]
