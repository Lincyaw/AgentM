"""Local durable catalog and versioned-resource store."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    CatalogActiveSetInput,
    CatalogActiveSetRecord,
    CatalogMeta,
    CatalogQuery,
    ResourceVersion,
)
from agentm.storage.serialization import (
    deserialize_catalog_record,
    deserialize_resource_version,
    json_safe,
    serialize_catalog_record,
    serialize_resource_version,
)


class JsonCatalogStore:
    """Filesystem-backed ``VersionedResourceStore`` and active-set catalog."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._versions_path = self._root / "versions.json"
        self._aliases_path = self._root / "aliases.json"
        self._active_sets_path = self._root / "active_sets.json"
        self._content_root = self._root / "content"
        self._root.mkdir(parents=True, exist_ok=True)
        self._content_root.mkdir(parents=True, exist_ok=True)
        self._versions = self._load_versions()
        self._aliases = self._load_aliases()
        self._records = self._load_records()

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
        if not self._content_path(version).exists():
            path = self._content_path(version)
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(path, bytes(content))
        if key not in {
            (item.resource_id, item.version_id)
            for item in self._versions.get(resource_id, ())
        }:
            self._versions.setdefault(resource_id, []).append(version)
            self._persist_versions()
        return version

    async def resolve(
        self,
        resource_id: str,
        *,
        version_id: str | None = None,
    ) -> ResourceVersion | None:
        versions = self._versions.get(resource_id, ())
        if not versions:
            return None
        if version_id is None:
            return versions[-1]
        for version in versions:
            if version.version_id == version_id:
                return version
        return None

    async def read(self, version: ResourceVersion) -> bytes:
        return self._content_path(version).read_bytes()

    async def alias(self, alias: str, version: ResourceVersion) -> None:
        self._aliases[alias] = version
        self._persist_aliases()

    async def resolve_alias(self, alias: str) -> ResourceVersion | None:
        return self._aliases.get(alias)

    async def list_versions(self, resource_id: str) -> list[ResourceVersion]:
        return list(self._versions.get(resource_id, ()))

    async def record_active_set(
        self,
        active_set: CatalogActiveSetInput,
    ) -> ActiveSetFingerprint:
        captured = tuple(active_set.atoms)
        fingerprint = ActiveSetFingerprint(
            algorithm="sha256",
            digest=_digest_json([_activation_record(atom) for atom in captured]),
            atoms=captured,
            metadata={"atom_count": len(captured)},
        )
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
        self._persist_records()
        return fingerprint

    async def get_active_set(self, session_id: str) -> ActiveSetFingerprint | None:
        record = self._records.get(session_id)
        return record.fingerprint if record is not None else None

    async def query_active_sets(
        self,
        query: CatalogQuery,
    ) -> list[CatalogActiveSetRecord]:
        records = list(self._records.values())
        if query.session_id is not None:
            records = [record for record in records if record.session_id == query.session_id]
        if query.root_session_id is not None:
            records = [
                record for record in records if record.root_session_id == query.root_session_id
            ]
        if query.parent_session_id is not None:
            records = [
                record
                for record in records
                if record.parent_session_id == query.parent_session_id
            ]
        if query.scenario is not None:
            records = [record for record in records if record.scenario == query.scenario]
        if query.provider is not None:
            records = [record for record in records if record.provider == query.provider]
        if query.digest is not None:
            records = [
                record for record in records if record.fingerprint.digest == query.digest
            ]
        if query.atom_name is not None:
            records = [
                record
                for record in records
                if any(atom.name == query.atom_name for atom in record.fingerprint.atoms)
            ]
        if query.module_path is not None:
            records = [
                record
                for record in records
                if any(
                    atom.module_path == query.module_path
                    for atom in record.fingerprint.atoms
                )
            ]
        if query.register is not None:
            records = [
                record
                for record in records
                if any(
                    query.register in atom.registers
                    or query.register in atom.provided_capabilities
                    for atom in record.fingerprint.atoms
                )
            ]
        if query.require is not None:
            records = [
                record
                for record in records
                if any(
                    query.require in atom.requires
                    or query.require in atom.required_capabilities
                    for atom in record.fingerprint.atoms
                )
            ]
        if query.version_id is not None:
            records = [
                record
                for record in records
                if any(
                    atom.version is not None and atom.version.version_id == query.version_id
                    for atom in record.fingerprint.atoms
                )
            ]
        records.sort(
            key=lambda record: (record.created_at, record.session_id),
            reverse=query.sort == "desc",
        )
        if query.limit is not None:
            records = records[: query.limit]
        return records

    def _load_versions(self) -> dict[str, list[ResourceVersion]]:
        rows = _read_json_array(self._versions_path)
        versions: dict[str, list[ResourceVersion]] = {}
        for item in rows:
            version = deserialize_resource_version(item)
            versions.setdefault(version.resource_id, []).append(version)
        return versions

    def _persist_versions(self) -> None:
        rows = [
            serialize_resource_version(version)
            for resource_id in sorted(self._versions)
            for version in self._versions[resource_id]
        ]
        _atomic_write_json_array(self._versions_path, rows)

    def _load_aliases(self) -> dict[str, ResourceVersion]:
        rows = _read_json_array(self._aliases_path)
        aliases: dict[str, ResourceVersion] = {}
        for item in rows:
            alias = item.get("alias")
            version_data = item.get("version")
            if isinstance(alias, str) and isinstance(version_data, Mapping):
                aliases[alias] = deserialize_resource_version(version_data)
        return aliases

    def _persist_aliases(self) -> None:
        _atomic_write_json_array(
            self._aliases_path,
            [
                {
                    "alias": alias,
                    "version": serialize_resource_version(version),
                }
                for alias, version in sorted(self._aliases.items())
            ],
        )

    def _load_records(self) -> dict[str, CatalogActiveSetRecord]:
        rows = _read_json_array(self._active_sets_path)
        return {
            record.session_id: record
            for row in rows
            for record in (deserialize_catalog_record(row),)
        }

    def _persist_records(self) -> None:
        _atomic_write_json_array(
            self._active_sets_path,
            [
                serialize_catalog_record(record)
                for _, record in sorted(self._records.items())
            ],
        )

    def _content_path(self, version: ResourceVersion) -> Path:
        return (
            self._content_root
            / _path_token(version.resource_id)
            / _path_token(version.version_id)
        )


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


def _digest_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _digest_json(value: Any) -> str:
    return _digest_bytes(_stable_json(json_safe(value)).encode("utf-8"))


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _path_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_array(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in value if isinstance(item, Mapping)]


def _atomic_write_json_array(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(path, json.dumps(rows, sort_keys=True, indent=2) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


__all__ = ["JsonCatalogStore"]
