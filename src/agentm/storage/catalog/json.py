# code-health: ignore-file[AM025] -- storage adapters normalize persisted JSON and database rows
"""Local durable catalog and versioned-resource store."""

# code-health: ignore-file[AM022] -- validates untyped persisted catalog JSON

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    CatalogActiveSetInput,
    CatalogActiveSetRecord,
    CatalogMeta,
    CatalogQuery,
    ResourceVersion,
)
from agentm.core.lib.async_cancel import await_known_outcome
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
        self._files = _CatalogFiles(Path(root))
        self._versions: dict[str, list[ResourceVersion]] = {}
        self._aliases: dict[str, ResourceVersion] = {}
        self._records: dict[str, CatalogActiveSetRecord] = {}
        with self._files.guard():
            self._reload_unlocked()

    async def put(
        self,
        *,
        resource_id: str,
        content: bytes,
        media_type: str | None = None,
        metadata: CatalogMeta | None = None,
    ) -> ResourceVersion:
        return await await_known_outcome(
            asyncio.to_thread(
                self._put,
                resource_id,
                content,
                media_type,
                metadata,
            )
        )

    def _put(
        self,
        resource_id: str,
        content: bytes,
        media_type: str | None,
        metadata: CatalogMeta | None,
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
        with self._files.guard():
            self._reload_unlocked()
            existing = next(
                (
                    item
                    for item in self._versions.get(resource_id, ())
                    if item.version_id == version.version_id
                ),
                None,
            )
            if existing is not None and existing != version:
                raise ValueError(
                    "content-addressed resource version metadata changed for "
                    f"{resource_id}:{version.version_id}"
                )
            path = self._files.content_path(version)
            if path.exists():
                stored = path.read_bytes()
                if stored != content:
                    raise ValueError(f"catalog content digest collision at {path}")
            else:
                _atomic_write_bytes(path, bytes(content))
            if existing is None:
                self._versions.setdefault(resource_id, []).append(version)
                self._files.write_versions(self._versions)
            return existing or version

    async def resolve(
        self,
        resource_id: str,
        *,
        version_id: str | None = None,
    ) -> ResourceVersion | None:
        return await asyncio.to_thread(self._resolve, resource_id, version_id)

    def _resolve(
        self,
        resource_id: str,
        version_id: str | None,
    ) -> ResourceVersion | None:
        with self._files.guard():
            self._reload_unlocked()
            versions = self._versions.get(resource_id, ())
            if not versions:
                return None
            if version_id is None:
                return versions[-1]
            return next(
                (version for version in versions if version.version_id == version_id),
                None,
            )

    async def read(self, version: ResourceVersion) -> bytes:
        return await asyncio.to_thread(self._read, version)

    def _read(self, version: ResourceVersion) -> bytes:
        with self._files.guard():
            self._reload_unlocked()
            known = any(
                item.version_id == version.version_id
                for item in self._versions.get(version.resource_id, ())
            )
            if not known:
                raise KeyError((version.resource_id, version.version_id))
            content = self._files.content_path(version).read_bytes()
            if _digest_bytes(content) != version.digest:
                raise ValueError(
                    f"catalog content digest mismatch for {version.resource_id}"
                )
            if len(content) != version.size_bytes:
                raise ValueError(
                    f"catalog content size mismatch for {version.resource_id}"
                )
            return content

    async def alias(self, alias: str, version: ResourceVersion) -> None:
        await await_known_outcome(asyncio.to_thread(self._alias, alias, version))

    def _alias(self, alias: str, version: ResourceVersion) -> None:
        with self._files.guard():
            self._reload_unlocked()
            if not any(
                item.version_id == version.version_id
                for item in self._versions.get(version.resource_id, ())
            ):
                raise KeyError((version.resource_id, version.version_id))
            self._aliases[alias] = version
            self._files.write_aliases(self._aliases)

    async def resolve_alias(self, alias: str) -> ResourceVersion | None:
        return await asyncio.to_thread(self._resolve_alias, alias)

    def _resolve_alias(self, alias: str) -> ResourceVersion | None:
        with self._files.guard():
            self._reload_unlocked()
            return self._aliases.get(alias)

    async def list_versions(self, resource_id: str) -> list[ResourceVersion]:
        return await asyncio.to_thread(self._list_versions, resource_id)

    def _list_versions(self, resource_id: str) -> list[ResourceVersion]:
        with self._files.guard():
            self._reload_unlocked()
            return list(self._versions.get(resource_id, ()))

    async def record_active_set(
        self,
        active_set: CatalogActiveSetInput,
    ) -> ActiveSetFingerprint:
        return await await_known_outcome(
            asyncio.to_thread(self._record_active_set, active_set)
        )

    def _record_active_set(
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
        with self._files.guard():
            self._reload_unlocked()
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
            self._files.write_records(self._records)
            return fingerprint

    async def get_active_set(self, session_id: str) -> ActiveSetFingerprint | None:
        return await asyncio.to_thread(self._get_active_set, session_id)

    def _get_active_set(self, session_id: str) -> ActiveSetFingerprint | None:
        with self._files.guard():
            self._reload_unlocked()
            record = self._records.get(session_id)
            return record.fingerprint if record is not None else None

    async def query_active_sets(
        self,
        query: CatalogQuery,
    ) -> list[CatalogActiveSetRecord]:
        return await asyncio.to_thread(self._query_active_sets, query)

    def _query_active_sets(
        self,
        query: CatalogQuery,
    ) -> list[CatalogActiveSetRecord]:
        if query.limit is not None and query.limit < 0:
            raise ValueError("catalog query limit cannot be negative")
        with self._files.guard():
            self._reload_unlocked()
            records = list(self._records.values())
        if query.session_id is not None:
            records = [
                record for record in records if record.session_id == query.session_id
            ]
        if query.root_session_id is not None:
            records = [
                record
                for record in records
                if record.root_session_id == query.root_session_id
            ]
        if query.parent_session_id is not None:
            records = [
                record
                for record in records
                if record.parent_session_id == query.parent_session_id
            ]
        if query.scenario is not None:
            records = [
                record for record in records if record.scenario == query.scenario
            ]
        if query.provider is not None:
            records = [
                record for record in records if record.provider == query.provider
            ]
        if query.digest is not None:
            records = [
                record
                for record in records
                if record.fingerprint.digest == query.digest
            ]
        if query.atom_name is not None:
            records = [
                record
                for record in records
                if any(
                    atom.name == query.atom_name for atom in record.fingerprint.atoms
                )
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
                    atom.version is not None
                    and atom.version.version_id == query.version_id
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

    def _reload_unlocked(self) -> None:
        self._versions = self._files.load_versions()
        self._aliases = self._files.load_aliases()
        self._records = self._files.load_records()
        known_versions = {
            (version.resource_id, version.version_id)
            for versions in self._versions.values()
            for version in versions
        }
        for alias, version in self._aliases.items():
            if (version.resource_id, version.version_id) not in known_versions:
                raise ValueError(
                    f"catalog alias {alias!r} references an unknown version"
                )


class _CatalogFiles:
    """Locked filesystem encoding for catalog state."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._versions_path = root / "versions.json"
        self._aliases_path = root / "aliases.json"
        self._active_sets_path = root / "active_sets.json"
        self._content_root = root / "content"
        self._lock_path = root / ".lock"
        self._process_lock = threading.RLock()
        root.mkdir(parents=True, exist_ok=True)
        self._content_root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def guard(self) -> Iterator[None]:
        with self._process_lock:
            self._root.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load_versions(self) -> dict[str, list[ResourceVersion]]:
        rows = _read_json_array(self._versions_path)
        versions: dict[str, list[ResourceVersion]] = {}
        for item in rows:
            version = deserialize_resource_version(item)
            versions.setdefault(version.resource_id, []).append(version)
        return versions

    def write_versions(
        self,
        versions: Mapping[str, Sequence[ResourceVersion]],
    ) -> None:
        rows = [
            serialize_resource_version(version)
            for resource_id in sorted(versions)
            for version in versions[resource_id]
        ]
        _atomic_write_json_array(self._versions_path, rows)

    def load_aliases(self) -> dict[str, ResourceVersion]:
        rows = _read_json_array(self._aliases_path)
        aliases: dict[str, ResourceVersion] = {}
        for item in rows:
            alias = item.get("alias")
            version_data = item.get("version")
            if not isinstance(alias, str) or not alias:
                raise ValueError("catalog alias row has no alias")
            if not isinstance(version_data, Mapping):
                raise ValueError(f"catalog alias {alias!r} has no version")
            aliases[alias] = deserialize_resource_version(version_data)
        return aliases

    def write_aliases(self, aliases: Mapping[str, ResourceVersion]) -> None:
        _atomic_write_json_array(
            self._aliases_path,
            [
                {
                    "alias": alias,
                    "version": serialize_resource_version(version),
                }
                for alias, version in sorted(aliases.items())
            ],
        )

    def load_records(self) -> dict[str, CatalogActiveSetRecord]:
        rows = _read_json_array(self._active_sets_path)
        return {
            record.session_id: record
            for row in rows
            for record in (deserialize_catalog_record(row),)
        }

    def write_records(
        self,
        records: Mapping[str, CatalogActiveSetRecord],
    ) -> None:
        _atomic_write_json_array(
            self._active_sets_path,
            [serialize_catalog_record(record) for _, record in sorted(records.items())],
        )

    def content_path(self, version: ResourceVersion) -> Path:
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
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{path} must contain only JSON objects")
    return list(value)


def _atomic_write_json_array(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(path, json.dumps(rows, sort_keys=True, indent=2) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["JsonCatalogStore"]
