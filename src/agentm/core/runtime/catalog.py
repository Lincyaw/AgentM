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
    CatalogMeta,
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

    __slots__ = ("_active_sets",)

    def __init__(self) -> None:
        self._active_sets: dict[str, ActiveSetFingerprint] = {}

    async def record_active_set(
        self,
        *,
        session_id: str,
        atoms: list[AtomActivation] | tuple[AtomActivation, ...],
    ) -> ActiveSetFingerprint:
        captured = tuple(atoms)
        payload = [_activation_record(atom) for atom in captured]
        fingerprint = ActiveSetFingerprint(
            algorithm="sha256",
            digest=_digest_json(payload),
            atoms=captured,
            metadata={"atom_count": len(captured)},
        )
        self._active_sets[session_id] = fingerprint
        return fingerprint

    async def get_active_set(self, session_id: str) -> ActiveSetFingerprint | None:
        return self._active_sets.get(session_id)


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
    config_record = _json_safe(normalize_atom_config(manifest, config))
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
        return []
    if spec.submodule_search_locations:
        files: list[tuple[str, bytes]] = []
        for root_text in spec.submodule_search_locations:
            root = Path(root_text)
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                try:
                    files.append((str(path.relative_to(root)), path.read_bytes()))
                except OSError:
                    continue
        return files
    if spec.origin is None:
        return []
    path = Path(spec.origin)
    if path.suffix != ".py":
        return []
    try:
        return [(path.name, path.read_bytes())]
    except OSError:
        return []


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

    return _normalized_config(manifest, config)


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


__all__ = [
    "InMemoryAtomCatalog",
    "InMemoryVersionedResourceStore",
    "build_atom_identity_payload",
    "normalize_atom_config",
]
