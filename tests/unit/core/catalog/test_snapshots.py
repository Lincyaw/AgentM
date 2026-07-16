"""Fail-stop checks for content-addressed catalog snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from agentm.core._internal.catalog import (
    CatalogCorruptionError,
    current_version,
    get_manifest_at,
    get_source_at,
    list_versions,
)
from agentm.core._internal.catalog.hashing import compute_atom_hash
from agentm.core.abi.manifest import ExtensionManifest
from agentm.core.runtime.catalog.freeze import freeze_current


class DemoConfig(BaseModel):
    enabled: bool = True


def _manifest() -> ExtensionManifest:
    return ExtensionManifest(
        name="demo",
        description="catalog snapshot test",
        registers=("tool:demo",),
        config_schema=DemoConfig,
    )


def test_freeze_materializes_validated_immutable_snapshot(tmp_path: Path) -> None:
    source = "def install(api, config):\n    pass\n"
    version = freeze_current("demo", source, _manifest(), root=tmp_path)

    assert version == compute_atom_hash(source)
    assert current_version("demo", tmp_path) == version
    assert list_versions("demo", tmp_path) == [version]
    assert get_source_at("demo", version, tmp_path) == source.encode()
    manifest = get_manifest_at("demo", version, tmp_path)
    assert manifest["name"] == "demo"
    assert manifest["content_hash"] == version
    assert manifest["config_schema"] == "DemoConfig"

    # Re-freezing identical bytes is idempotent.
    assert freeze_current("demo", source, _manifest(), root=tmp_path) == version


def test_source_tampering_is_detected_before_catalog_read(tmp_path: Path) -> None:
    source = "def install(api, config):\n    pass\n"
    version = freeze_current("demo", source, _manifest(), root=tmp_path)
    source_path = (
        tmp_path / ".agentm" / "catalog" / "atoms" / "demo" / version / "source.py"
    )
    source_path.write_text(source + "# corrupted\n", encoding="utf-8")

    with pytest.raises(CatalogCorruptionError, match="source hash mismatch"):
        get_source_at("demo", version, tmp_path)
    with pytest.raises(CatalogCorruptionError, match="immutable catalog snapshot"):
        freeze_current("demo", source, _manifest(), root=tmp_path)


def test_manifest_identity_tampering_is_detected(tmp_path: Path) -> None:
    source = "def install(api, config):\n    pass\n"
    version = freeze_current("demo", source, _manifest(), root=tmp_path)
    manifest_path = (
        tmp_path / ".agentm" / "catalog" / "atoms" / "demo" / version / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["name"] = "other"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogCorruptionError, match="manifest name mismatch"):
        get_manifest_at("demo", version, tmp_path)


def test_malformed_version_directory_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / ".agentm" / "catalog" / "atoms" / "demo" / "latest"
    malformed.mkdir(parents=True)

    with pytest.raises(ValueError, match="invalid content hash"):
        list_versions("demo", tmp_path)


def test_catalog_names_cannot_escape_the_catalog_root(tmp_path: Path) -> None:
    manifest = ExtensionManifest(name="../escape", description="invalid")
    with pytest.raises(ValueError, match="invalid atom name"):
        freeze_current("../escape", "x", manifest, root=tmp_path)


def test_catalog_symlink_cannot_redirect_snapshot_outside_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    atoms_dir = tmp_path / ".agentm" / "catalog" / "atoms"
    atoms_dir.mkdir(parents=True)
    (atoms_dir / "demo").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CatalogCorruptionError, match="must not be a symlink"):
        freeze_current("demo", "x", _manifest(), root=tmp_path)
    with pytest.raises(CatalogCorruptionError, match="must not be a symlink"):
        list_versions("demo", tmp_path)
    assert list(outside.iterdir()) == []


def test_snapshot_file_symlink_is_rejected(tmp_path: Path) -> None:
    source = "def install(api, config):\n    pass\n"
    version = compute_atom_hash(source)
    version_dir = tmp_path / ".agentm" / "catalog" / "atoms" / "demo" / version
    version_dir.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text(source, encoding="utf-8")
    (version_dir / "source.py").symlink_to(outside)

    with pytest.raises(CatalogCorruptionError, match="must not be a symlink"):
        freeze_current("demo", source, _manifest(), root=tmp_path)


def test_runs_directory_symlink_is_rejected(tmp_path: Path) -> None:
    source = "def install(api, config):\n    pass\n"
    version = compute_atom_hash(source)
    version_dir = tmp_path / ".agentm" / "catalog" / "atoms" / "demo" / version
    version_dir.mkdir(parents=True)
    outside = tmp_path / "outside-runs"
    outside.mkdir()
    (version_dir / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CatalogCorruptionError, match="must not be a symlink"):
        freeze_current("demo", source, _manifest(), root=tmp_path)


def test_unexpected_atom_file_is_not_silently_ignored(tmp_path: Path) -> None:
    atom_dir = tmp_path / ".agentm" / "catalog" / "atoms" / "demo"
    atom_dir.mkdir(parents=True)
    (atom_dir / "garbage").write_text("unexpected", encoding="utf-8")

    with pytest.raises(CatalogCorruptionError, match="unexpected non-directory"):
        list_versions("demo", tmp_path)
