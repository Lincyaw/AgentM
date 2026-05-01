"""Tests for freezing atom source into the catalog."""

from __future__ import annotations

from pathlib import Path

import yaml

from agentm.core._internal.catalog import freeze_current, list_atoms
from agentm.core._internal.catalog._layout import (
    atom_current_symlink,
    atom_manifest_path,
    atom_runs_dir,
    atom_source_path,
)
from agentm.core._internal.catalog.hashing import compute_atom_hash
from agentm.extensions import ExtensionManifest



def _manifest(name: str = "tool_read") -> ExtensionManifest:
    return ExtensionManifest(
        name=name,
        description="Read tool with truncation guard",
        registers=("tool:read",),
        config_schema={"type": "object"},
        requires=("permission",),
        conflicts=(),
        api_version=1,
        affects=("read.success_rate", "io.latency_ms"),
        tier=1,
    )



def test_freeze_writes_source_and_manifest(tmp_path: Path) -> None:
    source = "def install(api, config):\n    return None\n"

    content_hash = freeze_current("tool_read", source, _manifest(), root=tmp_path)

    source_path = atom_source_path("tool_read", content_hash, root=tmp_path)
    manifest_path = atom_manifest_path("tool_read", content_hash, root=tmp_path)

    assert source_path.read_text(encoding="utf-8") == source
    manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["name"] == "tool_read"
    assert manifest_data["content_hash"] == content_hash
    assert manifest_data["author"] == "human"
    assert manifest_data["parent_hash"] is None
    assert manifest_data["registers"] == ["tool:read"]
    assert manifest_data["affects"] == ["read.success_rate", "io.latency_ms"]



def test_M1_idempotent_no_rewrite_when_hash_exists(tmp_path: Path) -> None:
    source = "def install(api, config):\n    return None\n"
    manifest = _manifest()

    content_hash = freeze_current("tool_read", source, manifest, root=tmp_path)
    source_path = atom_source_path("tool_read", content_hash, root=tmp_path)
    manifest_path = atom_manifest_path("tool_read", content_hash, root=tmp_path)
    source_stat_before = source_path.stat()
    manifest_stat_before = manifest_path.stat()

    second_hash = freeze_current("tool_read", source, manifest, root=tmp_path)

    assert second_hash == content_hash
    assert source_path.stat().st_mtime_ns == source_stat_before.st_mtime_ns
    assert manifest_path.stat().st_mtime_ns == manifest_stat_before.st_mtime_ns



def test_freeze_updates_current_symlink(tmp_path: Path) -> None:
    first_source = "def install(api, config):\n    return 'first'\n"
    second_source = "def install(api, config):\n    return 'second'\n"

    first_hash = freeze_current("tool_read", first_source, _manifest(), root=tmp_path)
    second_hash = freeze_current("tool_read", second_source, _manifest(), root=tmp_path)

    current_path = atom_current_symlink("tool_read", root=tmp_path)
    assert current_path.exists()
    if current_path.is_symlink():
        assert Path(current_path.readlink()).name == second_hash
    else:
        assert current_path.read_text(encoding="utf-8").strip() == second_hash
    assert atom_source_path("tool_read", first_hash, root=tmp_path).exists()



def test_freeze_returns_content_hash(tmp_path: Path) -> None:
    source = "def install(api, config):\n    return 7\n"

    content_hash = freeze_current("tool_read", source, _manifest(), root=tmp_path)

    assert content_hash == compute_atom_hash(source)



def test_freeze_creates_runs_dir(tmp_path: Path) -> None:
    content_hash = freeze_current(
        "tool_read",
        "def install(api, config):\n    return None\n",
        _manifest(),
        root=tmp_path,
    )

    runs_dir = atom_runs_dir("tool_read", content_hash, root=tmp_path)
    assert runs_dir.is_dir()
    assert list(runs_dir.iterdir()) == []



def test_list_atoms_returns_current_catalog_metadata(tmp_path: Path) -> None:
    first_hash = freeze_current(
        "tool_read",
        "def install(api, config):\n    return 'first'\n",
        _manifest(),
        root=tmp_path,
    )
    freeze_current(
        "tool_bash",
        "def install(api, config):\n    return 'bash'\n",
        _manifest(name="tool_bash"),
        root=tmp_path,
    )

    atoms = list_atoms(root=tmp_path)

    assert atoms == [
        {
            "name": "tool_bash",
            "current_hash": compute_atom_hash(
                "def install(api, config):\n    return 'bash'\n"
            ),
            "tier": 1,
            "api_version": 1,
        },
        {
            "name": "tool_read",
            "current_hash": first_hash,
            "tier": 1,
            "api_version": 1,
        },
    ]
