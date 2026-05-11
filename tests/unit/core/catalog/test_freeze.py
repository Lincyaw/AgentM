"""Tests for freezing atom source into the catalog."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agentm.core.runtime.catalog import freeze_current, list_atoms
from agentm.core.runtime.catalog._layout import atom_runs_dir, atom_version_dir
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


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    (root / "README.md").write_text("baseline\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "initial", "--quiet")


def _write_atom(root: Path, name: str, source: str) -> None:
    atom_path = root / "src" / "agentm" / "extensions" / "builtin" / f"{name}.py"
    atom_path.parent.mkdir(parents=True, exist_ok=True)
    atom_path.write_text(source, encoding="utf-8")


def _configure_manifest(tmp_path: Path) -> None:
    from agentm.core._internal.catalog import manifest as manifest_mod

    manifest_path = tmp_path / "core-manifest.yaml"
    manifest_path.write_text(
        "version: 1\n"
        "constitution:\n"
        "  paths:\n"
        "    - core-manifest.yaml\n"
        "managed:\n"
        "  globs:\n"
        "    - src/agentm/extensions/builtin/**.py\n"
        "extension_api:\n"
        "  current: 1\n"
        "  semver_rules: {major: x, minor: x, patch: x}\n"
        "  deprecation:\n"
        "    grace: 1\n"
        "reload:\n"
        "  tier_2_atoms: []\n",
        encoding="utf-8",
    )
    manifest_mod.configure_manifest_path(manifest_path)


def test_freeze_returns_git_sha_and_creates_runs_dir(
    tmp_path: Path,
) -> None:
    _configure_manifest(tmp_path)
    _init_repo(tmp_path)
    source = "def install(api, config):\n    return None\n"
    _write_atom(tmp_path, "tool_read", source)

    version_key = freeze_current("tool_read", source, _manifest(), root=tmp_path)

    assert len(version_key) == 40
    assert atom_version_dir("tool_read", version_key, root=tmp_path).is_dir()
    runs_dir = atom_runs_dir("tool_read", version_key, root=tmp_path)
    assert runs_dir.is_dir()
    assert list(runs_dir.iterdir()) == []


def test_M1_idempotent_reuses_existing_git_version(
    tmp_path: Path,
) -> None:
    _configure_manifest(tmp_path)
    _init_repo(tmp_path)
    source = "def install(api, config):\n    return None\n"
    _write_atom(tmp_path, "tool_read", source)

    first = freeze_current("tool_read", source, _manifest(), root=tmp_path)
    second = freeze_current("tool_read", source, _manifest(), root=tmp_path)

    assert second == first


def test_freeze_changes_version_after_source_changes(
    tmp_path: Path,
) -> None:
    _configure_manifest(tmp_path)
    _init_repo(tmp_path)
    first_source = "def install(api, config):\n    return 'first'\n"
    second_source = "def install(api, config):\n    return 'second'\n"
    _write_atom(tmp_path, "tool_read", first_source)

    first = freeze_current("tool_read", first_source, _manifest(), root=tmp_path)
    _write_atom(tmp_path, "tool_read", second_source)
    second = freeze_current("tool_read", second_source, _manifest(), root=tmp_path)

    assert first != second
    assert atom_version_dir("tool_read", first, root=tmp_path).is_dir()
    assert atom_version_dir("tool_read", second, root=tmp_path).is_dir()


def test_list_atoms_returns_current_catalog_metadata(
    tmp_path: Path,
) -> None:
    _configure_manifest(tmp_path)
    _init_repo(tmp_path)
    tool_read_source = "def install(api, config):\n    return 'first'\n"
    tool_bash_source = "def install(api, config):\n    return 'bash'\n"
    _write_atom(tmp_path, "tool_read", tool_read_source)
    _write_atom(tmp_path, "tool_bash", tool_bash_source)

    tool_read_version = freeze_current(
        "tool_read",
        tool_read_source,
        _manifest(),
        root=tmp_path,
    )
    tool_bash_version = freeze_current(
        "tool_bash",
        tool_bash_source,
        _manifest(name="tool_bash"),
        root=tmp_path,
    )

    assert list_atoms(root=tmp_path) == [
        {
            "name": "tool_bash",
            "current_hash": tool_bash_version,
            "tier": 1,
            "api_version": 1,
        },
        {
            "name": "tool_read",
            "current_hash": tool_read_version,
            "tier": 1,
            "api_version": 1,
        },
    ]
