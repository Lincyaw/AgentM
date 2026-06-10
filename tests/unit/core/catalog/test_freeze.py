"""Tests for freezing atom source into the catalog."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agentm.core.runtime.catalog import freeze_current
from agentm.extensions import ExtensionManifest


def _manifest(name: str = "tool_read") -> ExtensionManifest:
    return ExtensionManifest(
        name=name,
        description="Read tool with truncation guard",
        registers=("tool:read",),
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
    _git(root, "init", "-q", "-b", "agent-tests")
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




