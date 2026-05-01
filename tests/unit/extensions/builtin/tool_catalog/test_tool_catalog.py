from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from agentm.extensions.validate import validate_builtin
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


async def _session(tmp_path: Path) -> AgentSession:
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_catalog", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )


def _seed_current(atom_root: Path, version: str) -> None:
    current = atom_root / "current"
    try:
        current.symlink_to(version)
    except OSError:
        current.write_text(version, encoding="utf-8")


def _seed_atom(
    tmp_path: Path,
    *,
    atom: str,
    version: str,
    manifest: dict[str, object] | None = None,
) -> Path:
    version_dir = tmp_path / ".agentm" / "catalog" / "atoms" / atom / version
    version_dir.mkdir(parents=True)
    payload = manifest or {"name": atom, "version": version}
    (version_dir / "manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )
    runs_dir = version_dir / "runs"
    runs_dir.mkdir()
    _seed_current(version_dir.parent, version)
    return version_dir


@pytest.mark.asyncio
async def test_M3_list_versions_includes_current_after_seed(tmp_path: Path) -> None:
    _seed_atom(tmp_path, atom="tool_read", version="abc123def456")
    session = await _session(tmp_path)

    tool = next(tool for tool in session.tools if tool.name == "catalog_list_versions")
    result = await tool.execute({"atom": "tool_read"})

    assert not result.is_error
    assert result.details == ["abc123def456"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_list_versions_unknown_atom_returns_empty_list(tmp_path: Path) -> None:
    session = await _session(tmp_path)

    tool = next(tool for tool in session.tools if tool.name == "catalog_list_versions")
    result = await tool.execute({"atom": "missing_atom"})

    assert not result.is_error
    assert result.details == []
    await session.shutdown()


@pytest.mark.asyncio
async def test_get_manifest_returns_yaml_content(tmp_path: Path) -> None:
    manifest = {
        "name": "tool_read",
        "version": "abc123def456",
        "author": "human",
        "authored_at": "2026-05-01T00:00:00Z",
    }
    _seed_atom(
        tmp_path,
        atom="tool_read",
        version="abc123def456",
        manifest=manifest,
    )
    session = await _session(tmp_path)

    tool = next(tool for tool in session.tools if tool.name == "catalog_get_manifest")
    result = await tool.execute({"atom": "tool_read", "version": "abc123def456"})

    assert not result.is_error
    assert result.details == manifest
    await session.shutdown()


@pytest.mark.asyncio
async def test_get_manifest_unknown_version_returns_error_result(tmp_path: Path) -> None:
    session = await _session(tmp_path)

    tool = next(tool for tool in session.tools if tool.name == "catalog_get_manifest")
    result = await tool.execute({"atom": "tool_read", "version": "missing"})

    assert result.is_error
    assert "tool_read@missing" in result.content[0].text
    await session.shutdown()


@pytest.mark.asyncio
async def test_runs_for_returns_trace_ids(tmp_path: Path) -> None:
    version_dir = _seed_atom(tmp_path, atom="tool_read", version="abc123def456")
    runs_dir = version_dir / "runs"
    target_root = tmp_path / ".agentm" / "observability"
    target_root.mkdir(parents=True)
    for trace_id in ("trace-1", "trace-2"):
        target = target_root / f"{trace_id}.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        link = runs_dir / trace_id
        try:
            link.symlink_to(target)
        except OSError:
            link.write_text(str(target), encoding="utf-8")
    session = await _session(tmp_path)

    tool = next(tool for tool in session.tools if tool.name == "catalog_runs_for")
    result = await tool.execute({"fingerprint": {"tool_read": "tool_read@abc123def456"}})

    assert not result.is_error
    assert result.details == ["trace-1", "trace-2"]
    await session.shutdown()


def test_atom_passes_section_11_validator() -> None:
    issues = validate_builtin()

    assert not [
        issue for issue in issues if issue.module_path.endswith("tool_catalog")
    ], issues


def test_atom_imports_only_core_catalog_public_surface() -> None:
    source_path = Path("src/agentm/extensions/builtin/tool_catalog.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("agentm.core.catalog."):
                violations.append(node.module)
            if node.module in {
                "agentm.core.catalog._layout",
                "agentm.core.catalog.freeze",
                "agentm.core.catalog.indexer",
            }:
                violations.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agentm.core.catalog."):
                    violations.append(alias.name)
    assert violations == []
