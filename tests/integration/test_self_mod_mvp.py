"""Acceptance tests for the self-modifiable MVP."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agentm.core.runtime.catalog import freeze_current
from agentm.core.runtime.catalog.indexer import index_trace, rebuild_catalog
from agentm.extensions import ExtensionManifest


SHA_TOOL_READ = "d" * 40


def _manifest(name: str = "tool_read") -> ExtensionManifest:
    return ExtensionManifest(
        name=name,
        description="test atom",
        registers=("tool:read",),
        config_schema={"type": "object"},
        api_version=1,
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


def _write_trace(tmp_path: Path, trace_id: str, records: list[dict[str, object]]) -> Path:
    observability_dir = tmp_path / ".agentm" / "observability"
    observability_dir.mkdir(parents=True, exist_ok=True)
    trace_path = observability_dir / f"{trace_id}.jsonl"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return trace_path


def _record(kind: str, attributes: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "otel/span/v0",
        "kind": kind,
        "trace_id": "trace",
        "span_id": "span",
        "name": kind,
        "attributes": attributes,
        "status": {"code": "OK"},
    }


def _capture_metrics(root: Path) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {}
    for metrics_path in sorted((root / ".agentm" / "catalog").rglob("metrics.jsonl")):
        entries: list[dict[str, object]] = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            payload.pop("indexed_at", None)
            entries.append(payload)
        rows[str(metrics_path.relative_to(root))] = entries
    return rows


@pytest.mark.asyncio
async def test_E5_rebuild_is_idempotent(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-e5",
        [
            _record(
                "session.fingerprint",
                {
                    "core": None,
                    "scenario": None,
                    "atoms": {"tool_read": f"tool_read@{SHA_TOOL_READ}"},
                },
            ),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )

    first = index_trace(trace_path, root=tmp_path)
    assert first.n_atoms_attributed == 1

    before = _capture_metrics(tmp_path)
    assert before

    rebuild_catalog(
        root=tmp_path,
        observability=tmp_path / ".agentm" / "observability",
    )
    after = _capture_metrics(tmp_path)
    assert before == after


@pytest.mark.asyncio
async def test_M1_freeze_idempotent(
    tmp_path: Path,
) -> None:
    _configure_manifest(tmp_path)
    _init_repo(tmp_path)
    source = "def install(api, config):\n    return 'first'\n"
    _write_atom(tmp_path, "tool_read", source)

    first = freeze_current(
        "tool_read",
        source,
        _manifest(),
        root=tmp_path,
    )
    second = freeze_current(
        "tool_read",
        source,
        _manifest(),
        root=tmp_path,
    )

    version_dir = tmp_path / ".agentm" / "catalog" / "atoms" / "tool_read" / first
    assert first == second
    assert version_dir.is_dir()
    children = {child.name for child in version_dir.iterdir()}
    assert children == {"runs"}


@pytest.mark.asyncio
async def test_M3_list_versions_after_first_session(
    tmp_path: Path,
) -> None:
    _configure_manifest(tmp_path)
    _init_repo(tmp_path)
    source = "def install(api, config):\n    return 'read'\n"
    _write_atom(tmp_path, "tool_read", source)
    version_hash = freeze_current(
        "tool_read",
        source,
        _manifest(),
        root=tmp_path,
    )

    from agentm.core._internal.catalog import list_versions

    assert list_versions("tool_read", tmp_path) == [version_hash]
