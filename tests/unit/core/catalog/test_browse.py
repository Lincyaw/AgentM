from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from agentm.core._internal.catalog import _layout
from agentm.core._internal.catalog.browse import (
    UnparseableManifestError,
    current_version,
    get_manifest_at,
    get_source_at,
    list_versions,
    runs_for,
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


def _commit_atom(root: Path, *, body: str, message: str) -> str:
    atom_path = root / "src" / "agentm" / "extensions" / "builtin" / "tool_read.py"
    atom_path.parent.mkdir(parents=True, exist_ok=True)
    atom_path.write_text(body, encoding="utf-8")
    _git(root, "add", str(atom_path.relative_to(root)))
    _git(root, "commit", "-m", message, "--quiet")
    return _git(root, "rev-parse", "HEAD")


def _manifest_source(*, description: str, computed: bool = False) -> str:
    manifest_value = (
        'os.environ["TOOL_DESC"]'
        if computed
        else repr(description)
    )
    return f"""from __future__ import annotations

import os

from agentm.extensions import ExtensionManifest

SIDE_EFFECT = 1 / 0

MANIFEST = ExtensionManifest(
    name="tool_read",
    description={manifest_value},
    registers=("tool:read",),
    config_schema={{"type": "object", "properties": {{"path": {{"type": "string"}}}}}},
    api_version=1,
    affects=("read.success_rate",),
    tier=1,
)
"""


def test_list_versions_and_current_version_use_git_history(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    first_sha = _commit_atom(
        tmp_path,
        body=_manifest_source(description="v1"),
        message="add tool_read v1",
    )
    second_sha = _commit_atom(
        tmp_path,
        body=_manifest_source(description="v2"),
        message="update tool_read v2",
    )

    assert list_versions("tool_read", tmp_path) == [second_sha, first_sha]
    assert current_version("tool_read", tmp_path) == second_sha
    assert list_versions("src/agentm/extensions/builtin/missing.py", tmp_path) == []


def test_get_source_at_reads_historical_blob_and_validates_inputs(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    first_source = _manifest_source(description="v1")
    first_sha = _commit_atom(
        tmp_path,
        body=first_source,
        message="add tool_read v1",
    )
    second_sha = _commit_atom(
        tmp_path,
        body=_manifest_source(description="v2"),
        message="update tool_read v2",
    )
    del second_sha

    assert get_source_at("tool_read", first_sha, tmp_path) == first_source.encode("utf-8")

    with pytest.raises(KeyError):
        get_source_at("src/agentm/extensions/builtin/missing.py", first_sha, tmp_path)

    with pytest.raises(ValueError):
        get_source_at("tool_read", "not-a-sha", tmp_path)


def test_get_manifest_at_ast_parses_without_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.setenv("TOOL_DESC", "env description")
    sha = _commit_atom(
        tmp_path,
        body=_manifest_source(description="static description"),
        message="add tool_read",
    )

    payload = get_manifest_at("tool_read", sha, tmp_path)

    assert payload == {
        "name": "tool_read",
        "description": "static description",
        "registers": ("tool:read",),
        "config_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
        "api_version": 1,
        "affects": ("read.success_rate",),
        "tier": 1,
        "content_hash": sha,
    }


def test_get_manifest_at_rejects_computed_values(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    sha = _commit_atom(
        tmp_path,
        body=_manifest_source(description="ignored", computed=True),
        message="add computed manifest",
    )

    with pytest.raises(UnparseableManifestError):
        get_manifest_at("tool_read", sha, tmp_path)


def test_runs_for_intersects_trace_symlink_sets_by_git_sha(tmp_path: Path) -> None:
    first_sha = "a" * 40
    second_sha = "b" * 40
    trace_root = tmp_path / ".agentm" / "observability"
    trace_root.mkdir(parents=True, exist_ok=True)
    for trace_id in ("trace-1", "trace-2"):
        (trace_root / f"{trace_id}.jsonl").write_text(
            json.dumps({"trace_id": trace_id}) + "\n",
            encoding="utf-8",
        )

    tool_read_runs = _layout.atom_runs_dir("tool_read", first_sha, root=tmp_path)
    tool_find_runs = _layout.atom_runs_dir("tool_find", second_sha, root=tmp_path)
    tool_read_runs.mkdir(parents=True, exist_ok=True)
    tool_find_runs.mkdir(parents=True, exist_ok=True)

    os.symlink(trace_root / "trace-1.jsonl", tool_read_runs / "trace-1")
    os.symlink(trace_root / "trace-2.jsonl", tool_read_runs / "trace-2")
    os.symlink(trace_root / "trace-2.jsonl", tool_find_runs / "trace-2")

    assert runs_for(
        {
            "atoms": {
                "tool_read": f"tool_read@{first_sha}",
                "tool_find": f"tool_find@{second_sha}",
            }
        },
        tmp_path,
    ) == ["trace-2"]
