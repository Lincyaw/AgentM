"""Path-allow / -deny gate for the ``tool_read`` builtin atom.

The gate is the only thing standing between an agent given the ``read``
tool and arbitrary filesystem reads. In RCA evals this matters: the per-
case dataset directory contains ``label.txt`` / ``injection.json`` /
``causal_graph.json`` next to the telemetry parquets the agent legitimately
needs. Without a fence, an agent that figures out the directory layout
can bypass the eval by reading the ground-truth file directly.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi.operations import FileOperations
from agentm.extensions.builtin import tool_read


class _StubFileOps(FileOperations):
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    async def read_file(self, path: str) -> bytes:
        return self._files[path]

    async def write_file(self, path: str, data: bytes) -> None:
        self._files[path] = data

    async def remove_file(self, path: str) -> None:
        self._files.pop(path, None)

    async def exists(self, path: str) -> bool:
        return path in self._files

    async def stat(self, path: str) -> Any:
        raise NotImplementedError

    async def mkdir(self, path: str, *, parents: bool = False, exist_ok: bool = False) -> None:
        raise NotImplementedError

    async def list_dir(self, path: str) -> list[str]:
        raise NotImplementedError


class _Api:
    def __init__(self, cwd: str, file_ops: FileOperations) -> None:
        self.cwd = cwd
        self._file_ops = file_ops
        self.tools: dict[str, Any] = {}

    def register_tool(self, t: Any) -> None:
        self.tools[t.name] = t

    def get_operations(self) -> Any:
        class _Ops:
            file = self._file_ops
        return _Ops()


def _install(api: _Api, **config: Any) -> None:
    tool_read.install(api, dict(config))


def _read(api: _Api, path: str) -> tuple[str, bool]:
    result = asyncio.run(api.tools["read"].fn({"path": path}))
    return result.content[0].text, bool(result.is_error)


def test_no_globs_means_no_gate(tmp_path: Path) -> None:
    target = tmp_path / "anywhere.txt"
    target.write_text("ok")
    api = _Api(str(tmp_path), _StubFileOps({str(target): b"ok"}))
    _install(api)
    text, is_err = _read(api, str(target))
    assert not is_err
    assert text == "ok"


def test_allow_globs_block_paths_outside_scope(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "guide.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# guide")
    leak = tmp_path / "label.txt"
    leak.write_text("answer=foo")
    api = _Api(
        str(tmp_path),
        _StubFileOps({str(skill): b"# guide", str(leak): b"answer=foo"}),
    )
    _install(api, allow_globs=["skills/**"])

    text_ok, is_err_ok = _read(api, str(skill))
    assert not is_err_ok and text_ok == "# guide"

    text_blocked, is_err_blocked = _read(api, str(leak))
    assert is_err_blocked
    assert "Access denied" in text_blocked
    assert "allow_globs" in text_blocked


def test_relative_allow_glob_anchors_at_cwd(tmp_path: Path) -> None:
    """A scenario manifest writing ``skills/**`` must mean ``<cwd>/skills/**``
    regardless of the user's working directory at agent launch."""

    inside = tmp_path / "skills" / "a.md"
    inside.parent.mkdir(parents=True)
    inside.write_text("a")
    outside = tmp_path / "skills_sibling" / "evil.md"
    outside.parent.mkdir(parents=True)
    outside.write_text("evil")
    api = _Api(
        str(tmp_path),
        _StubFileOps({str(inside): b"a", str(outside): b"evil"}),
    )
    _install(api, allow_globs=["skills/**"])

    _, is_err_in = _read(api, str(inside))
    assert not is_err_in
    _, is_err_out = _read(api, str(outside))
    assert is_err_out


def test_deny_globs_evaluated_after_allow(tmp_path: Path) -> None:
    """Eval-data layout: agent legitimately reads telemetry parquets in
    ``dataset/<case>/`` but ``label.txt`` in the same directory must
    stay out of reach."""

    case = tmp_path / "dataset" / "case1"
    case.mkdir(parents=True)
    parquet = case / "abnormal_traces.parquet"
    label = case / "label.txt"
    parquet.write_text("ok")
    label.write_text("answer")
    api = _Api(
        str(tmp_path),
        _StubFileOps({str(parquet): b"ok", str(label): b"answer"}),
    )
    _install(api, allow_globs=["dataset/**"], deny_globs=["**/label.txt"])

    _, is_err_ok = _read(api, str(parquet))
    assert not is_err_ok

    text_blocked, is_err_blocked = _read(api, str(label))
    assert is_err_blocked
    assert "deny_glob" in text_blocked


def test_symlink_resolved_before_matching(tmp_path: Path) -> None:
    """A symlink that lives inside the allow tree but POINTS outside must
    still be rejected — otherwise the gate is trivially defeatable."""

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    sym = skills_dir / "shortcut.txt"
    os.symlink(secret, sym)

    api = _Api(
        str(tmp_path),
        _StubFileOps({str(secret): b"secret"}),
    )
    _install(api, allow_globs=["skills/**"])

    text, is_err = _read(api, str(sym))
    assert is_err
    assert "Access denied" in text


@pytest.mark.parametrize("missing_or_bad", [None, "not-a-list", 42])
def test_invalid_glob_config_treated_as_no_gate(
    missing_or_bad: Any, tmp_path: Path
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("x")
    api = _Api(str(tmp_path), _StubFileOps({str(target): b"x"}))
    _install(api, allow_globs=missing_or_bad)
    text, is_err = _read(api, str(target))
    assert not is_err and text == "x"
