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
    assert not is_err_ok and "# guide" in text_ok

    text_blocked, is_err_blocked = _read(api, str(leak))
    assert is_err_blocked
    assert "Access denied" in text_blocked
    assert "allow_globs" in text_blocked






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


