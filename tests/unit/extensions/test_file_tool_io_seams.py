"""Regression coverage for builtin file-tool IO seams."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agentm.core.abi.resource import PathClass, WriteResult, WriterAuthor
from agentm.extensions.builtin import (
    tool_edit,
    tool_find,
    tool_grep,
    tool_ls,
    tool_read,
    tool_write,
)


class _RecordingFileOperations:
    def __init__(self) -> None:
        self.reads: list[str] = []
        self.accesses: list[str] = []
        self.listed: list[str] = []
        self.dirs = {"/repo", "/repo/src"}
        self.files = {
            "/repo/.gitignore": b"ignored.py\n",
            "/repo/README.md": b"hello readme\n",
            "/repo/src/ignored.py": b"needle hidden\n",
            "/repo/src/target.py": b"needle\nother\n",
        }

    async def read_file(self, path: str) -> bytes:
        self.reads.append(path)
        return self.files[path]

    async def access(self, path: str) -> bool:
        self.accesses.append(path)
        return path in self.dirs or path in self.files

    async def is_dir(self, path: str) -> bool:
        return path in self.dirs

    async def list_dir(self, path: str) -> list[str]:
        self.listed.append(path)
        prefix = path.rstrip("/") + "/"
        names: set[str] = set()
        for candidate in [*self.dirs, *self.files]:
            if candidate == path or not candidate.startswith(prefix):
                continue
            remainder = candidate[len(prefix) :]
            names.add(remainder.split("/", 1)[0])
        return sorted(names)


class _FailingFileOperations:
    async def read_file(self, path: str) -> bytes:
        raise AssertionError(f"FileOperations.read_file should not be used for writes: {path}")

    async def access(self, path: str) -> bool:
        raise AssertionError(f"FileOperations.access should not be used for writes: {path}")

    async def is_dir(self, path: str) -> bool:
        raise AssertionError(f"FileOperations.is_dir should not be used for writes: {path}")

    async def list_dir(self, path: str) -> list[str]:
        raise AssertionError(f"FileOperations.list_dir should not be used for writes: {path}")


@dataclass
class _RecordingResourceWriter:
    reads: dict[str, bytes] = field(default_factory=lambda: {"note.txt": b"before\n"})
    writes: list[tuple[str, bytes, str, WriterAuthor]] = field(default_factory=list)
    replacements: list[tuple[str, bytes, bytes, str, WriterAuthor]] = field(default_factory=list)

    async def read(self, path: str) -> bytes:
        return self.reads[path]

    async def write(
        self,
        path: str,
        content: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        self.writes.append((path, content, rationale, author))
        return _write_result(path)

    async def replace(
        self,
        path: str,
        old: bytes,
        new: bytes,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        self.replacements.append((path, old, new, rationale, author))
        return _write_result(path)

    async def delete(
        self,
        path: str,
        *,
        rationale: str,
        author: WriterAuthor = "agent",
    ) -> WriteResult:
        del rationale, author
        return _write_result(path)

    def classify(self, path: str) -> PathClass:
        del path
        return "unmanaged"

    def restore(self, path: PurePosixPath, version: str) -> None:
        del path, version

    def current_version_for_path(self, path: str) -> str | None:
        del path
        return None

    def batch(self, *, rationale: str, author: WriterAuthor = "agent") -> Any:
        del rationale, author
        raise NotImplementedError


class _FakeAPI:
    def __init__(self, *, file_ops: Any, writer: Any | None = None) -> None:
        self.cwd = "/repo"
        self.tools: list[Any] = []
        self._file_ops = file_ops
        self._writer = writer

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def get_operations(self) -> Any:
        return SimpleNamespace(file=self._file_ops)

    def get_resource_writer(self) -> Any:
        if self._writer is None:
            raise AssertionError("ResourceWriter should not be used for read-only tools")
        return self._writer


@pytest.mark.asyncio
async def test_read_only_file_atoms_use_file_operations() -> None:
    file_ops = _RecordingFileOperations()
    api = _FakeAPI(file_ops=file_ops)

    for module in (tool_read, tool_ls, tool_find, tool_grep):
        module.install(api, {})

    by_name = {tool.name: tool for tool in api.tools}
    read_result = await by_name["read"].execute({"path": "/repo/README.md"})
    ls_result = await by_name["ls"].execute({"path": "."})
    find_result = await by_name["find"].execute({"pattern": "target.py", "path": "."})
    grep_result = await by_name["grep"].execute({"pattern": "needle", "path": "."})

    assert read_result.is_error is False
    assert "hello readme" in read_result.content[0].text
    assert "src/" in ls_result.content[0].text
    assert "README.md" in ls_result.content[0].text
    assert "src/target.py" in find_result.content[0].text
    assert "src/ignored.py" not in find_result.content[0].text
    assert "src/target.py:1: needle" in grep_result.content[0].text
    assert "src/ignored.py" not in grep_result.content[0].text
    assert "/repo/README.md" in file_ops.reads
    assert "/repo/.gitignore" in file_ops.reads
    assert "/repo" in file_ops.listed
    assert "/repo/src" in file_ops.listed


@pytest.mark.asyncio
async def test_write_atoms_use_resource_writer_exclusively() -> None:
    writer = _RecordingResourceWriter()
    api = _FakeAPI(file_ops=_FailingFileOperations(), writer=writer)

    tool_write.install(cast(Any, api), {})
    tool_edit.install(cast(Any, api), {})

    by_name = {tool.name: tool for tool in api.tools}
    write_result = await by_name["write"].execute(
        {"path": "created.txt", "content": "payload", "rationale": "test write"}
    )
    edit_result = await by_name["edit"].execute(
        {
            "path": "note.txt",
            "old_string": "before",
            "new_string": "after",
            "rationale": "test edit",
        }
    )

    assert write_result.is_error is False
    assert edit_result.is_error is False
    assert writer.writes == [("created.txt", b"payload", "test write", "agent")]
    assert writer.replacements == [("note.txt", b"before\n", b"after\n", "test edit", "agent")]


def _write_result(path: str) -> WriteResult:
    return WriteResult(
        path=path,
        path_class="unmanaged",
        committed=False,
        commit_sha_before=None,
        commit_sha_after=None,
    )
