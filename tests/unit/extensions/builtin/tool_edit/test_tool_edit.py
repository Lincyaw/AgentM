from __future__ import annotations

from pathlib import Path

import pytest

from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class RecordingFileOps:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = dict(files)
        self.write_calls: list[tuple[str, bytes]] = []

    async def read_file(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path: str, content: bytes) -> None:
        self.write_calls.append((path, content))
        self.files[path] = content

    async def access(self, path: str) -> bool:
        return path in self.files

    async def list_dir(self, path: str) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_tool_edit_install_smoke(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_edit", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    assert [tool.name for tool in session.tools] == ["edit"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_edit_replaces_unique_match(tmp_path: Path) -> None:
    file_ops = RecordingFileOps({"note.txt": b"hello world\n"})
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_edit", {"file_ops": file_ops})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute(
        {"path": "note.txt", "old_string": "world", "new_string": "agent"}
    )

    assert not result.is_error
    assert file_ops.write_calls == [("note.txt", b"hello agent\n")]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_edit_returns_error_for_non_unique_match(tmp_path: Path) -> None:
    file_ops = RecordingFileOps({"note.txt": b"dup dup\n"})
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_edit", {"file_ops": file_ops})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute(
        {"path": "note.txt", "old_string": "dup", "new_string": "x"}
    )

    assert result.is_error
    assert "not unique" in result.content[0].text
    assert file_ops.write_calls == []
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_edit_rejects_constitution_paths(tmp_path: Path) -> None:
    file_ops = RecordingFileOps({"src/agentm/harness/session.py": b"nope\n"})
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_edit", {"file_ops": file_ops})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute(
        {
            "path": "src/agentm/harness/session.py",
            "old_string": "nope",
            "new_string": "yep",
        }
    )

    assert result.is_error
    assert "constitution path" in result.content[0].text
    assert file_ops.write_calls == []
    await session.shutdown()
