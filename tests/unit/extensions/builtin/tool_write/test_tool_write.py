from __future__ import annotations

from pathlib import Path

import pytest

from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class RecordingFileOps:
    def __init__(self) -> None:
        self.write_calls: list[tuple[str, bytes]] = []

    async def read_file(self, path: str) -> bytes:
        raise FileNotFoundError(path)

    async def write_file(self, path: str, content: bytes) -> None:
        self.write_calls.append((path, content))

    async def access(self, path: str) -> bool:
        return False

    async def list_dir(self, path: str) -> list[str]:
        return []


class FailingFileOps(RecordingFileOps):
    async def write_file(self, path: str, content: bytes) -> None:
        del content
        raise OSError(path)


@pytest.mark.asyncio
async def test_tool_write_install_smoke(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_write", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    assert [tool.name for tool in session.tools] == ["write"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_write_executes_via_file_ops(tmp_path: Path) -> None:
    file_ops = RecordingFileOps()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_write", {"file_ops": file_ops})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute({"path": "note.txt", "content": "hello"})

    assert not result.is_error
    assert file_ops.write_calls == [("note.txt", b"hello")]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_write_returns_error_result_on_failure(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_write", {"file_ops": FailingFileOps()})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute({"path": "note.txt", "content": "hello"})

    assert result.is_error
    assert "note.txt" in result.content[0].text
    await session.shutdown()
