from __future__ import annotations

from pathlib import Path

import pytest

from agentm.core.kernel import TextContent
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class RecordingFileOps:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = dict(files)
        self.read_calls: list[str] = []

    async def read_file(self, path: str) -> bytes:
        self.read_calls.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content

    async def access(self, path: str) -> bool:
        return path in self.files

    async def list_dir(self, path: str) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_tool_read_install_smoke(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_read", {})],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    assert [tool.name for tool in session.tools] == ["read"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_read_executes_with_line_slicing(tmp_path: Path) -> None:
    file_ops = RecordingFileOps({"note.txt": b"zero\none\ntwo\nthree\n"})
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_read", {"file_ops": file_ops})],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    tool = session.tools[0]
    result = await tool.execute({"path": "note.txt", "offset": 1, "limit": 2})

    assert file_ops.read_calls == ["note.txt"]
    assert not result.is_error
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "one\ntwo"
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_read_returns_error_result_for_missing_file(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.tool_read", {"file_ops": RecordingFileOps({})})
            ],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute({"path": "missing.txt"})

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "missing.txt" in result.content[0].text
    await session.shutdown()
