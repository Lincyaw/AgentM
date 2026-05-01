from __future__ import annotations

import json
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
async def test_tool_trajectory_loader_install_smoke(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_trajectory_loader", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    assert [tool.name for tool in session.tools] == [
        "load_trajectory",
        "summarize_trajectory",
        "find_event",
        "compare_trajectories",
    ]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_trajectory_loader_loads_and_summarizes_jsonl(tmp_path: Path) -> None:
    file_ops = RecordingFileOps(
        {
            "traj.jsonl": (
                b'{"channel":"tool_call","event":{"name":"read"}}\n'
                b'{"channel":"tool_result","event":{"name":"read"}}\n'
            )
        }
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.tool_trajectory_loader", {"file_ops": file_ops})
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    tools = {tool.name: tool for tool in session.tools}

    loaded = await tools["load_trajectory"].execute({"path": "traj.jsonl"})
    summary = await tools["summarize_trajectory"].execute({})

    assert not loaded.is_error
    assert file_ops.read_calls == ["traj.jsonl"]
    assert isinstance(summary.content[0], TextContent)
    assert json.loads(summary.content[0].text) == {
        "channels": {"tool_call": 1, "tool_result": 1},
        "event_count": 2,
        "path": "traj.jsonl",
    }
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_trajectory_loader_returns_error_for_missing_file(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.tool_trajectory_loader", {"file_ops": RecordingFileOps({})})
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    result = await session.tools[0].execute({"path": "missing.jsonl"})

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "missing.jsonl" in result.content[0].text
    await session.shutdown()
