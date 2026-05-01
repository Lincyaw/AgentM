from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class FakeLsOps:
    def __init__(self, names: list[str], directories: set[str] | None = None) -> None:
        self.names = names
        self.directories = directories or set()
        self.used = False

    async def exists(self, path: str) -> bool:
        return True

    async def stat(self, path: str):
        self.used = True
        basename = os.path.basename(path)
        mode = stat.S_IFDIR if basename in self.directories or basename not in self.names else stat.S_IFREG
        return os.stat_result((mode, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    async def listdir(self, path: str) -> list[str]:
        self.used = True
        return list(self.names)


async def _session(tmp_path: Path, config: dict) -> AgentSession:
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_ls", config)],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )


@pytest.mark.asyncio
async def test_ls_uses_custom_ops_and_formats_sorted_entries(tmp_path: Path) -> None:
    ops = FakeLsOps(["beta", "Alpha", "docs"], {"docs"})
    session = await _session(tmp_path, {"ops": ops})

    result = await session.tools[0].execute({"path": "."})

    assert ops.used is True
    assert result.content[0].text == "Alpha\nbeta\ndocs/"
    await session.shutdown()


@pytest.mark.asyncio
async def test_ls_reports_byte_truncation(tmp_path: Path) -> None:
    names = [f"file_{index:04d}_{'x' * 90}" for index in range(600)]
    ops = FakeLsOps(names)
    session = await _session(tmp_path, {"ops": ops})

    result = await session.tools[0].execute({"path": ".", "limit": 600})

    assert "[Truncated:" in result.content[0].text
    await session.shutdown()
