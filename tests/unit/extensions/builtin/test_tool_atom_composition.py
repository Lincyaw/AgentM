from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.core.kernel import TextContent
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_tool_read_and_tool_write_roundtrip(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.tool_write", {}),
                ("agentm.extensions.builtin.tool_read", {}),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    tools = {tool.name: tool for tool in session.tools}
    path = tmp_path / "roundtrip.txt"

    wrote = await tools["write"].execute({"path": str(path), "content": "alpha\nbeta\n"})
    read = await tools["read"].execute({"path": str(path)})

    assert not wrote.is_error
    assert not read.is_error
    assert isinstance(read.content[0], TextContent)
    assert read.content[0].text == "alpha\nbeta"
    await session.shutdown()


@pytest.mark.asyncio
async def test_hypothesis_store_persists_entries_in_active_branch(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_hypothesis_store", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    tools = {tool.name: tool for tool in session.tools}

    await tools["add_hypothesis"].execute({"id": "H1", "description": "one"})
    await tools["add_hypothesis"].execute({"id": "H2", "description": "two"})
    await tools["add_hypothesis"].execute({"id": "H3", "description": "three"})
    listed = await tools["list_hypotheses"].execute({})

    assert isinstance(listed.content[0], TextContent)
    assert len(json.loads(listed.content[0].text)) == 3
    branch = session.session_manager.get_active_branch()
    assert len([entry for entry in branch if entry.type == "hypothesis"]) == 3
    await session.shutdown()
