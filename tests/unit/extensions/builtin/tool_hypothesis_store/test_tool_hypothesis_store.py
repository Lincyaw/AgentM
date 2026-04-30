from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_tool_hypothesis_store_install_smoke(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_hypothesis_store", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    assert [tool.name for tool in session.tools] == [
        "add_hypothesis",
        "update_hypothesis",
        "list_hypotheses",
    ]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_hypothesis_store_adds_and_lists_entries(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_hypothesis_store", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    tools = {tool.name: tool for tool in session.tools}

    added = await tools["add_hypothesis"].execute(
        {"id": "H1", "description": "cache issue", "evidence_summary": "error spike"}
    )
    listed = await tools["list_hypotheses"].execute({})

    assert not added.is_error
    payload = json.loads(listed.content[0].text)
    assert payload == [
        {
            "description": "cache issue",
            "evidence": ["error spike"],
            "id": "H1",
            "parent_id": None,
            "status": "formed",
        }
    ]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_hypothesis_store_returns_error_for_unknown_update(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_hypothesis_store", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    tools = {tool.name: tool for tool in session.tools}

    result = await tools["update_hypothesis"].execute({"id": "missing", "status": "confirmed"})

    assert result.is_error
    assert "Unknown hypothesis" in result.content[0].text
    await session.shutdown()
