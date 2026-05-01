from __future__ import annotations

from pathlib import Path

import pytest

from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_session_can_boot_from_registry_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[],
            provider="anthropic",
            model="claude-sonnet-4-6",
            resource_loader=InMemoryResourceLoader(),
        )
    )
    try:
        assert session.model is not None
        assert session.model.provider == "anthropic"
        assert session.model.id == "claude-sonnet-4-6"
    finally:
        await session.shutdown()
