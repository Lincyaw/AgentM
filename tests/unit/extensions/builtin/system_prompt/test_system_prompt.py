from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.extensions.builtin import _helpers
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_system_prompt_prepends_configured_text(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.system_prompt", {"prompt": "PREFIX"}),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    await session.prompt("hello")

    assert _helpers.LAST_STREAM is not None
    assert _helpers.LAST_STREAM.seen_systems[-1] == "PREFIX"
    await session.shutdown()


@pytest.mark.asyncio
async def test_system_prompt_stacks_in_declaration_order(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.system_prompt", {"prompt": "FIRST"}),
                ("agentm.extensions.builtin.system_prompt", {"prompt": "SECOND"}),
            ],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    await session.prompt("hello")

    assert _helpers.LAST_STREAM is not None
    assert _helpers.LAST_STREAM.seen_systems[-1] == "SECOND\n\nFIRST"
    await session.shutdown()
