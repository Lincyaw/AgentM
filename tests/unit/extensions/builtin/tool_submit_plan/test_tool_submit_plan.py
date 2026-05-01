from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest

from agentm.core.kernel import TextContent
from agentm.harness.events import PlanSubmittedEvent
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_tool_submit_plan_install_smoke(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_submit_plan", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    assert [tool.name for tool in session.tools] == ["submit_plan"]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_submit_plan_appends_entry_and_emits_event(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_submit_plan", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    events: list[PlanSubmittedEvent] = []
    session.bus.on("plan_submitted", lambda event: events.append(event))

    result = await session.tools[0].execute({"plan": "1. observe\n2. act"})

    assert not result.is_error
    assert result.extras == {"plan_submitted": True, "plan_id": events[0].plan_id}
    assert result.details == result.extras
    branch = session.session_manager.get_active_branch()
    assert branch[-1].type == "plan"
    assert branch[-1].payload == {"text": "1. observe\n2. act"}
    assert events == [PlanSubmittedEvent(plan_id=events[0].plan_id, plan_text="1. observe\n2. act")]
    await session.shutdown()


@pytest.mark.asyncio
async def test_tool_submit_plan_returns_error_when_session_append_fails(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.tool_submit_plan", {})],
            provider=("tests.unit.extensions.builtin._helpers", {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    original = session._extension_api._session.append_entry  # type: ignore[attr-defined]

    def _boom(type: str, payload: object, parent_id: str | None = None) -> str:
        del type, payload, parent_id
        raise RuntimeError("broken")

    session._extension_api._session.append_entry = MethodType(_boom, session._extension_api._session)  # type: ignore[method-assign]
    result = await session.tools[0].execute({"plan": "draft"})
    session._extension_api._session.append_entry = original  # type: ignore[method-assign]

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "broken" in result.content[0].text
    await session.shutdown()
