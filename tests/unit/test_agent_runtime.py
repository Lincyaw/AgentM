"""Focused regression tests for AgentRuntime lifecycle and coordination."""
from __future__ import annotations

import asyncio

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.harness.types import AgentStatus

from tests.helpers import FakeAgentLoop, MockEventHandler, NeverEndingLoop


@pytest.mark.asyncio
async def test_spawn_and_wait_returns_completed_result() -> None:
    runtime = AgentRuntime()
    handle = await runtime.spawn("agent-1", loop=FakeAgentLoop(steps=2, result_output="done"), input="task")
    result = await handle.wait(timeout=5.0)
    assert result.status == AgentStatus.COMPLETED
    assert result.output == "done"
    assert result.agent_id == "agent-1"


@pytest.mark.asyncio
async def test_abort_transitions_running_agent_to_aborted() -> None:
    runtime = AgentRuntime()
    await runtime.spawn("slow-agent", loop=NeverEndingLoop(), input="work")
    await asyncio.sleep(0.05)
    assert await runtime.abort("slow-agent", reason="test") is True
    result = runtime.get_result("slow-agent")
    assert result is not None
    assert result.status == AgentStatus.ABORTED


@pytest.mark.asyncio
async def test_wait_any_can_scope_to_specific_agent_ids() -> None:
    runtime = AgentRuntime()
    await runtime.spawn("a", loop=FakeAgentLoop(steps=1, result_output="a", delay=0.05), input="go")
    await runtime.spawn("b", loop=NeverEndingLoop(), input="go")
    await runtime.spawn("c", loop=FakeAgentLoop(steps=1, result_output="c", delay=0.01), input="go")

    completed = await runtime.wait_any(agent_ids=["a", "b"], timeout=5.0)
    assert "a" in completed
    assert "c" not in completed
    await runtime.abort("b", reason="cleanup")


@pytest.mark.asyncio
async def test_parent_completion_aborts_running_children() -> None:
    runtime = AgentRuntime()
    await runtime.spawn("parent", loop=FakeAgentLoop(steps=1, result_output="ok", delay=0.02), input="run")
    await runtime.spawn("child", loop=NeverEndingLoop(), input="work", parent_id="parent")

    parent = await runtime.wait("parent", timeout=5.0)
    assert parent.status == AgentStatus.COMPLETED
    await asyncio.sleep(0.1)

    child = runtime.get_result("child")
    assert child is not None
    assert child.status == AgentStatus.ABORTED


@pytest.mark.asyncio
async def test_send_injects_message_into_loop_inbox() -> None:
    runtime = AgentRuntime()
    loop = FakeAgentLoop(steps=1, result_output="done", delay=0.1)
    await runtime.spawn("msg-agent", loop=loop, input="start")
    await runtime.send("msg-agent", "hello")
    assert "hello" in loop._inbox


@pytest.mark.asyncio
async def test_wait_timeout_raises_timeout_error() -> None:
    runtime = AgentRuntime()
    handle = await runtime.spawn("forever", loop=NeverEndingLoop(), input="go")
    with pytest.raises(TimeoutError):
        await handle.wait(timeout=0.1)
    await runtime.abort("forever", reason="cleanup")


@pytest.mark.asyncio
async def test_runtime_forwards_events_to_handler() -> None:
    handler = MockEventHandler()
    runtime = AgentRuntime(event_handler=handler)
    handle = await runtime.spawn("observed", loop=FakeAgentLoop(steps=2, result_output="done"), input="go")
    await handle.wait(timeout=5.0)
    event_types = [e.type for e in handler.events]
    assert "llm_end" in event_types
    assert "complete" in event_types
    assert all(e.agent_id == "observed" for e in handler.events)
