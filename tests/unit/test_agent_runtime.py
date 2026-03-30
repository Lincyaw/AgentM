"""Tests for AgentRuntime — lifecycle, messaging, coordination.

RED phase: all tests should FAIL until AgentRuntime is implemented.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentm.harness.types import (
    AgentEvent,
    AgentResult,
    AgentStatus,
    Message,
    RunConfig,
)
from agentm.harness.runtime import AgentRuntime


# ---------------------------------------------------------------------------
# FakeAgentLoop — controllable AgentLoop for testing
# ---------------------------------------------------------------------------


class FakeAgentLoop:
    """Controllable AgentLoop implementation for testing.

    Emits a configurable number of llm_end events, then a complete event
    carrying an AgentResult with the given output.
    """

    def __init__(
        self,
        steps: int = 1,
        result_output: str = "done",
        delay: float = 0,
    ) -> None:
        self._steps = steps
        self._result_output = result_output
        self._delay = delay
        self._inbox: list[str] = []
        self._cancelled = False

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AgentResult:
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                return event.data.get("result")  # type: ignore[return-value]
        raise RuntimeError("stream ended without complete event")

    async def stream(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> Any:
        config = config or RunConfig()
        agent_id = config.metadata.get("agent_id", "")
        for step in range(self._steps):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield AgentEvent(type="llm_end", agent_id=agent_id, step=step)
        result = AgentResult(
            agent_id=agent_id,
            status=AgentStatus.COMPLETED,
            output=self._result_output,
            steps=self._steps,
        )
        yield AgentEvent(
            type="complete", agent_id=agent_id, data={"result": result}
        )


class NeverEndingLoop:
    """A loop that never completes — hangs until cancelled."""

    def __init__(self) -> None:
        self._inbox: list[str] = []

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AgentResult:
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                return event.data.get("result")  # type: ignore[return-value]
        raise RuntimeError("stream ended without complete event")

    async def stream(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> Any:
        config = config or RunConfig()
        agent_id = config.metadata.get("agent_id", "")
        # Emit one event then hang forever
        yield AgentEvent(type="llm_start", agent_id=agent_id, step=0)
        try:
            await asyncio.sleep(3600)  # effectively forever
        except asyncio.CancelledError:
            return


class MockEventHandler:
    """Collects all events for assertion."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentRuntimeSpawnAndWait:
    """Basic lifecycle: spawn an agent, wait for completion."""

    @pytest.mark.asyncio
    async def test_should_return_completed_result_when_agent_finishes(self) -> None:
        """spawn -> wait -> get COMPLETED result with expected output."""
        runtime = AgentRuntime()
        loop = FakeAgentLoop(steps=2, result_output="analysis complete")

        handle = await runtime.spawn(
            "agent-1", loop=loop, input="analyze this"
        )
        result = await handle.wait(timeout=5.0)

        assert result.status == AgentStatus.COMPLETED
        assert result.output == "analysis complete"
        assert result.agent_id == "agent-1"


class TestAgentRuntimeSpawnMultiple:
    """Multiple agents can run concurrently."""

    @pytest.mark.asyncio
    async def test_should_complete_all_agents_and_track_status(self) -> None:
        """spawn 3 agents -> all complete -> all appear in get_status()."""
        runtime = AgentRuntime()

        handles = []
        for i in range(3):
            h = await runtime.spawn(
                f"worker-{i}",
                loop=FakeAgentLoop(steps=1, result_output=f"result-{i}"),
                input=f"task {i}",
            )
            handles.append(h)

        # Wait for all
        for h in handles:
            await h.wait(timeout=5.0)

        status = runtime.get_status()
        assert len(status) == 3
        for i in range(3):
            assert status[f"worker-{i}"].status == AgentStatus.COMPLETED


class TestAgentRuntimeAbort:
    """Aborting a running agent."""

    @pytest.mark.asyncio
    async def test_should_abort_running_agent(self) -> None:
        """spawn slow agent -> abort -> status becomes ABORTED."""
        runtime = AgentRuntime()
        loop = NeverEndingLoop()

        await runtime.spawn("slow-agent", loop=loop, input="work")
        # Give the task a moment to start
        await asyncio.sleep(0.05)

        aborted = await runtime.abort("slow-agent", reason="test abort")
        assert aborted is True

        result = runtime.get_result("slow-agent")
        assert result is not None
        assert result.status == AgentStatus.ABORTED


class TestAgentRuntimeSendMessage:
    """Message injection into a running agent."""

    @pytest.mark.asyncio
    async def test_should_inject_message_into_agent_loop(self) -> None:
        """spawn -> send("hello") -> loop.inject was called."""
        runtime = AgentRuntime()
        loop = FakeAgentLoop(steps=1, result_output="done", delay=0.1)

        await runtime.spawn("msg-agent", loop=loop, input="start")
        await runtime.send("msg-agent", "hello")

        assert "hello" in loop._inbox


class TestAgentRuntimeWaitAny:
    """wait_any returns the first agent(s) to complete."""

    @pytest.mark.asyncio
    async def test_should_return_first_completed_agent(self) -> None:
        """spawn fast(0.01s) + slow(10s) -> wait_any -> fast one returned."""
        runtime = AgentRuntime()

        await runtime.spawn(
            "fast",
            loop=FakeAgentLoop(steps=1, result_output="quick", delay=0.01),
            input="go",
        )
        await runtime.spawn(
            "slow",
            loop=NeverEndingLoop(),
            input="go",
        )

        completed = await runtime.wait_any(timeout=5.0)
        assert "fast" in completed

        # Cleanup
        await runtime.abort("slow", reason="cleanup")

    @pytest.mark.asyncio
    async def test_should_only_monitor_specified_agents(self) -> None:
        """spawn a,b,c -> wait_any([a,b]) -> only those monitored."""
        runtime = AgentRuntime()

        await runtime.spawn(
            "a",
            loop=FakeAgentLoop(steps=1, result_output="a-done", delay=0.05),
            input="go",
        )
        await runtime.spawn(
            "b",
            loop=NeverEndingLoop(),
            input="go",
        )
        await runtime.spawn(
            "c",
            loop=FakeAgentLoop(steps=1, result_output="c-done", delay=0.01),
            input="go",
        )

        # Only monitor a and b — c finishes first but should not be reported
        completed = await runtime.wait_any(agent_ids=["a", "b"], timeout=5.0)
        assert "a" in completed
        assert "c" not in completed

        # Cleanup
        await runtime.abort("b", reason="cleanup")


class TestAgentRuntimeParentChild:
    """Parent-child cascade behavior."""

    @pytest.mark.asyncio
    async def test_should_cascade_abort_to_children(self) -> None:
        """spawn parent + child(parent_id=parent) -> abort parent -> child also ABORTED."""
        runtime = AgentRuntime()

        await runtime.spawn(
            "parent", loop=NeverEndingLoop(), input="orchestrate"
        )
        await runtime.spawn(
            "child",
            loop=NeverEndingLoop(),
            input="work",
            parent_id="parent",
        )
        await asyncio.sleep(0.05)

        await runtime.abort("parent", reason="cascade test")

        child_result = runtime.get_result("child")
        assert child_result is not None
        assert child_result.status == AgentStatus.ABORTED

        parent_result = runtime.get_result("parent")
        assert parent_result is not None
        assert parent_result.status == AgentStatus.ABORTED

    @pytest.mark.asyncio
    async def test_should_abort_children_when_parent_completes(self) -> None:
        """parent completes normally -> running children auto-aborted."""
        runtime = AgentRuntime()

        # Parent finishes quickly
        await runtime.spawn(
            "parent",
            loop=FakeAgentLoop(steps=1, result_output="parent-done", delay=0.02),
            input="orchestrate",
        )
        # Child is long-running
        await runtime.spawn(
            "child",
            loop=NeverEndingLoop(),
            input="work",
            parent_id="parent",
        )

        # Wait for parent to finish
        parent_result = await runtime.wait("parent", timeout=5.0)
        assert parent_result.status == AgentStatus.COMPLETED

        # Give cascade a moment to propagate
        await asyncio.sleep(0.1)

        child_result = runtime.get_result("child")
        assert child_result is not None
        assert child_result.status == AgentStatus.ABORTED


class TestAgentRuntimeGetResult:
    """get_result behavior for completed vs running agents."""

    @pytest.mark.asyncio
    async def test_should_return_result_for_completed_agent(self) -> None:
        """Completed agent -> returns AgentResult."""
        runtime = AgentRuntime()
        handle = await runtime.spawn(
            "done-agent",
            loop=FakeAgentLoop(steps=1, result_output="finished"),
            input="go",
        )
        await handle.wait(timeout=5.0)

        result = runtime.get_result("done-agent")
        assert result is not None
        assert result.status == AgentStatus.COMPLETED
        assert result.output == "finished"

    @pytest.mark.asyncio
    async def test_should_return_none_for_running_agent(self) -> None:
        """Running agent -> returns None."""
        runtime = AgentRuntime()
        await runtime.spawn(
            "running-agent", loop=NeverEndingLoop(), input="go"
        )
        await asyncio.sleep(0.05)

        result = runtime.get_result("running-agent")
        assert result is None

        # Cleanup
        await runtime.abort("running-agent", reason="cleanup")


class TestAgentRuntimeGetRunningIds:
    """get_running_ids filters to only RUNNING agents."""

    @pytest.mark.asyncio
    async def test_should_return_only_running_agent_ids(self) -> None:
        """Mix of running and completed -> only running IDs returned."""
        runtime = AgentRuntime()

        # This one will complete immediately
        handle = await runtime.spawn(
            "done",
            loop=FakeAgentLoop(steps=1, result_output="ok"),
            input="go",
        )
        await handle.wait(timeout=5.0)

        # This one stays running
        await runtime.spawn(
            "active", loop=NeverEndingLoop(), input="go"
        )
        await asyncio.sleep(0.05)

        running = runtime.get_running_ids()
        assert "active" in running
        assert "done" not in running

        # Cleanup
        await runtime.abort("active", reason="cleanup")


class TestAgentRuntimeEventHandler:
    """EventHandler receives forwarded events."""

    @pytest.mark.asyncio
    async def test_should_forward_events_to_handler(self) -> None:
        """spawn with MockEventHandler -> events are forwarded."""
        handler = MockEventHandler()
        runtime = AgentRuntime(event_handler=handler)

        handle = await runtime.spawn(
            "observed",
            loop=FakeAgentLoop(steps=2, result_output="done"),
            input="go",
        )
        await handle.wait(timeout=5.0)

        # Should have received llm_end events and a complete event
        event_types = [e.type for e in handler.events]
        assert "llm_end" in event_types
        assert "complete" in event_types
        # All events should be for our agent
        assert all(e.agent_id == "observed" for e in handler.events)


class TestAgentRuntimeWaitTimeout:
    """wait() raises TimeoutError when agent doesn't finish in time."""

    @pytest.mark.asyncio
    async def test_should_raise_timeout_error(self) -> None:
        """spawn never-ending agent -> wait(timeout=0.1) -> TimeoutError."""
        runtime = AgentRuntime()
        handle = await runtime.spawn(
            "forever", loop=NeverEndingLoop(), input="go"
        )

        with pytest.raises(TimeoutError):
            await handle.wait(timeout=0.1)

        # Cleanup
        await runtime.abort("forever", reason="cleanup")
