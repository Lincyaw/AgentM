"""Integration tests for the migrated harness system.

Verifies that the wiring between WorkerLoopFactory, orchestrator tools,
AgentRuntime, and builder.py is correct.  Does NOT re-test SimpleAgentLoop
or AgentRuntime internals (those are covered in their own test files).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentm.harness.protocols import AgentLoop
from agentm.harness.runtime import AgentRuntime
from agentm.harness.types import (
    AgentEvent,
    AgentResult,
    AgentStatus,
    Message,
    RunConfig,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


@dataclass
class MockAIResponse:
    """Minimal mock of an LLM AI response."""

    content: str = "done"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    type: str = "ai"


class MockModel:
    """Mock ChatModel that returns pre-configured responses in sequence."""

    def __init__(self, responses: list[MockAIResponse] | None = None) -> None:
        self._responses = iter(responses or [MockAIResponse()])
        self.invocations: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> MockAIResponse:
        self.invocations.append(messages)
        return next(self._responses)

    def bind_tools(self, tools: list[Any]) -> MockModel:
        return self

    def with_structured_output(self, schema: type) -> MockModel:
        return self


class MockTool:
    """Mock tool with a fixed result."""

    def __init__(self, name: str, result: str = "tool result") -> None:
        self.name = name
        self.description = f"Mock {name} tool"
        self._result = result

    async def ainvoke(self, args: dict[str, Any]) -> str:
        return self._result

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }


class FakeAgentLoop:
    """Controllable AgentLoop for testing orchestrator tool wiring."""

    def __init__(self, result_output: Any = "worker-result") -> None:
        self._result_output = result_output
        self._inbox: list[str] = []

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AgentResult:
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                return event.data["result"]
        raise RuntimeError("no complete event")

    async def stream(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> Any:
        config = config or RunConfig()
        agent_id = config.metadata.get("agent_id", "")
        result = AgentResult(
            agent_id=agent_id,
            status=AgentStatus.COMPLETED,
            output=self._result_output,
            steps=1,
        )
        yield AgentEvent(
            type="complete", agent_id=agent_id, data={"result": result}
        )


class FakeWorkerFactory:
    """Mock WorkerFactory for orchestrator tool tests."""

    def __init__(self, loop: FakeAgentLoop | None = None) -> None:
        self._loop = loop or FakeAgentLoop()
        self.create_calls: list[tuple[str, str]] = []

    def create_worker(self, agent_id: str, task_type: str) -> AgentLoop:
        self.create_calls.append((agent_id, task_type))
        return self._loop


# =========================================================================
# Test 1: WorkerLoopFactory creates valid SimpleAgentLoop
# =========================================================================


class TestWorkerLoopFactoryCreatesValidLoop:
    """WorkerLoopFactory.create_worker produces a SimpleAgentLoop that
    satisfies the AgentLoop protocol."""

    @patch("agentm.harness.worker_factory.create_chat_model")
    def test_factory_returns_agent_loop(
        self, mock_create_model: MagicMock
    ) -> None:
        """Given a minimal ScenarioConfig, factory produces a loop that
        satisfies AgentLoop protocol."""
        from agentm.config.schema import (
            AgentConfig,
            ExecutionConfig,
            OrchestratorConfig,
            ScenarioConfig,
            SystemTypeConfig,
        )
        from agentm.harness.worker_factory import WorkerLoopFactory

        mock_model = MockModel()
        mock_create_model.return_value = mock_model

        tool_registry = MagicMock()
        mock_tool_def = MagicMock()
        mock_tool_def.create_tool.return_value = MockTool("query_logs")
        tool_registry.get.return_value = mock_tool_def

        scenario_config = ScenarioConfig(
            system=SystemTypeConfig(type="test"),
            orchestrator=OrchestratorConfig(model="gpt-4o", temperature=0.7, tools=[]),
            agents={
                "worker": AgentConfig(
                    model="gpt-4o",
                    temperature=0.0,
                    tools=["query_logs"],
                    execution=ExecutionConfig(max_steps=10),
                )
            },
        )

        factory = WorkerLoopFactory(scenario_config, tool_registry)
        loop = factory.create_worker("w1", "scout")

        # Verify it satisfies the AgentLoop protocol
        assert isinstance(loop, AgentLoop)
        assert hasattr(loop, "run")
        assert hasattr(loop, "stream")
        assert hasattr(loop, "inject")

    @patch("agentm.harness.worker_factory.create_chat_model")
    def test_factory_wires_middleware(
        self, mock_create_model: MagicMock
    ) -> None:
        """Factory includes BudgetMiddleware and LoopDetectionMiddleware
        in the middleware list."""
        from agentm.config.schema import (
            AgentConfig,
            ExecutionConfig,
            OrchestratorConfig,
            ScenarioConfig,
            SystemTypeConfig,
        )
        from agentm.harness.middleware import BudgetMiddleware, LoopDetectionMiddleware
        from agentm.harness.worker_factory import WorkerLoopFactory

        mock_create_model.return_value = MockModel()

        tool_registry = MagicMock()
        mock_tool_def = MagicMock()
        mock_tool_def.create_tool.return_value = MockTool("t1")
        tool_registry.get.return_value = mock_tool_def

        scenario_config = ScenarioConfig(
            system=SystemTypeConfig(type="test"),
            orchestrator=OrchestratorConfig(model="gpt-4o", temperature=0.7, tools=[]),
            agents={
                "worker": AgentConfig(
                    model="gpt-4o",
                    temperature=0.0,
                    tools=["t1"],
                    execution=ExecutionConfig(max_steps=15),
                )
            },
        )

        factory = WorkerLoopFactory(scenario_config, tool_registry)
        loop = factory.create_worker("w1", "scout")

        # Check middleware was wired
        mw_types = [type(m) for m in loop._middleware]
        assert BudgetMiddleware in mw_types
        assert LoopDetectionMiddleware in mw_types


# =========================================================================
# Test 2: dispatch_agent calls runtime.spawn
# =========================================================================


class TestDispatchAgentUsesRuntimeSpawn:
    """dispatch_agent tool calls runtime.spawn with the correct arguments."""

    @pytest.mark.asyncio
    async def test_dispatch_calls_spawn(self) -> None:
        """Mock runtime, call dispatch_agent -> verify runtime.spawn was called."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()
        factory = FakeWorkerFactory()
        tools = create_orchestrator_tools(runtime, factory)

        await tools["dispatch_agent"](
            agent_id="worker-1",
            task="Investigate logs",
            task_type="scout",
        )

        # Factory should have been called to create a worker
        assert len(factory.create_calls) == 1
        assert factory.create_calls[0][0] == "worker-1"
        assert factory.create_calls[0][1] == "scout"

        # Runtime should have spawned the agent (check via status)
        status = runtime.get_status()
        assert len(status) == 1
        # The spawned agent has a unique ID like "worker-1-<hash>"
        spawned_id = list(status.keys())[0]
        assert spawned_id.startswith("worker-1-")

    @pytest.mark.asyncio
    async def test_dispatch_auto_blocks_for_single_worker(self) -> None:
        """When only one worker is running, dispatch_agent waits and
        returns the result directly."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()
        factory = FakeWorkerFactory(FakeAgentLoop(result_output="findings-here"))
        tools = create_orchestrator_tools(runtime, factory)

        result = await tools["dispatch_agent"](
            agent_id="w1",
            task="Do analysis",
            task_type="scout",
        )

        # Auto-block: result contains the worker's output as a JSON string
        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert parsed["result"] is not None


# =========================================================================
# Test 3: check_tasks uses runtime.get_status
# =========================================================================


class TestCheckTasksUsesRuntimeGetStatus:
    """check_tasks reads runtime status and formats output."""

    @pytest.mark.asyncio
    async def test_check_tasks_returns_completed_results(self) -> None:
        """Spawn a worker that completes, then call check_tasks ->
        completed list has the result."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()
        loop = FakeAgentLoop(result_output="scout-findings")
        factory = FakeWorkerFactory(loop)
        tools = create_orchestrator_tools(runtime, factory)

        # Dispatch a worker and let it complete
        await tools["dispatch_agent"](
            agent_id="scout-1",
            task="Scout services",
            task_type="scout",
        )

        # Now check tasks
        result = await tools["check_tasks"](
            request="waiting for results",
        )

        parsed = json.loads(result)

        assert parsed["completed_count"] >= 1
        assert isinstance(parsed["completed"], list)
        assert isinstance(parsed["running"], list)
        assert isinstance(parsed["failed"], list)


# =========================================================================
# Test 4: inject_instruction uses runtime.send
# =========================================================================


class TestInjectInstructionUsesRuntimeSend:
    """inject_instruction delegates to runtime.send."""

    @pytest.mark.asyncio
    async def test_inject_calls_runtime_send(self) -> None:
        """Mock runtime, call inject_instruction -> verify runtime.send called."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()
        loop = FakeAgentLoop()
        # Spawn an agent so it exists in runtime
        await runtime.spawn("test-task", loop=loop, input="go")

        factory = FakeWorkerFactory()
        tools = create_orchestrator_tools(runtime, factory)

        result = await tools["inject_instruction"](
            task_id="test-task", instruction="Focus on service X"
        )

        assert "injected" in result.lower()
        assert "Focus on service X" in loop._inbox

    @pytest.mark.asyncio
    async def test_inject_returns_error_for_missing_task(self) -> None:
        """inject_instruction for non-existent task returns error message."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()
        factory = FakeWorkerFactory()
        tools = create_orchestrator_tools(runtime, factory)

        result = await tools["inject_instruction"](
            task_id="no-such-task", instruction="hello"
        )

        assert "not found" in result.lower()


# =========================================================================
# Test 5: abort_task uses runtime.abort
# =========================================================================


class TestAbortTaskUsesRuntimeAbort:
    """abort_task delegates to runtime.abort."""

    @pytest.mark.asyncio
    async def test_abort_calls_runtime_abort(self) -> None:
        """Spawn a long-running agent, call abort_task -> verify aborted."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()

        # Use a loop that stays running long enough to abort
        class SlowLoop:
            def __init__(self) -> None:
                self._inbox: list[str] = []

            def inject(self, message: str) -> None:
                self._inbox.append(message)

            async def run(self, input: str | list[Message], *, config: RunConfig | None = None) -> AgentResult:
                async for event in self.stream(input, config=config):
                    if event.type == "complete":
                        return event.data["result"]
                raise RuntimeError("no complete")

            async def stream(self, input: str | list[Message], *, config: RunConfig | None = None) -> Any:
                config = config or RunConfig()
                agent_id = config.metadata.get("agent_id", "")
                yield AgentEvent(type="llm_start", agent_id=agent_id, step=0)
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    return

        await runtime.spawn("slow-agent", loop=SlowLoop(), input="go")
        await asyncio.sleep(0.05)

        factory = FakeWorkerFactory()
        tools = create_orchestrator_tools(runtime, factory)

        result = await tools["abort_task"](
            task_id="slow-agent", reason="test abort"
        )

        assert "aborted" in result.lower()
        agent_result = runtime.get_result("slow-agent")
        assert agent_result is not None
        assert agent_result.status == AgentStatus.ABORTED

    @pytest.mark.asyncio
    async def test_abort_returns_error_for_missing_task(self) -> None:
        """abort_task for non-existent task returns error message."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()
        factory = FakeWorkerFactory()
        tools = create_orchestrator_tools(runtime, factory)

        result = await tools["abort_task"](
            task_id="no-such-task", reason="test"
        )

        assert "not found" in result.lower()


# =========================================================================
# Test 6: Builder produces AgentSystem with runtime
# =========================================================================


class TestBuilderProducesAgentSystemWithRuntime:
    """build_agent_system creates an AgentSystem with AgentRuntime."""

    def test_builder_creates_runtime(self) -> None:
        """build_agent_system produces system.runtime as AgentRuntime.

        We mock the heavy dependencies (LLM, vault, scenario, tools)
        at the level where builder.py imports them, and verify the
        AgentSystem is wired with an AgentRuntime instance.
        """
        from agentm.builder import AgentSystem, build_agent_system
        from agentm.config.schema import (
            OrchestratorConfig,
            ScenarioConfig,
            SystemTypeConfig,
        )
        from agentm.harness.scenario import ScenarioWiring

        scenario_config = ScenarioConfig(
            system=SystemTypeConfig(type="test"),
            orchestrator=OrchestratorConfig(
                model="gpt-4o",
                temperature=0.7,
                tools=["dispatch_agent"],
                orchestrator_mode="node",
            ),
            agents={},
        )

        # Mock Scenario that returns minimal wiring
        mock_scenario = MagicMock()
        mock_scenario.name = "test"
        mock_scenario.setup.return_value = ScenarioWiring()

        with (
            patch("agentm.builder.get_scenario", return_value=mock_scenario),
            patch("agentm.builder.create_orchestrator_tools") as mock_create_tools,
            patch("agentm.builder.create_chat_model") as mock_create_model,
            patch("agentm.builder.WorkerLoopFactory"),
            patch("agentm.builder.ToolRegistry") as mock_registry_cls,
            patch("agentm.tools.vault.MarkdownVault"),
            patch("agentm.tools.vault.create_vault_tools", return_value={}),
            patch("agentm.scenarios.discover"),
        ):
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_create_model.return_value = mock_model

            mock_create_tools.return_value = {
                "dispatch_agent": lambda: None,
            }

            # ToolRegistry mock: has() returns False for unknown tools
            mock_registry_instance = MagicMock()
            mock_registry_instance.has.return_value = False
            mock_registry_cls.return_value = mock_registry_instance

            system = build_agent_system("test", scenario_config)

            assert isinstance(system, AgentSystem)
            assert isinstance(system.runtime, AgentRuntime)


# =========================================================================
# Test 7: End-to-end dispatch + check cycle
# =========================================================================


class TestEndToEndDispatchCheckCycle:
    """Full wiring: runtime + factory with mock model -> dispatch -> check."""

    @pytest.mark.asyncio
    async def test_dispatch_then_check_returns_result(self) -> None:
        """Create runtime + factory -> dispatch_agent -> check_tasks ->
        verify result flows through the full pipeline."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()
        expected_output = {"findings": "Service X is down", "leads": []}
        factory = FakeWorkerFactory(FakeAgentLoop(result_output=expected_output))
        tools = create_orchestrator_tools(runtime, factory)

        # Step 1: Dispatch
        dispatch_result = await tools["dispatch_agent"](
            agent_id="investigator",
            task="Check service health",
            task_type="scout",
        )

        # The dispatch auto-blocked (single worker), so result is already there
        dispatch_content = json.loads(dispatch_result)
        assert dispatch_content["status"] == "completed"

        # Step 2: Check tasks
        check_result = await tools["check_tasks"](
            request="what happened?",
        )
        check_content = json.loads(check_result)
        assert check_content["completed_count"] >= 1

        # The completed entry should carry the worker's output
        completed_entries = check_content["completed"]
        assert len(completed_entries) >= 1
        found = any(
            entry.get("result") == expected_output
            for entry in completed_entries
        )
        assert found, f"Expected output not found in completed entries: {completed_entries}"

    @pytest.mark.asyncio
    async def test_multiple_dispatches_then_check(self) -> None:
        """Dispatch two workers -> both complete -> check_tasks sees both."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        runtime = AgentRuntime()
        factory = FakeWorkerFactory(FakeAgentLoop(result_output="result-data"))
        tools = create_orchestrator_tools(runtime, factory)

        # Dispatch two workers in sequence (each auto-blocks since it's the only one running)
        await tools["dispatch_agent"](
            agent_id="w1", task="task-1", task_type="scout",
        )
        await tools["dispatch_agent"](
            agent_id="w2", task="task-2", task_type="verify",
        )

        # Check tasks should show both completed
        check_result = await tools["check_tasks"](
            request="results",
        )
        check_content = json.loads(check_result)

        assert check_content["completed_count"] >= 2
