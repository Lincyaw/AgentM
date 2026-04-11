"""Focused integration tests for runtime + orchestrator tool wiring."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentm.harness.runtime import AgentRuntime
from agentm.harness.types import AgentEvent, AgentResult, AgentStatus, Message, RunConfig

from tests.helpers import FakeAgentLoop, FakeWorkerFactory


@patch("agentm.harness.worker_factory.create_chat_model")
def test_worker_factory_creates_valid_loop_with_core_middleware(mock_create_model: MagicMock) -> None:
    from agentm.config.schema import AgentConfig, ExecutionConfig, OrchestratorConfig, ScenarioConfig, SystemTypeConfig
    from agentm.harness.middleware import BudgetMiddleware, LoopDetectionMiddleware
    from agentm.harness.worker_factory import WorkerLoopFactory
    from tests.helpers import MockModel, MockTool

    mock_create_model.return_value = MockModel()
    tool_registry = MagicMock()
    tool_def = MagicMock()
    tool_def.create_tool.return_value = MockTool("query_logs")
    tool_registry.get.return_value = tool_def

    cfg = ScenarioConfig(
        system=SystemTypeConfig(type="test"),
        orchestrator=OrchestratorConfig(model="gpt-4o", temperature=0.7, tools=[]),
        agents={"worker": AgentConfig(model="gpt-4o", temperature=0.0, tools=["query_logs"], execution=ExecutionConfig(max_steps=10))},
    )
    loop = WorkerLoopFactory(cfg, tool_registry).create_worker("w1", "scout")
    assert hasattr(loop, "run") and hasattr(loop, "stream") and hasattr(loop, "inject")
    mw_types = {type(m) for m in loop._middleware}
    assert BudgetMiddleware in mw_types
    assert LoopDetectionMiddleware in mw_types
    assert loop._output_schema is None


@pytest.mark.asyncio
async def test_dispatch_agent_calls_runtime_spawn_and_returns_completed_payload() -> None:
    from agentm.tools.orchestrator import create_orchestrator_tools

    runtime = AgentRuntime()
    worker_factory = FakeWorkerFactory(FakeAgentLoop(result_output="ok"))
    tools = create_orchestrator_tools(runtime, worker_factory)
    result = await tools["dispatch_agent"](agent_id="w1", task="Investigate", task_type="scout")
    parsed = json.loads(result)
    assert parsed["status"] == "completed"
    assert worker_factory.create_calls == [("w1", "scout", parsed["task_id"])]
    assert runtime.get_status()


@pytest.mark.asyncio
async def test_check_tasks_reports_completed_entries() -> None:
    from agentm.tools.orchestrator import create_orchestrator_tools

    runtime = AgentRuntime()
    tools = create_orchestrator_tools(runtime, FakeWorkerFactory(FakeAgentLoop(result_output="scout-findings")))
    await tools["dispatch_agent"](agent_id="scout-1", task="Scout", task_type="scout")
    parsed = json.loads(await tools["check_tasks"](request="status"))
    assert parsed["completed_count"] >= 1
    assert isinstance(parsed["completed"], list)


@pytest.mark.asyncio
async def test_inject_and_abort_tools_delegate_to_runtime() -> None:
    from agentm.tools.orchestrator import create_orchestrator_tools

    runtime = AgentRuntime()

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

    loop = SlowLoop()
    await runtime.spawn("slow-agent", loop=loop, input="go")
    await asyncio.sleep(0.05)

    tools = create_orchestrator_tools(runtime, FakeWorkerFactory())
    inject_result = await tools["inject_instruction"](task_id="slow-agent", instruction="focus X")
    assert "injected" in inject_result.lower()
    assert "focus X" in loop._inbox

    abort_result = await tools["abort_task"](task_id="slow-agent", reason="test abort")
    assert "aborted" in abort_result.lower()
    assert runtime.get_result("slow-agent").status == AgentStatus.ABORTED  # type: ignore[union-attr]


def test_builder_builds_agent_system_with_runtime() -> None:
    from agentm.builder import AgentSystem, build_agent_system
    from agentm.config.schema import OrchestratorConfig, ScenarioConfig, SystemTypeConfig
    from agentm.harness.scenario import ScenarioWiring

    cfg = ScenarioConfig(
        system=SystemTypeConfig(type="test"),
        orchestrator=OrchestratorConfig(model="gpt-4o", temperature=0.7, tools=["dispatch_agent"], orchestrator_mode="node"),
        agents={},
    )

    mock_scenario = MagicMock()
    mock_scenario.name = "test"
    mock_scenario.setup.return_value = ScenarioWiring()

    with (
        patch("agentm.builder.get_scenario", return_value=mock_scenario),
        patch("agentm.builder.create_orchestrator_tools", return_value={"dispatch_agent": lambda: None}),
        patch("agentm.builder.create_chat_model") as mock_create_model,
        patch("agentm.builder.WorkerLoopFactory"),
        patch("agentm.builder.ToolRegistry") as mock_registry_cls,
        patch("agentm.tools.vault.MarkdownVault"),
        patch("agentm.tools.vault.create_vault_tools", return_value={}),
        patch("agentm.scenarios.discover"),
    ):
        model = MagicMock()
        model.bind_tools.return_value = model
        mock_create_model.return_value = model
        registry = MagicMock()
        registry.has.return_value = False
        mock_registry_cls.return_value = registry

        system = build_agent_system("test", cfg)
        assert isinstance(system, AgentSystem)
        assert isinstance(system.runtime, AgentRuntime)
