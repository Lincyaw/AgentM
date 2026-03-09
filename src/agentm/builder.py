"""AgentSystem and AgentSystemBuilder — unified entry point for all agent systems."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from agentm.agents.orchestrator import create_orchestrator
from agentm.agents.sub_agent import AgentPool
from agentm.config.schema import ScenarioConfig, StorageConfig
from agentm.core.state_registry import get_state_schema
from agentm.core.task_manager import TaskManager
from agentm.core.tool_registry import ToolRegistry
from agentm.core.trajectory import TrajectoryCollector
from agentm.tools.orchestrator import create_orchestrator_tools


class AgentSystem:
    """Unified interface for all agent systems."""

    def __init__(
        self,
        graph: Any,
        config: dict[str, Any] | None = None,
        scenario_config: ScenarioConfig | None = None,
        task_manager: TaskManager | None = None,
        trajectory: TrajectoryCollector | None = None,
        thread_id: str = "",
        _pending_storage: StorageConfig | None = None,
    ) -> None:
        self.graph = graph
        self.langgraph_config = config or {}
        self.scenario_config = scenario_config
        self.task_manager = task_manager
        self.trajectory = trajectory
        self.thread_id = thread_id
        self._pending_storage = _pending_storage
        self._initialized = _pending_storage is None

    async def __aenter__(self) -> AgentSystem:
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        if self.trajectory is not None:
            await self.trajectory.close()

    async def _ensure_checkpointer(self) -> None:
        """Lazily create the async checkpointer on first use."""
        if self._initialized:
            return
        self._initialized = True
        if self._pending_storage is not None:
            checkpointer = await _create_async_checkpointer(self._pending_storage)
            if checkpointer is not None:
                # Patch the compiled graph's checkpointer
                self.graph.checkpointer = checkpointer
            self._pending_storage = None

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent system with the given input. Returns final state."""
        await self._ensure_checkpointer()
        return await self.graph.ainvoke(input_data, config=self.langgraph_config)

    async def stream(
        self,
        input_data: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events from the agent system execution.

        Events are recorded to the TrajectoryCollector, which notifies all
        registered listeners (DebugConsole, WebSocket broadcast, etc.).
        """
        await self._ensure_checkpointer()
        step = 0
        async for event in self.graph.astream(input_data, config=self.langgraph_config):
            step += 1

            # Record to trajectory — listeners are notified automatically
            for node_name, node_data in event.items():
                if node_name == "__interrupt__":
                    continue
                if not isinstance(node_data, dict):
                    continue
                await self._record_node_event(node_name, node_data, step)

            yield event

    async def _record_node_event(
        self, node_name: str, node_data: dict[str, Any], step: int
    ) -> None:
        """Parse orchestrator node output and record structured events."""
        if self.trajectory is None:
            return

        messages = node_data.get("messages", [])
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")

            if role == "ai":
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        await self.trajectory.record(
                            event_type="tool_call",
                            agent_path=["orchestrator"],
                            node_name=node_name,
                            data={
                                "tool_name": tc["name"],
                                "args": tc.get("args", {}),
                                "step": step,
                            },
                        )
                elif content:
                    await self.trajectory.record(
                        event_type="llm_end",
                        agent_path=["orchestrator"],
                        node_name=node_name,
                        data={
                            "content_preview": content[:500],
                            "content_length": len(content),
                            "step": step,
                        },
                    )

            elif role == "tool":
                tool_name = getattr(msg, "name", "unknown")
                await self.trajectory.record(
                    event_type="tool_result",
                    agent_path=["orchestrator"],
                    node_name=node_name,
                    data={
                        "tool_name": tool_name,
                        "result_preview": content[:500] if content else "",
                        "result_length": len(content) if content else 0,
                        "step": step,
                    },
                )


def _create_checkpointer(storage_config: StorageConfig) -> Any:
    """Create a LangGraph checkpointer from storage config.

    For 'memory' backend, returns a MemorySaver directly.
    For 'sqlite' backend, returns None here — the AsyncSqliteSaver must be
    created inside an async context. Use _create_async_checkpointer() instead.
    """
    backend = storage_config.checkpointer.backend

    if backend == "memory":
        return MemorySaver()

    # sqlite and others handled by _create_async_checkpointer
    return None


async def _create_async_checkpointer(storage_config: StorageConfig) -> Any:
    """Create an async checkpointer. Must be called from within an event loop."""
    backend = storage_config.checkpointer.backend

    if backend == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: I001

        url = storage_config.checkpointer.url or "./checkpoints.db"
        conn = aiosqlite.connect(url)
        saver = AsyncSqliteSaver(conn=conn)
        await saver.setup()
        return saver

    return None


TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "tools"


class AgentSystemBuilder:
    """Unified entry point for building any agent system.

    Internally selects the appropriate architecture based on system_type:
    - ReAct-based (create_react_agent): For exploratory, non-linear scenarios like RCA
    - StateGraph-based (custom graph with phase nodes): For linear, deterministic scenarios
    """

    @staticmethod
    def build(
        system_type: str,
        scenario_config: ScenarioConfig,
        system_config: Any | None = None,
    ) -> AgentSystem:
        """Build an AgentSystem from a system type and scenario config."""
        get_state_schema(system_type)  # validate system_type exists

        if scenario_config.orchestrator.orchestrator_mode == "react":
            tool_registry = ToolRegistry()

            # Load all tool YAMLs
            for yaml_file in sorted(TOOLS_DIR.glob("*.yaml")):
                tool_registry.load_from_yaml(yaml_file)

            # Resolve model config for the orchestrator model
            orch_model_name = scenario_config.orchestrator.model
            orch_model_config = (
                system_config.models.get(orch_model_name)
                if system_config is not None
                else None
            )

            # Resolve model config for workers
            worker_config = scenario_config.agents.get("worker")
            worker_model_config = None
            if worker_config is not None and system_config is not None:
                worker_model_config = system_config.models.get(worker_config.model)

            agent_pool = AgentPool(scenario_config, tool_registry, worker_model_config)
            task_manager = TaskManager()

            # --- Checkpointer ---
            checkpointer = None
            pending_storage: StorageConfig | None = None
            if system_config is not None:
                checkpointer = _create_checkpointer(system_config.storage)
                if checkpointer is None and system_config.storage.checkpointer.backend != "memory":
                    pending_storage = system_config.storage

            thread_id = str(uuid.uuid4())
            langgraph_config: dict[str, Any] = {
                "configurable": {"thread_id": thread_id},
            }

            # --- Trajectory ---
            trajectory: TrajectoryCollector | None = None
            if system_config is not None and system_config.debug.trajectory.enabled:
                run_id = f"rca-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                trajectory = TrajectoryCollector(
                    run_id=run_id,
                    output_dir=system_config.debug.trajectory.output_dir,
                )

            # --- Dependency injection via closure factory ---
            injected_tools = create_orchestrator_tools(task_manager, agent_pool, trajectory=trajectory)

            # Wire trajectory into task_manager
            task_manager.set_trajectory(trajectory)

            # Build orchestrator tools: YAML-registered + factory-injected
            tools: list[Any] = []

            for name in scenario_config.orchestrator.tools:
                if name in injected_tools:
                    # Factory-created tool (has closure over task_manager/agent_pool)
                    func = injected_tools[name]
                    if asyncio.iscoroutinefunction(func):
                        tool = StructuredTool.from_function(
                            coroutine=func, name=name, description=func.__doc__ or name,
                        )
                    else:
                        tool = StructuredTool.from_function(
                            func=func, name=name, description=func.__doc__ or name,
                        )
                    tools.append(tool)
                elif tool_registry.has(name):
                    # YAML-registered tool (module-level function)
                    tools.append(tool_registry.get(name).create_with_config())
                else:
                    raise ValueError(f"Tool {name!r} not found in registry or factory")

            graph = create_orchestrator(
                config=scenario_config.orchestrator,
                tools=tools,
                checkpointer=checkpointer,
                store=None,
                model_config=orch_model_config,
            )
            return AgentSystem(
                graph,
                config=langgraph_config,
                scenario_config=scenario_config,
                task_manager=task_manager,
                trajectory=trajectory,
                thread_id=thread_id,
                _pending_storage=pending_storage,
            )

        raise NotImplementedError(
            f"Orchestrator mode {scenario_config.orchestrator.orchestrator_mode!r} not yet implemented"
        )
