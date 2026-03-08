"""AgentSystem and AgentSystemBuilder — unified entry point for all agent systems."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from agentm.config.schema import ScenarioConfig, StorageConfig, SystemConfig
from agentm.core.state_registry import get_state_schema
from agentm.core.trajectory import TrajectoryCollector


class AgentSystem:
    """Unified interface for all agent systems."""

    def __init__(
        self,
        graph: Any,
        config: dict[str, Any] | None = None,
        scenario_config: Any | None = None,
        task_manager: Any | None = None,
        trajectory: TrajectoryCollector | None = None,
        thread_id: str = "",
    ) -> None:
        self.graph = graph
        self.langgraph_config = config or {}
        self.scenario_config = scenario_config
        self.task_manager = task_manager
        self.trajectory = trajectory
        self.thread_id = thread_id

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent system with the given input. Returns final state."""
        return await self.graph.ainvoke(input_data, config=self.langgraph_config)

    async def stream(
        self,
        input_data: dict[str, Any],
        on_event: Callable[..., Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events from the agent system execution.

        Args:
            input_data: Initial state for the agent system.
            on_event: Optional async callback invoked with each event dict.
        """
        step = 0
        async for event in self.graph.astream(input_data, config=self.langgraph_config):
            step += 1

            # Record to trajectory and notify callback
            for node_name, node_data in event.items():
                if node_name == "__interrupt__":
                    continue
                await self._record_node_event(node_name, node_data, step)
                if on_event is not None:
                    envelope = {
                        "agent_path": ["orchestrator"],
                        "node_name": node_name,
                        "mode": "updates",
                        "data": event,
                        "timestamp": datetime.now().isoformat(),
                        "step": step,
                    }
                    await on_event(envelope)

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
    """Create a LangGraph checkpointer from storage config."""
    backend = storage_config.checkpointer.backend

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if backend == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        url = storage_config.checkpointer.url or "./checkpoints.db"
        conn = aiosqlite.connect(url)
        return AsyncSqliteSaver(conn=conn)

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
        system_config: SystemConfig | None = None,
        scenario_dir: Path | str | None = None,
    ) -> AgentSystem:
        """Build an AgentSystem from a system type, scenario config, and system config.

        Args:
            system_type: The agent system type (e.g. "hypothesis_driven").
            scenario_config: Parsed scenario configuration.
            system_config: Parsed system configuration (optional, for model API keys/base_url).
            scenario_dir: Directory containing the scenario files. Used to resolve
                relative prompt paths in scenario config. If None, prompt paths are
                used as-is.
        """
        from agentm.agents.orchestrator import create_orchestrator
        from agentm.agents.sub_agent import AgentPool
        from agentm.core.task_manager import TaskManager
        from agentm.core.tool_registry import ToolRegistry

        get_state_schema(system_type)  # validate system_type exists

        # Resolve relative prompt paths against scenario_dir
        if scenario_dir is not None:
            scenario_dir = Path(scenario_dir)
            resolved_prompts = {
                k: str(scenario_dir / v)
                for k, v in scenario_config.orchestrator.prompts.items()
            }
            scenario_config.orchestrator.prompts = resolved_prompts

            for agent_config in scenario_config.agents.values():
                if agent_config.prompt is not None:
                    agent_config.prompt = str(scenario_dir / agent_config.prompt)
                if agent_config.task_type_prompts:
                    agent_config.task_type_prompts = {
                        k: str(scenario_dir / v)
                        for k, v in agent_config.task_type_prompts.items()
                    }
                if agent_config.compression and agent_config.compression.prompt:
                    agent_config.compression.prompt = str(
                        scenario_dir / agent_config.compression.prompt
                    )

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
            if system_config is not None:
                checkpointer = _create_checkpointer(system_config.storage)

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

            # Wire module-level references for orchestrator tools
            import agentm.tools.orchestrator as orch_tools
            orch_tools._task_manager = task_manager
            orch_tools._agent_pool = agent_pool
            orch_tools._trajectory = trajectory

            # Wire trajectory into task_manager
            task_manager._trajectory = trajectory

            # Build orchestrator tools from scenario config
            tools: list[Any] = [
                tool_registry.get(name).create_with_config()
                for name in scenario_config.orchestrator.tools
            ]

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
            )

        raise NotImplementedError(
            f"Orchestrator mode {scenario_config.orchestrator.orchestrator_mode!r} not yet implemented"
        )
