"""AgentSystem and builder entry points for all agent systems.

Two builder paths:

1. **``AgentSystemBuilder.build()``** (legacy) --- Config-driven factory that
   resolves system_type from scenario config.  Backward-compatible with all
   existing callers.

2. **``GenericAgentSystemBuilder[S]``** --- New fluent builder parameterized
   over a user-defined state type ``S``.  Accepts a ``ReasoningStrategy[S]``
   and composes middleware, backend, and tools explicitly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Generic

from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from dataclasses import asdict

from agentm.exceptions import ConfigError
from agentm.agents.node.orchestrator import create_node_orchestrator
from agentm.agents.node.worker import AgentPool
from agentm.config.schema import ScenarioConfig, StorageConfig
from agentm.core.backend import StorageBackend
from agentm.core.state_registry import get_state_schema
from agentm.core.strategy import ReasoningStrategy
from agentm.core.task_manager import TaskManager
from agentm.core.tool_registry import ToolRegistry
from agentm.core.trajectory import TrajectoryCollector
from agentm.middleware import AgentMMiddleware
from agentm.models.state import S
from agentm.tools import knowledge as knowledge_module
from agentm.tools import memory as memory_module
from agentm.tools.orchestrator import create_orchestrator_tools
from agentm.tools.think import think


def _resolve_format_context(system_type: str) -> Any:
    """Lazily resolve the context formatter for a system type.

    Imports scenario formatters on demand.  By this point
    ``scenarios.discover()`` has already run.
    """
    if system_type == "hypothesis_driven":
        from agentm.scenarios.rca.formatters import format_rca_context

        return format_rca_context
    if system_type == "memory_extraction":
        from agentm.scenarios.memory_extraction.formatters import (
            format_memory_extraction_context,
        )

        return format_memory_extraction_context
    return None


def _serialize_notebook(notebook: Any) -> dict[str, Any]:
    """Serialize a DiagnosticNotebook dataclass to a JSON-safe dict."""
    if hasattr(notebook, "__dataclass_fields__"):
        raw = asdict(notebook)
    elif isinstance(notebook, dict):
        raw = notebook
    else:
        return {}
    # asdict converts enums to their value automatically
    return raw


class AgentSystem(Generic[S]):
    """Unified interface for all agent systems.

    Parameterized over ``S`` so that callers using the generic builder
    get typed ``execute`` / ``stream`` signatures.
    """

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

    async def __aenter__(self) -> AgentSystem[S]:
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        if self.trajectory is not None:
            await self.trajectory.close()
        await self._close_checkpointer()

    async def _close_checkpointer(self) -> None:
        """Close the checkpointer's underlying connection if applicable."""
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer is None:
            return
        conn = getattr(checkpointer, "conn", None)
        if conn is not None and hasattr(conn, "close"):
            try:
                await conn.close()
            except Exception:
                pass

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

            # Record to trajectory --- listeners are notified automatically
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
        """Parse node output and record structured trajectory events.

        ``llm_call`` node events (tool_call, llm_end) are recorded by
        ``NodePipeline`` inside the node itself --- only state_update and
        tool_result events are recorded here from the stream.
        """
        if self.trajectory is None:
            return

        # Broadcast notebook state updates for the Conversation View
        notebook = node_data.get("notebook")
        if notebook is not None:
            notebook_data = _serialize_notebook(notebook)
            task_id = (
                notebook_data.get("task_id", "")
                if isinstance(notebook_data, dict)
                else ""
            )
            raw_phase = node_data.get("current_phase", "")
            current_phase = (
                raw_phase.value if hasattr(raw_phase, "value") else str(raw_phase)
            )
            await self.trajectory.record(
                event_type="state_update",
                agent_path=["orchestrator"],
                node_name=node_name,
                data={
                    "notebook": notebook_data,
                    "task_id": task_id,
                    "current_phase": current_phase,
                    "step": step,
                },
            )

        # llm_call events are already recorded by NodePipeline
        if node_name == "llm_call":
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
                            "content": content,
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
                        "result": content if content else "",
                        "step": step,
                    },
                )


def _create_checkpointer(storage_config: StorageConfig) -> Any:
    """Create a LangGraph checkpointer from storage config.

    For 'memory' backend, returns a MemorySaver directly.
    For 'sqlite' backend, returns None here --- the AsyncSqliteSaver must be
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


_DEFAULT_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "tools"


# ---------------------------------------------------------------------------
# New generic fluent builder
# ---------------------------------------------------------------------------


class GenericAgentSystemBuilder(Generic[S]):
    """Fluent builder for constructing ``AgentSystem[S]`` instances.

    Uses explicit ``ReasoningStrategy[S]``, middleware, and backend
    rather than inferring everything from a config file.

    Example::

        system = (
            GenericAgentSystemBuilder(SupportTicketState)
            .with_strategy(SupportTicketStrategy())
            .with_scenario(scenario_config)
            .with_middleware([CompressionMiddleware(cfg), BudgetMiddleware(20)])
            .build()
        )
    """

    def __init__(self, state_schema: type[S]) -> None:
        self._state_schema = state_schema
        self._strategy: ReasoningStrategy[S] | None = None
        self._scenario: ScenarioConfig | None = None
        self._backend: StorageBackend | None = None
        self._middleware: list[AgentMMiddleware] = []
        self._tools: list[Any] = []
        self._system_config: Any | None = None
        self._thread_id: str | None = None

    def with_strategy(
        self, strategy: ReasoningStrategy[S]
    ) -> GenericAgentSystemBuilder[S]:
        """Set the reasoning strategy."""
        self._strategy = strategy
        return self

    def with_scenario(
        self, config: ScenarioConfig
    ) -> GenericAgentSystemBuilder[S]:
        """Set the scenario configuration."""
        self._scenario = config
        return self

    def with_backend(
        self, backend: StorageBackend
    ) -> GenericAgentSystemBuilder[S]:
        """Set the storage backend."""
        self._backend = backend
        return self

    def with_middleware(
        self, middleware: Sequence[AgentMMiddleware]
    ) -> GenericAgentSystemBuilder[S]:
        """Set the middleware pipeline (replaces any previous list)."""
        self._middleware = list(middleware)
        return self

    def with_tools(self, tools: list[Any]) -> GenericAgentSystemBuilder[S]:
        """Set additional tools for the orchestrator."""
        self._tools = list(tools)
        return self

    def with_system_config(
        self, system_config: Any
    ) -> GenericAgentSystemBuilder[S]:
        """Set the system-level config (models, storage, debug)."""
        self._system_config = system_config
        return self

    def with_thread_id(
        self, thread_id: str
    ) -> GenericAgentSystemBuilder[S]:
        """Set an explicit thread ID for checkpoint continuity."""
        self._thread_id = thread_id
        return self

    def build(self) -> AgentSystem[S]:
        """Build the ``AgentSystem[S]``.

        Requires at least a strategy and scenario config.
        """
        from agentm.scenarios import discover as _discover_scenarios

        _discover_scenarios()

        if self._strategy is None:
            raise ValueError("Strategy is required --- call with_strategy()")
        if self._scenario is None:
            raise ValueError(
                "Scenario config is required --- call with_scenario()"
            )

        strategy = self._strategy
        scenario = self._scenario
        system_config = self._system_config

        # --- Checkpointer ---
        checkpointer = None
        pending_storage: StorageConfig | None = None
        if system_config is not None:
            checkpointer = _create_checkpointer(system_config.storage)
            if (
                checkpointer is None
                and system_config.storage.checkpointer.backend != "memory"
            ):
                pending_storage = system_config.storage

        thread_id = self._thread_id or str(uuid.uuid4())
        langgraph_config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
        }

        # --- Trajectory ---
        trajectory: TrajectoryCollector | None = None
        if system_config is not None and system_config.debug.trajectory.enabled:
            run_id = f"{strategy.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            checkpoint_db_path = ""
            if system_config.storage.checkpointer.backend == "sqlite":
                db_url = (
                    system_config.storage.checkpointer.url or "./checkpoints.db"
                )
                checkpoint_db_path = str(Path(db_url).resolve())
            trajectory = TrajectoryCollector(
                run_id=run_id,
                output_dir=system_config.debug.trajectory.output_dir,
                thread_id=thread_id,
                checkpoint_db=checkpoint_db_path,
            )

        # --- Tools ---
        tools = list(self._tools)
        tools.append(think)

        # --- Format context via strategy ---
        format_context = strategy.format_context

        # --- State schema ---
        state_schema = strategy.state_schema()

        # --- Build graph ---
        orch_model_config = None
        if system_config is not None:
            orch_model_name = scenario.orchestrator.model
            orch_model_config = system_config.models.get(orch_model_name)

        graph = create_node_orchestrator(
            config=scenario.orchestrator,
            tools=tools,
            checkpointer=checkpointer,
            store=None,
            state_schema=state_schema,
            format_context=format_context,
            model_config=orch_model_config,
            trajectory=trajectory,
        )

        return AgentSystem(
            graph,
            config=langgraph_config,
            scenario_config=scenario,
            trajectory=trajectory,
            thread_id=thread_id,
            _pending_storage=pending_storage,
        )

    @staticmethod
    def create(
        strategy: ReasoningStrategy[S],
        scenario_config: ScenarioConfig,
        system_config: Any | None = None,
        middleware: Sequence[AgentMMiddleware] | None = None,
        tools: list[Any] | None = None,
        thread_id: str | None = None,
    ) -> AgentSystem[S]:
        """Convenience factory --- builds an AgentSystem in one call."""
        builder: GenericAgentSystemBuilder[S] = GenericAgentSystemBuilder(
            strategy.state_schema()
        )
        builder.with_strategy(strategy).with_scenario(scenario_config)
        if system_config is not None:
            builder.with_system_config(system_config)
        if middleware is not None:
            builder.with_middleware(middleware)
        if tools is not None:
            builder.with_tools(tools)
        if thread_id is not None:
            builder.with_thread_id(thread_id)
        return builder.build()


# ---------------------------------------------------------------------------
# Legacy builder --- preserved for backward compatibility
# ---------------------------------------------------------------------------


class AgentSystemBuilder:
    """Unified entry point for building any agent system (legacy API).

    Internally selects the appropriate architecture based on system_type:
    - ReAct-based (create_react_agent): For exploratory, non-linear scenarios like RCA
    - StateGraph-based (custom graph with phase nodes): For linear, deterministic scenarios

    For new code, prefer ``GenericAgentSystemBuilder[S]``.
    """

    @staticmethod
    def build(
        system_type: str,
        scenario_config: ScenarioConfig,
        system_config: Any | None = None,
        existing_thread_id: str | None = None,
        tools_dir: Path | str | None = None,
        knowledge_base_dir: str | None = None,
    ) -> AgentSystem[Any]:
        """Build an AgentSystem from a system type and scenario config."""
        from agentm.scenarios import discover as _discover_scenarios

        _discover_scenarios()
        get_state_schema(system_type)  # validate system_type exists

        resolved_tools_dir = Path(tools_dir) if tools_dir is not None else _DEFAULT_TOOLS_DIR
        resolved_kb_dir = knowledge_base_dir if knowledge_base_dir is not None else "./knowledge"

        if scenario_config.orchestrator.orchestrator_mode in ("react", "node"):
            tool_registry = ToolRegistry()

            # Load all tool YAMLs
            for yaml_file in sorted(resolved_tools_dir.glob("*.yaml")):
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

            # --- Checkpointer ---
            checkpointer = None
            pending_storage: StorageConfig | None = None
            if system_config is not None:
                checkpointer = _create_checkpointer(system_config.storage)
                if (
                    checkpointer is None
                    and system_config.storage.checkpointer.backend != "memory"
                ):
                    pending_storage = system_config.storage

            # Wire checkpointer into agent_pool so workers persist per-task state
            agent_pool._checkpointer = checkpointer

            task_manager = TaskManager()

            thread_id = existing_thread_id if existing_thread_id else str(uuid.uuid4())
            langgraph_config: dict[str, Any] = {
                "configurable": {"thread_id": thread_id},
            }

            # --- Trajectory ---
            trajectory: TrajectoryCollector | None = None
            if system_config is not None and system_config.debug.trajectory.enabled:
                run_id = f"{system_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                # Compute checkpoint db path for metadata
                checkpoint_db_path = ""
                if system_config.storage.checkpointer.backend == "sqlite":
                    db_url = (
                        system_config.storage.checkpointer.url or "./checkpoints.db"
                    )
                    checkpoint_db_path = str(Path(db_url).resolve())
                trajectory = TrajectoryCollector(
                    run_id=run_id,
                    output_dir=system_config.debug.trajectory.output_dir,
                    thread_id=thread_id,
                    checkpoint_db=checkpoint_db_path,
                )

            # --- Dependency injection via closure factory ---
            injected_tools = create_orchestrator_tools(
                task_manager, agent_pool, trajectory=trajectory,
                config=scenario_config.orchestrator,
            )

            # Load scenario-specific tools based on system_type
            format_context = None  # may be overridden by scenario-specific block below
            if system_type == "hypothesis_driven":
                from functools import partial

                from agentm.scenarios.rca.formatters import format_rca_context
                from agentm.scenarios.rca.service_profile import ServiceProfileStore
                from agentm.scenarios.rca.tools import create_rca_tools

                profile_store = ServiceProfileStore()
                rca_tools = create_rca_tools(
                    trajectory=trajectory, profile_store=profile_store
                )

                # Extract worker tools before registering orchestrator tools
                worker_profile_tools = rca_tools.pop("_worker_profile_tools", [])
                injected_tools.update(rca_tools)

                # Wire profile store into format_context (replaces lazy-resolved one)
                format_context = partial(format_rca_context, profile_store=profile_store)

                # Inject profile tools into worker pool
                if worker_profile_tools:
                    agent_pool._extra_worker_tools = worker_profile_tools

            # Wire trajectory into task_manager
            if trajectory is not None:
                task_manager.set_trajectory(trajectory)

                # Wire trajectory into agent_pool for llm_start events
                agent_pool._trajectory = trajectory

            # Extract graph reference setter before tool registration (not a real tool)
            set_graph_ref = injected_tools.pop("_set_graph_ref", None)

            # --- Knowledge Store ---
            knowledge_module.init(base_dir=resolved_kb_dir)

            # Wire checkpointer DB path into memory module for memory-extraction system
            if system_config is not None:
                db_url = system_config.storage.checkpointer.url or "./checkpoints.db"
                memory_module.set_db_path(str(Path(db_url).resolve()))

            # Knowledge tools are standalone functions (file-system backend)
            KNOWLEDGE_TOOLS: dict[str, Any] = {
                "knowledge_search": knowledge_module.knowledge_search,
                "knowledge_list": knowledge_module.knowledge_list,
                "knowledge_read": knowledge_module.knowledge_read,
                "knowledge_write": knowledge_module.knowledge_write,
                "knowledge_delete": knowledge_module.knowledge_delete,
            }

            # Memory tools are standalone functions (checkpointer injected above)
            MEMORY_TOOLS: dict[str, Any] = {
                "read_trajectory": memory_module.read_trajectory,
                "get_checkpoint_history": memory_module.get_checkpoint_history,
            }

            # Build orchestrator tools: YAML-registered + factory-injected + knowledge + memory
            tools: list[Any] = []

            for name in scenario_config.orchestrator.tools:
                if name in injected_tools:
                    # Factory-created tool (has closure over task_manager/agent_pool)
                    func = injected_tools[name]
                    if asyncio.iscoroutinefunction(func):
                        tool = StructuredTool.from_function(
                            coroutine=func,
                            name=name,
                            description=func.__doc__ or name,
                        )
                    else:
                        tool = StructuredTool.from_function(
                            func=func,
                            name=name,
                            description=func.__doc__ or name,
                        )
                    tools.append(tool)
                elif name in KNOWLEDGE_TOOLS:
                    # Knowledge tools (file-system backend, initialized via init())
                    func = KNOWLEDGE_TOOLS[name]
                    tool = StructuredTool.from_function(
                        func=func,
                        name=name,
                        description=func.__doc__ or name,
                    )
                    tools.append(tool)
                elif name in MEMORY_TOOLS:
                    # Memory tools (standalone functions with checkpointer injected)
                    func = MEMORY_TOOLS[name]
                    tool = StructuredTool.from_function(
                        func=func,
                        name=name,
                        description=func.__doc__ or name,
                    )
                    tools.append(tool)
                elif tool_registry.has(name):
                    # YAML-registered tool (module-level function)
                    tools.append(tool_registry.get(name).create_with_config())
                else:
                    raise ConfigError(f"Tool {name!r} not found in registry or factory")

            # Think tool is always available for structured reasoning
            tools.append(think)

            # Resolve state schema and context formatter for this system type
            state_schema = get_state_schema(system_type)
            if format_context is None:
                format_context = _resolve_format_context(system_type)

            graph = create_node_orchestrator(
                config=scenario_config.orchestrator,
                tools=tools,
                checkpointer=checkpointer,
                store=None,
                state_schema=state_schema,
                format_context=format_context,
                model_config=orch_model_config,
                trajectory=trajectory,
            )

            # Wire graph reference for recall_history (must happen after graph compilation)
            if set_graph_ref is not None:
                set_graph_ref(graph, langgraph_config)

            return AgentSystem(
                graph,
                config=langgraph_config,
                scenario_config=scenario_config,
                task_manager=task_manager,
                trajectory=trajectory,
                thread_id=thread_id,
                _pending_storage=pending_storage,
            )

        raise ConfigError(
            f"Orchestrator mode {scenario_config.orchestrator.orchestrator_mode!r} not yet implemented"
        )


class AgentSystemBuilderFluent:
    """Fluent builder for customized AgentSystem construction.

    Usage::

        system = (
            AgentSystemBuilderFluent("hypothesis_driven", scenario_cfg)
            .with_system_config(sys_cfg)
            .with_tools_dir("/my/tools")
            .with_knowledge_base_dir("/my/knowledge")
            .build()
        )
    """

    def __init__(self, system_type: str, scenario_config: ScenarioConfig) -> None:
        self._system_type = system_type
        self._scenario_config = scenario_config
        self._system_config: Any | None = None
        self._thread_id: str | None = None
        self._tools_dir: Path | str | None = None
        self._knowledge_base_dir: str | None = None

    def with_system_config(self, config: Any) -> AgentSystemBuilderFluent:
        self._system_config = config
        return self

    def with_thread_id(self, thread_id: str) -> AgentSystemBuilderFluent:
        self._thread_id = thread_id
        return self

    def with_tools_dir(self, path: Path | str) -> AgentSystemBuilderFluent:
        self._tools_dir = path
        return self

    def with_knowledge_base_dir(self, path: str) -> AgentSystemBuilderFluent:
        self._knowledge_base_dir = path
        return self

    def build(self) -> AgentSystem[Any]:
        """Build with all configured options."""
        return AgentSystemBuilder.build(
            system_type=self._system_type,
            scenario_config=self._scenario_config,
            system_config=self._system_config,
            existing_thread_id=self._thread_id,
            tools_dir=self._tools_dir,
            knowledge_base_dir=self._knowledge_base_dir,
        )


def build_from_type(
    system_type: str,
    scenario_config: ScenarioConfig,
    system_config: Any | None = None,
    existing_thread_id: str | None = None,
) -> AgentSystem[Any]:
    """Bridge function --- delegates to ``AgentSystemBuilder.build()``.

    Provided for callers that prefer a plain function over a static method.
    """
    return AgentSystemBuilder.build(
        system_type, scenario_config, system_config, existing_thread_id
    )
