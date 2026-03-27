"""AgentSystem and build_agent_system() — single entry point for all agent systems.

``build_agent_system()`` is the canonical way to construct an ``AgentSystem``.
It uses the Scenario protocol to wire domain-specific behavior, and the SDK
takes care of platform resources (vault, trajectory, runtime, tools).

Legacy ``AgentSystemBuilder.build()`` and ``build_from_type()`` are kept as
backward-compatible aliases.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from agentm.config.schema import ScenarioConfig, create_chat_model
from agentm.core.prompt import load_prompt_template
from agentm.core.tool_registry import ToolRegistry
from agentm.core.trajectory import TrajectoryCollector
from agentm.exceptions import ConfigError
from agentm.harness.adapters import TrajectoryEventAdapter
from agentm.harness.loops.simple import SimpleAgentLoop
from agentm.harness.middleware import (
    CompressionMiddleware,
    DynamicContextMiddleware,
    LoopDetectionMiddleware,
    SkillMiddleware,
    TrajectoryMiddleware,
)
from agentm.harness.runtime import AgentRuntime
from agentm.harness.scenario import SetupContext, get_scenario
from agentm.harness.tool import Tool, tool_from_function
from agentm.harness.types import AgentResult, RunConfig
from agentm.harness.worker_factory import WorkerLoopFactory
from agentm.tools.orchestrator import create_orchestrator_tools
from agentm.tools.think import think


# ---------------------------------------------------------------------------
# Decision-based termination (replaces <decision> routing in the LangGraph
# orchestrator). The LLM emits <decision>finalize</decision> when it wants
# to stop; otherwise it keeps calling tools.
# ---------------------------------------------------------------------------

_DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)


def _orchestrator_should_terminate(response: Any) -> bool:
    """Return True when the LLM signals finalize via <decision> tag.

    Falls back to the standard no-tool-calls heuristic when the tag is
    absent, for backward compatibility with prompts that don't emit it.
    """
    content = getattr(response, "content", "") or ""
    match = _DECISION_RE.search(content)
    if match:
        return match.group(1).strip().lower() == "finalize"
    # No decision tag: fall back to no-tool-calls = terminate
    return not getattr(response, "tool_calls", None)


# ---------------------------------------------------------------------------
# AgentSystem
# ---------------------------------------------------------------------------


class AgentSystem:
    """Unified interface for all agent systems.

    The orchestrator is a SimpleAgentLoop. ``execute()`` and ``stream()``
    delegate to the loop's ``run()`` and ``stream()`` methods.
    """

    def __init__(
        self,
        loop: Any,
        scenario_config: ScenarioConfig | None = None,
        runtime: AgentRuntime | None = None,
        trajectory: TrajectoryCollector | None = None,
        thread_id: str = "",
    ) -> None:
        self.loop = loop
        self.scenario_config = scenario_config
        self.runtime = runtime
        self.trajectory = trajectory
        self.thread_id = thread_id

    async def __aenter__(self) -> AgentSystem:
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        if self.trajectory is not None:
            await self.trajectory.close()

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent system with the given input. Returns final result."""
        task = input_data.get("task_description") or input_data.get("messages", [{}])[0].get("content", "")
        if isinstance(task, list):
            # Handle LangChain message objects
            task = str(task[0]) if task else ""
        result: AgentResult = await self.loop.run(
            str(task),
            config=RunConfig(metadata={"agent_id": "orchestrator"}),
        )
        return {"output": result.output, "status": result.status.value, "steps": result.steps}

    async def stream(
        self,
        input_data: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events from the agent system execution.

        Yields AgentEvent-like dicts from the SimpleAgentLoop's stream().
        """
        task = input_data.get("task_description") or input_data.get("messages", [{}])[0].get("content", "")
        if isinstance(task, list):
            task = str(task[0]) if task else ""

        async for event in self.loop.stream(
            str(task),
            config=RunConfig(metadata={"agent_id": "orchestrator"}),
        ):
            yield {"event": event}


_DEFAULT_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "tools"


# ---------------------------------------------------------------------------
# build_agent_system() — the single canonical builder
# ---------------------------------------------------------------------------


def build_agent_system(
    scenario_name: str,
    scenario_config: ScenarioConfig,
    system_config: Any | None = None,
    *,
    thread_id: str | None = None,
    tools_dir: Path | str | None = None,
    knowledge_base_dir: str | None = None,
) -> AgentSystem:
    """Build an AgentSystem from a scenario name and config.

    This is the single canonical entry point for constructing any agent
    system.  It uses the Scenario protocol to wire domain-specific behavior.

    Flow:
    1. Discover scenarios, look up by name
    2. Create platform resources (vault, trajectory, tool_registry)
    3. Call scenario.setup(ctx) to get ScenarioWiring
    4. Create AgentRuntime, WorkerLoopFactory
    5. Build orchestrator tools (SDK + scenario + registry)
    6. Build middleware stack (SDK + scenario)
    7. Build SimpleAgentLoop
    8. Return AgentSystem

    Args:
        scenario_name: Registered scenario name (e.g. "hypothesis_driven").
        scenario_config: Full scenario configuration.
        system_config: Optional system-level config (models, storage, debug).
        thread_id: Optional thread ID for checkpoint continuity.
        tools_dir: Directory containing tool YAML definitions.
        knowledge_base_dir: Path to the knowledge base (vault root).
    """
    from agentm.scenarios import discover as _discover_scenarios

    _discover_scenarios()

    # 1. Look up the Scenario
    scenario = get_scenario(scenario_name)

    # --- Resolve directories ---
    resolved_tools_dir = (
        Path(tools_dir) if tools_dir is not None else _DEFAULT_TOOLS_DIR
    )
    resolved_kb_dir = (
        knowledge_base_dir if knowledge_base_dir is not None else "./knowledge"
    )

    resolved_thread_id = thread_id if thread_id else str(uuid.uuid4())

    # --- Tool registry ---
    tool_registry = ToolRegistry()
    for yaml_file in sorted(resolved_tools_dir.glob("*.yaml")):
        tool_registry.load_from_yaml(yaml_file)

    # --- Resolve model config ---
    orch_model_name = scenario_config.orchestrator.model
    orch_model_config = (
        system_config.models.get(orch_model_name)
        if system_config is not None
        else None
    )

    worker_config = scenario_config.agents.get("worker")
    worker_model_config = None
    if worker_config is not None and system_config is not None:
        worker_model_config = system_config.models.get(worker_config.model)

    # --- Trajectory ---
    trajectory: TrajectoryCollector | None = None
    if system_config is not None and system_config.debug.trajectory.enabled:
        run_id = f"{scenario_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        checkpoint_db_path = ""
        if system_config.storage.checkpointer.backend == "sqlite":
            db_url = (
                system_config.storage.checkpointer.url or "./checkpoints.db"
            )
            checkpoint_db_path = str(Path(db_url).resolve())
        trajectory = TrajectoryCollector(
            run_id=run_id,
            output_dir=system_config.debug.trajectory.output_dir,
            thread_id=resolved_thread_id,
            checkpoint_db=checkpoint_db_path,
        )

    # --- Vault ---
    from agentm.tools.vault import MarkdownVault, create_vault_tools

    vault_dir = Path(resolved_kb_dir) / "vault"
    vault = MarkdownVault(vault_dir)
    vault_tools = create_vault_tools(vault)

    # Register vault tools into tool_registry
    for vt_name, vt_func in vault_tools.items():
        tool_registry.register(vt_name, vt_func, {})

    # --- Memory tools ---
    from agentm.tools import memory as memory_module
    from agentm.tools import trajectory_reader as traj_reader_module

    if system_config is not None:
        db_url = system_config.storage.checkpointer.url or "./checkpoints.db"
        memory_module.set_db_path(str(Path(db_url).resolve()))

    memory_tools: dict[str, Any] = {
        "read_trajectory": memory_module.read_trajectory,
        "get_checkpoint_history": memory_module.get_checkpoint_history,
        "jq_query": traj_reader_module.jq_query,
    }

    for mt_name, mt_func in memory_tools.items():
        tool_registry.register(mt_name, mt_func, {})

    # 2. Create SetupContext and call scenario.setup()
    ctx = SetupContext(
        vault=vault,
        trajectory=trajectory,
        tool_registry=tool_registry,
    )
    wiring = scenario.setup(ctx)

    # 3. Create AgentRuntime
    event_handler = TrajectoryEventAdapter(trajectory) if trajectory is not None else None
    runtime = AgentRuntime(event_handler=event_handler)

    # 4. Create WorkerLoopFactory
    # Compose worker middleware: scenario middleware + SkillMiddleware (if applicable)
    worker_middleware: list[Any] = list(wiring.worker_middleware or [])
    worker_config = scenario_config.agents.get("worker")
    if worker_config is not None and worker_config.skills and vault is not None:
        worker_middleware.append(SkillMiddleware(vault, worker_config.skills))

    worker_factory = WorkerLoopFactory(
        scenario_config,
        tool_registry,
        worker_model_config,
        extra_tools=wiring.worker_tools or None,
        extra_middleware=worker_middleware or None,
        trajectory=trajectory,
        answer_schemas=wiring.answer_schemas or None,
    )

    # 5. Create SDK tools (dispatch, check, inject, abort)
    sdk_tools_dict = create_orchestrator_tools(
        runtime,
        worker_factory,
    )

    # Build orchestrator-tools lookup from wiring
    scenario_tools_by_name: dict[str, Tool] = {
        t.name: t for t in wiring.orchestrator_tools
    }

    # 6. Build the tool list from config
    tools: list[Tool] = []
    for name in scenario_config.orchestrator.tools:
        if name in sdk_tools_dict:
            # SDK tool (has closure over runtime/worker_factory)
            tools.append(tool_from_function(sdk_tools_dict[name], name=name))
        elif name in scenario_tools_by_name:
            # Scenario-provided tool — use directly
            tools.append(scenario_tools_by_name[name])
        elif name in memory_tools:
            # Memory tools (standalone functions)
            tools.append(tool_from_function(memory_tools[name], name=name))
        elif tool_registry.has(name):
            # Registry tool (YAML-declared or vault tools)
            tools.append(tool_registry.get(name).create_tool())
        else:
            raise ConfigError(f"Tool {name!r} not found in registry or factory")

    # Think tool is always available
    tools.append(think)

    # 7. Build middleware stack
    hooks = wiring.hooks
    format_context = wiring.format_context

    # --- System prompt ---
    config = scenario_config.orchestrator
    system_prompt_template = config.prompts.get("system", "")
    if system_prompt_template:
        static_system_prompt: str = load_prompt_template(
            system_prompt_template,
            notebook="",
            context="",
        )
    else:
        static_system_prompt = "You are an agent orchestrator."

    max_rounds: int = config.max_rounds

    orch_middleware: list[Any] = []

    # 7a. Dynamic context
    orch_middleware.append(DynamicContextMiddleware(
        format_context_fn=format_context if callable(format_context) else lambda: "",
        base_system_prompt=static_system_prompt,
        max_rounds=max_rounds,
    ))

    # 7b. Think-stall detection
    if hooks.think_stall_enabled:
        orch_middleware.append(
            LoopDetectionMiddleware(
                threshold=5,
                window_size=15,
                think_stall_limit=hooks.think_stall_limit,
            )
        )

    # 7c. Compression
    compression_cfg = config.compression
    if compression_cfg is not None and compression_cfg.enabled:
        orch_middleware.append(
            CompressionMiddleware(compression_cfg, model_config=orch_model_config)
        )

    # 7d. Skills
    if config.skills and vault is not None:
        orch_middleware.append(SkillMiddleware(vault, config.skills))

    # 7e. Scenario-specific middleware
    if wiring.orchestrator_middleware:
        orch_middleware.extend(wiring.orchestrator_middleware)

    # 7f. Trajectory
    if trajectory is not None:
        orch_middleware.append(
            TrajectoryMiddleware(trajectory, agent_path=["orchestrator"])
        )

    # 8. Build model
    model_plain = create_chat_model(
        model=config.model,
        temperature=config.temperature,
        model_config=orch_model_config,
    )
    if config.disable_tool_binding:
        model_with_tools = model_plain
    else:
        model_with_tools = model_plain.bind_tools(
            [t.to_openai_schema() for t in tools]
        )

    # --- Structured output ---
    output_schema = wiring.output_schema
    output_prompt_text = ""
    if config.output is not None:
        output_prompt_text = load_prompt_template(config.output.prompt)

    # --- Termination ---
    should_terminate = wiring.should_terminate or _orchestrator_should_terminate

    # 9. Build SimpleAgentLoop
    orch_loop = SimpleAgentLoop(
        model=model_with_tools,
        tools=tools,
        system_prompt="",  # Managed by DynamicContextMiddleware
        middleware=orch_middleware,
        output_schema=output_schema,
        output_prompt=output_prompt_text,
        synthesize_retries=hooks.synthesize_max_retries,
        should_terminate=should_terminate,
    )

    return AgentSystem(
        orch_loop,
        scenario_config=scenario_config,
        runtime=runtime,
        trajectory=trajectory,
        thread_id=resolved_thread_id,
    )


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------


class AgentSystemBuilder:
    """Legacy entry point — delegates to ``build_agent_system()``.

    Preserved for backward compatibility with existing callers.
    """

    @staticmethod
    def build(
        system_type: str,
        scenario_config: ScenarioConfig,
        system_config: Any | None = None,
        existing_thread_id: str | None = None,
        tools_dir: Path | str | None = None,
        knowledge_base_dir: str | None = None,
    ) -> AgentSystem:
        """Build an AgentSystem from a system type and scenario config."""
        return build_agent_system(
            system_type,
            scenario_config,
            system_config,
            thread_id=existing_thread_id,
            tools_dir=tools_dir,
            knowledge_base_dir=knowledge_base_dir,
        )


def build_from_type(
    system_type: str,
    scenario_config: ScenarioConfig,
    system_config: Any | None = None,
    existing_thread_id: str | None = None,
) -> AgentSystem:
    """Bridge function --- delegates to ``build_agent_system()``.

    Provided for callers that prefer a plain function over a static method.
    """
    return build_agent_system(
        system_type,
        scenario_config,
        system_config,
        thread_id=existing_thread_id,
    )
