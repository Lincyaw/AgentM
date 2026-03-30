"""AgentSystem and build_agent_system() — single entry point for all agent systems.

``build_agent_system()`` is the canonical way to construct an ``AgentSystem``.
It uses the Scenario protocol to wire domain-specific behavior, and the SDK
takes care of platform resources (vault, trajectory, runtime, tools).

The build process is decomposed into four phases:
1. _create_platform_resources — vault, trajectory, tool registry
2. _create_worker_infrastructure — runtime, worker factory
3. _assemble_orchestrator_tools — SDK + scenario + registry tools
4. _build_orchestrator_loop — middleware stack + LLM + SimpleAgentLoop

"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agentm.config.schema import CompressionConfig, ModelConfig, ScenarioConfig, SystemConfig, create_chat_model
from agentm.core.prompt import load_prompt_template
from agentm.core.tool_registry import ToolRegistry
from agentm.core.trajectory import TrajectoryCollector
from agentm.exceptions import ConfigError
from agentm.harness.loops.simple import SimpleAgentLoop
from agentm.harness.middleware import (
    CompressionMiddleware,
    DynamicContextMiddleware,
    LoopDetectionMiddleware,
    MiddlewareBase,
    SkillMiddleware,
    TrajectoryMiddleware,
)
from agentm.harness.runtime import AgentRuntime
from agentm.harness.scenario import ScenarioWiring, SetupContext, get_scenario
from agentm.harness.tool import Tool, tool_from_function
from agentm.harness.types import AgentInput, AgentOutput, AgentResult, Message, RunConfig, ToolCallable
from agentm.harness.worker_factory import WorkerLoopFactory
from agentm.models.data import OrchestratorHooks
from agentm.tools.orchestrator import create_orchestrator_tools
from agentm.tools.vault.store import MarkdownVault


# ---------------------------------------------------------------------------
# Decision-based termination (replaces <decision> routing in the LangGraph
# orchestrator). The LLM emits <decision>finalize</decision> when it wants
# to stop; otherwise it keeps calling tools.
# ---------------------------------------------------------------------------

_DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)


def _orchestrator_should_terminate(response: object) -> bool:
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


def _default_hooks() -> OrchestratorHooks:
    """Return default orchestrator hooks."""
    return OrchestratorHooks()


# ---------------------------------------------------------------------------
# AgentSystem
# ---------------------------------------------------------------------------


def _to_messages(input_data: AgentInput | dict[str, Any]) -> list[Message]:
    """Convert agent input to a message list.

    Priority:
      1. ``messages`` -- used directly if present and non-empty.
      2. ``task_description`` -- wrapped as a single human message.
    """
    messages = input_data.get("messages")
    if messages:
        return list(messages)
    task = input_data.get("task_description", "")
    if task:
        if isinstance(task, list):
            task = str(task[0]) if task else ""
        return [{"role": "human", "content": str(task)}]
    return []


class AgentSystem:
    """Unified interface for all agent systems.

    The orchestrator is a SimpleAgentLoop. ``execute()`` and ``stream()``
    delegate to the loop's ``run()`` and ``stream()`` methods.
    """

    def __init__(
        self,
        loop: SimpleAgentLoop,
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
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object | None
    ) -> None:
        if self.trajectory is not None:
            await self.trajectory.close()

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute the agent system with the given input. Returns final result."""
        messages = _to_messages(input_data)
        result: AgentResult = await self.loop.run(
            messages,
            config=RunConfig(metadata={"agent_id": "orchestrator"}),
        )
        return {"output": result.output, "status": result.status.value, "steps": result.steps}

    async def stream(
        self,
        input_data: AgentInput,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events from the agent system execution.

        Yields AgentEvent-like dicts from the SimpleAgentLoop's stream().
        """
        messages = _to_messages(input_data)

        async for event in self.loop.stream(
            messages,
            config=RunConfig(metadata={"agent_id": "orchestrator"}),
        ):
            yield {"event": event}


_DEFAULT_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "tools"


# ---------------------------------------------------------------------------
# PlatformResources — intermediate container for phase 1 outputs
# ---------------------------------------------------------------------------


@dataclass
class _PlatformResources:
    """Intermediate container holding all platform resources created in phase 1."""

    tool_registry: ToolRegistry
    vault: MarkdownVault | None
    trajectory: TrajectoryCollector | None
    memory_tools: dict[str, ToolCallable]
    orch_model_config: ModelConfig | None
    worker_model_config: ModelConfig | None
    thread_id: str


# ---------------------------------------------------------------------------
# Phase 1: Create platform resources
# ---------------------------------------------------------------------------


def _create_platform_resources(
    scenario_name: str,
    scenario_config: ScenarioConfig,
    system_config: SystemConfig | None,
    *,
    thread_id: str | None,
    tools_dir: Path | str | None,
    knowledge_base_dir: str | None,
) -> _PlatformResources:
    """Create vault, trajectory, tool registry, and memory tools.

    Pure setup — no scenario-specific logic.
    """
    resolved_tools_dir = (
        Path(tools_dir) if tools_dir is not None else _DEFAULT_TOOLS_DIR
    )
    resolved_kb_dir = (
        knowledge_base_dir if knowledge_base_dir is not None else "./knowledge"
    )
    resolved_thread_id = thread_id if thread_id else str(uuid.uuid4())

    # Tool registry
    tool_registry = ToolRegistry()
    for yaml_file in sorted(resolved_tools_dir.glob("*.yaml")):
        tool_registry.load_from_yaml(yaml_file)

    # Model configs
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

    # Trajectory
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

    # Vault
    from agentm.tools.vault import MarkdownVault, create_vault_tools

    vault_dir = Path(resolved_kb_dir) / "vault"
    vault = MarkdownVault(vault_dir)
    vault_tools = create_vault_tools(vault)
    for vt_name, vt_func in vault_tools.items():
        tool_registry.register(vt_name, vt_func, {})

    # Memory tools
    from agentm.tools import memory as memory_module
    from agentm.tools import trajectory_reader as traj_reader_module
    from agentm.tools import case_data as case_data_module

    if system_config is not None:
        db_url = system_config.storage.checkpointer.url or "./checkpoints.db"
        memory_module.set_db_path(str(Path(db_url).resolve()))

    memory_tools: dict[str, ToolCallable] = {
        "read_trajectory": memory_module.read_trajectory,
        "get_checkpoint_history": memory_module.get_checkpoint_history,
        "jq_query": traj_reader_module.jq_query,
        "load_case_data": case_data_module.load_case_data,
    }
    for mt_name, mt_func in memory_tools.items():
        tool_registry.register(mt_name, mt_func, {})

    return _PlatformResources(
        tool_registry=tool_registry,
        vault=vault,
        trajectory=trajectory,
        memory_tools=memory_tools,
        orch_model_config=orch_model_config,
        worker_model_config=worker_model_config,
        thread_id=resolved_thread_id,
    )


# ---------------------------------------------------------------------------
# Phase 2: Create worker infrastructure
# ---------------------------------------------------------------------------


def _create_worker_infrastructure(
    scenario_config: ScenarioConfig,
    resources: _PlatformResources,
    wiring: ScenarioWiring,
) -> tuple[AgentRuntime, WorkerLoopFactory]:
    """Create AgentRuntime and WorkerLoopFactory from platform resources + wiring."""
    runtime = AgentRuntime(trajectory=resources.trajectory)

    # Compose worker middleware: scenario middleware + SkillMiddleware
    from agentm.harness.middleware import MiddlewareBase

    worker_middleware: list[MiddlewareBase] = list(wiring.worker_middleware or [])
    worker_config = scenario_config.agents.get("worker")
    if worker_config is not None and worker_config.skills and resources.vault is not None:
        worker_middleware.append(SkillMiddleware(resources.vault, worker_config.skills))

    worker_factory = WorkerLoopFactory(
        scenario_config,
        resources.tool_registry,
        resources.worker_model_config,
        extra_tools=wiring.worker_tools or None,
        extra_middleware=worker_middleware or None,
        trajectory=resources.trajectory,
        answer_schemas=wiring.answer_schemas or None,
    )

    return runtime, worker_factory


# ---------------------------------------------------------------------------
# Phase 3: Assemble orchestrator tools
# ---------------------------------------------------------------------------


def _assemble_orchestrator_tools(
    scenario_config: ScenarioConfig,
    resources: _PlatformResources,
    wiring: ScenarioWiring,
    runtime: AgentRuntime,
    worker_factory: WorkerLoopFactory,
) -> list[Tool]:
    """Resolve and assemble the orchestrator's tool list from all sources."""
    worker_config = scenario_config.agents.get("worker")
    max_cw = (
        worker_config.execution.max_concurrent_workers
        if worker_config is not None
        else None
    )
    sdk_tools_dict = create_orchestrator_tools(
        runtime,
        worker_factory,
        max_concurrent_workers=max_cw,
    )

    scenario_tools_by_name: dict[str, Tool] = {
        t.name: t for t in wiring.orchestrator_tools
    }

    tools: list[Tool] = []
    for name in scenario_config.orchestrator.tools:
        if name in sdk_tools_dict:
            tools.append(tool_from_function(sdk_tools_dict[name], name=name))
        elif name in scenario_tools_by_name:
            tools.append(scenario_tools_by_name[name])
        elif name in resources.memory_tools:
            tools.append(tool_from_function(resources.memory_tools[name], name=name))
        elif resources.tool_registry.has(name):
            tools.append(resources.tool_registry.get(name).create_tool())
        else:
            raise ConfigError(f"Tool {name!r} not found in registry or factory")

    if scenario_config.orchestrator.include_think_tool:
        from agentm.tools.think import think
        tools.append(think)
    return tools


# ---------------------------------------------------------------------------
# Phase 4: Build orchestrator loop
# ---------------------------------------------------------------------------


def _build_orchestrator_loop(
    scenario_config: ScenarioConfig,
    resources: _PlatformResources,
    wiring: ScenarioWiring,
    tools: list[Tool],
) -> SimpleAgentLoop:
    """Build the orchestrator's middleware stack, model, and SimpleAgentLoop."""
    config = scenario_config.orchestrator
    hooks = wiring.hooks

    # System prompt
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
    format_context = wiring.format_context

    # Middleware stack
    orch_middleware: list[MiddlewareBase] = []

    # Hooks should never be None after __post_init__, but mypy doesn't know that
    hooks_safe = hooks or _default_hooks()

    orch_middleware.append(DynamicContextMiddleware(
        format_context_fn=format_context if callable(format_context) else lambda: "",
        base_system_prompt=static_system_prompt,
        max_rounds=max_rounds,
    ))

    if hooks_safe.think_stall_enabled:
        ld = config.loop_detection
        orch_middleware.append(
            LoopDetectionMiddleware(
                threshold=ld.threshold,
                window_size=ld.window_size,
                think_stall_limit=ld.think_stall_limit,
            )
        )

    compression_cfg = config.compression
    if compression_cfg is not None and compression_cfg.enabled:
        orch_middleware.append(
            CompressionMiddleware(compression_cfg, model_config=resources.orch_model_config)
        )

    if config.skills and resources.vault is not None:
        orch_middleware.append(SkillMiddleware(resources.vault, config.skills))

    if wiring.orchestrator_middleware:
        orch_middleware.extend(wiring.orchestrator_middleware)

    if resources.trajectory is not None:
        orch_middleware.append(
            TrajectoryMiddleware(resources.trajectory, agent_path=["orchestrator"])
        )

    # Model
    model_plain = create_chat_model(
        model=config.model,
        temperature=config.temperature,
        model_config=resources.orch_model_config,
    )
    if config.disable_tool_binding:
        model_with_tools = model_plain
    else:
        model_with_tools = model_plain.bind_tools(
            [t.to_openai_schema() for t in tools]
        )

    # Structured output
    output_schema = wiring.output_schema
    output_prompt_text = ""
    if config.output is not None:
        output_prompt_text = load_prompt_template(config.output.prompt)

    # Termination
    should_terminate = wiring.should_terminate or _orchestrator_should_terminate

    # Retry
    retry_cfg = config.retry

    return SimpleAgentLoop(
        model=model_with_tools,
        tools=tools,
        system_prompt="",  # Managed by DynamicContextMiddleware
        middleware=orch_middleware,
        output_schema=output_schema,
        output_prompt=output_prompt_text,
        synthesize_retries=hooks_safe.synthesize_max_retries,
        should_terminate=should_terminate,
        retry_max_attempts=retry_cfg.max_attempts,
        retry_initial_interval=retry_cfg.initial_interval,
        retry_backoff_factor=retry_cfg.backoff_factor,
    )


# ---------------------------------------------------------------------------
# build_agent_system() — the single canonical builder
# ---------------------------------------------------------------------------


def build_agent_system(
    scenario_name: str,
    scenario_config: ScenarioConfig,
    system_config: SystemConfig | None = None,
    *,
    thread_id: str | None = None,
    tools_dir: Path | str | None = None,
    knowledge_base_dir: str | None = None,
) -> AgentSystem:
    """Build an AgentSystem from a scenario name and config.

    This is the single canonical entry point for constructing any agent
    system.  It uses the Scenario protocol to wire domain-specific behavior.

    The build is decomposed into four phases:

    1. ``_create_platform_resources`` — vault, trajectory, tool registry
    2. ``_create_worker_infrastructure`` — runtime, worker factory
    3. ``_assemble_orchestrator_tools`` — SDK + scenario + registry tools
    4. ``_build_orchestrator_loop`` — middleware stack + LLM + loop

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

    # 2. Create platform resources
    resources = _create_platform_resources(
        scenario_name,
        scenario_config,
        system_config,
        thread_id=thread_id,
        tools_dir=tools_dir,
        knowledge_base_dir=knowledge_base_dir,
    )

    # 2b. Validate cross-references
    if system_config is not None:
        from agentm.config.validator import validate_references

        errors = validate_references(system_config, scenario_config, resources.tool_registry)
        if errors:
            raise ConfigError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    # 3. Call scenario.setup()
    ctx = SetupContext(
        vault=resources.vault,
        trajectory=resources.trajectory,
        tool_registry=resources.tool_registry,
    )
    wiring = scenario.setup(ctx)

    # 4. Create worker infrastructure
    runtime, worker_factory = _create_worker_infrastructure(
        scenario_config, resources, wiring,
    )

    # 5. Assemble orchestrator tools
    tools = _assemble_orchestrator_tools(
        scenario_config, resources, wiring, runtime, worker_factory,
    )

    # 6. Build orchestrator loop
    orch_loop = _build_orchestrator_loop(
        scenario_config, resources, wiring, tools,
    )

    return AgentSystem(
        orch_loop,
        scenario_config=scenario_config,
        runtime=runtime,
        trajectory=resources.trajectory,
        thread_id=resources.thread_id,
    )
