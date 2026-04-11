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

from agentm.config.schema import ModelConfig, ScenarioConfig, SystemConfig, create_chat_model
from agentm.core.prompt import load_prompt_template
from agentm.core.tool_registry import ToolRegistry
from agentm.core.trajectory import TrajectoryCollector
from agentm.exceptions import ConfigError
from agentm.harness.loops.simple import SimpleAgentLoop
from agentm.harness.middleware import (
    DynamicContextMiddleware,
    LoopDetectionMiddleware,
    MiddlewareBase,
    PrefillMiddleware,
    SkillMiddleware,
    TrajectoryMiddleware,
)
from agentm.harness.runtime import AgentRuntime
from agentm.harness.scenario import OrchestratorHooks, ScenarioWiring, SetupContext, get_scenario
from agentm.core.tool import Tool, ToolCallable, tool_from_function
from agentm.harness.types import AgentInput, AgentOutput, AgentResult, Message, RunConfig
from agentm.harness.worker_factory import WorkerLoopFactory
from agentm.tools.orchestrator import create_orchestrator_tools
from agentm.tools.vault.store import MarkdownVault


# ---------------------------------------------------------------------------
# Decision-based termination (replaces <decision> routing in the LangGraph
# orchestrator). The LLM emits <decision>finalize</decision> when it wants
# to stop; otherwise it keeps calling tools.
# ---------------------------------------------------------------------------

_DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)

# TODO: move to agentm.defaults once orchestrator prompt is added there
_DEFAULT_ORCHESTRATOR_PROMPT = "You are an agent orchestrator."


def _generate_run_id(scenario_name: str) -> str:
    """Generate a unique run ID with timestamp and short UUID suffix."""
    return f"{scenario_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


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


# TODO: move to agentm.defaults once tools_dir default is added there
_CANDIDATE_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "tools"
_DEFAULT_TOOLS_DIR: Path | None = _CANDIDATE_TOOLS_DIR if _CANDIDATE_TOOLS_DIR.is_dir() else None


# ---------------------------------------------------------------------------
# AgentSystemContext — immutable, reusable resources shared across runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSystemContext:
    """Immutable, reusable resources shared across multiple agent runs.

    Created once by ``build_system_context()`` and passed to
    ``create_agent_run()`` for each case.  Everything stored here is
    safe to share between concurrent runs — no mutable per-run state.
    """

    scenario_name: str
    scenario_config: ScenarioConfig
    system_config: SystemConfig | None
    tool_registry: ToolRegistry
    vault: MarkdownVault | None
    memory_tools: dict[str, ToolCallable]
    orch_model_config: ModelConfig | None
    worker_model_config: ModelConfig | None
    wiring: ScenarioWiring
    # Resolved directory paths (avoid re-resolving each run)
    _checkpoint_db_path: str = ""


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
    checkpoint_db_path: str = ""


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
    resolved_tools_dir: Path | None = (
        Path(tools_dir) if tools_dir is not None else _DEFAULT_TOOLS_DIR
    )
    resolved_kb_dir = (
        knowledge_base_dir if knowledge_base_dir is not None else "./knowledge"
    )
    resolved_thread_id = thread_id if thread_id else str(uuid.uuid4())

    # Tool registry
    tool_registry = ToolRegistry()
    if resolved_tools_dir is not None:
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

    # Checkpoint DB path (computed once, reused for trajectory creation)
    checkpoint_db_path = ""
    if system_config is not None and system_config.storage.checkpointer.backend == "sqlite":
        db_url = system_config.storage.checkpointer.url or "./checkpoints.db"
        checkpoint_db_path = str(Path(db_url).resolve())

    # Trajectory
    trajectory: TrajectoryCollector | None = None
    if system_config is not None and system_config.debug.trajectory.enabled:
        run_id = _generate_run_id(scenario_name)
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
        checkpoint_db_path=checkpoint_db_path,
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
    worker_middleware = list(wiring.worker_middleware or [])
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
        static_system_prompt = _DEFAULT_ORCHESTRATOR_PROMPT

    max_rounds: int = config.max_rounds
    format_context = wiring.format_context

    # Middleware stack
    orch_middleware: list[MiddlewareBase] = []

    # Hooks should never be None after __post_init__, but mypy doesn't know that
    hooks_safe = hooks or OrchestratorHooks()

    # 1. Replace system message with base prompt (runs first)
    orch_middleware.append(DynamicContextMiddleware(
        base_system_prompt=static_system_prompt,
    ))

    # 2. Loop detection (may inject human warning messages)
    ld = config.loop_detection
    if ld.enabled and hooks_safe.think_stall_enabled:
        orch_middleware.append(
            LoopDetectionMiddleware(
                threshold=ld.threshold,
                window_size=ld.window_size,
                think_stall_limit=ld.think_stall_limit,
            )
        )

    # 3. Compression (preserves system messages, compresses conversation)
    compression_cfg = config.compression
    if compression_cfg is not None and compression_cfg.enabled:
        from agentm.harness.middleware import create_compression_middleware

        orch_middleware.append(
            create_compression_middleware(compression_cfg, resources.orch_model_config)
        )

    # 4. Skill injection into system message
    if config.skills and resources.vault is not None:
        orch_middleware.append(SkillMiddleware(resources.vault, config.skills))

    # 5. Scenario middleware (e.g. SanitizerMW — may inject human messages)
    if wiring.orchestrator_middleware:
        orch_middleware.extend(wiring.orchestrator_middleware)

    # 6. Trajectory recording
    if resources.trajectory is not None:
        orch_middleware.append(
            TrajectoryMiddleware(resources.trajectory, agent_path=["orchestrator"])
        )

    # 7. Prefill — MUST be last so it's never displaced by injected messages
    if config.prefill:
        orch_middleware.append(PrefillMiddleware(
            format_context_fn=format_context if callable(format_context) else lambda: "",
            max_rounds=max_rounds,
        ))

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
        synthesis_model=model_plain,
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
# build_system_context() — create reusable context (call once)
# ---------------------------------------------------------------------------


def build_system_context(
    scenario_name: str,
    scenario_config: ScenarioConfig,
    system_config: SystemConfig | None = None,
    *,
    tools_dir: Path | str | None = None,
    knowledge_base_dir: str | None = None,
) -> AgentSystemContext:
    """Create reusable, immutable context -- call once, use for many runs.

    Performs the expensive work: scenario discovery, tool registry loading,
    vault initialisation, config validation, and scenario wiring.  The
    returned ``AgentSystemContext`` can be passed to ``create_agent_run()``
    repeatedly without redoing any of this work.

    Args:
        scenario_name: Registered scenario name (e.g. "hypothesis_driven").
        scenario_config: Full scenario configuration.
        system_config: Optional system-level config (models, storage, debug).
        tools_dir: Directory containing tool YAML definitions.
        knowledge_base_dir: Path to the knowledge base (vault root).
    """
    from agentm.scenarios import discover as _discover_scenarios

    _discover_scenarios()

    scenario = get_scenario(scenario_name)

    # Build platform resources (trajectory=None since context is immutable)
    resources = _create_platform_resources(
        scenario_name,
        scenario_config,
        system_config,
        thread_id=None,
        tools_dir=tools_dir,
        knowledge_base_dir=knowledge_base_dir,
    )

    # Validate cross-references once
    if system_config is not None:
        from agentm.config.validator import validate_references

        errors = validate_references(system_config, scenario_config, resources.tool_registry)
        if errors:
            raise ConfigError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    # Scenario wiring (immutable for a given config)
    setup_ctx = SetupContext(
        vault=resources.vault,
        trajectory=None,  # No per-run trajectory in the shared context
        tool_registry=resources.tool_registry,
        config=scenario_config,
    )
    wiring = scenario.setup(setup_ctx)

    return AgentSystemContext(
        scenario_name=scenario_name,
        scenario_config=scenario_config,
        system_config=system_config,
        tool_registry=resources.tool_registry,
        vault=resources.vault,
        memory_tools=resources.memory_tools,
        orch_model_config=resources.orch_model_config,
        worker_model_config=resources.worker_model_config,
        wiring=wiring,
        _checkpoint_db_path=resources.checkpoint_db_path,
    )


# ---------------------------------------------------------------------------
# create_agent_run() — cheap, per-run agent system from a context
# ---------------------------------------------------------------------------


def create_agent_run(ctx: AgentSystemContext) -> AgentSystem:
    """Create a fresh agent system from an existing context -- cheap, per-run.

    Only creates the mutable, per-run resources: ``TrajectoryCollector``,
    ``AgentRuntime``, ``WorkerLoopFactory``, and ``SimpleAgentLoop``.
    """
    thread_id = str(uuid.uuid4())

    # Per-run trajectory
    trajectory: TrajectoryCollector | None = None
    if ctx.system_config is not None and ctx.system_config.debug.trajectory.enabled:
        run_id = _generate_run_id(ctx.scenario_name)
        trajectory = TrajectoryCollector(
            run_id=run_id,
            output_dir=ctx.system_config.debug.trajectory.output_dir,
            thread_id=thread_id,
            checkpoint_db=ctx._checkpoint_db_path,
        )

    # Build a _PlatformResources adapter for the existing phase helpers
    resources = _PlatformResources(
        tool_registry=ctx.tool_registry,
        vault=ctx.vault,
        trajectory=trajectory,
        memory_tools=ctx.memory_tools,
        orch_model_config=ctx.orch_model_config,
        worker_model_config=ctx.worker_model_config,
        thread_id=thread_id,
        checkpoint_db_path=ctx._checkpoint_db_path,
    )

    # Inject per-run trajectory into scenario wiring so closures see it
    ctx.wiring.bind_trajectory(trajectory)

    runtime, worker_factory = _create_worker_infrastructure(
        ctx.scenario_config, resources, ctx.wiring,
    )
    tools = _assemble_orchestrator_tools(
        ctx.scenario_config, resources, ctx.wiring, runtime, worker_factory,
    )
    orch_loop = _build_orchestrator_loop(
        ctx.scenario_config, resources, ctx.wiring, tools,
    )

    return AgentSystem(
        orch_loop,
        scenario_config=ctx.scenario_config,
        runtime=runtime,
        trajectory=trajectory,
        thread_id=thread_id,
    )


# ---------------------------------------------------------------------------
# build_agent_system() — convenience wrapper (backward compatible)
# ---------------------------------------------------------------------------


def build_agent_system(
    scenario_name: str,
    scenario_config: ScenarioConfig,
    system_config: SystemConfig | None = None,
    *,
    thread_id: str | None = None,  # noqa: ARG001 — kept for backward compatibility
    tools_dir: Path | str | None = None,
    knowledge_base_dir: str | None = None,
) -> AgentSystem:
    """Build an AgentSystem from a scenario name and config.

    Convenience wrapper: builds context + creates a run in one call.
    All existing callers continue to work unchanged.

    Delegates to ``build_system_context()`` for the expensive, reusable
    setup and ``create_agent_run()`` for the cheap, per-run resources.

    Args:
        scenario_name: Registered scenario name (e.g. "hypothesis_driven").
        scenario_config: Full scenario configuration.
        system_config: Optional system-level config (models, storage, debug).
        thread_id: Deprecated — each run now generates its own thread ID.
            Kept for backward compatibility; ignored.
        tools_dir: Directory containing tool YAML definitions.
        knowledge_base_dir: Path to the knowledge base (vault root).
    """
    ctx = build_system_context(
        scenario_name,
        scenario_config,
        system_config,
        tools_dir=tools_dir,
        knowledge_base_dir=knowledge_base_dir,
    )
    return create_agent_run(ctx)
