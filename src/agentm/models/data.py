"""Data structures (dataclasses) for AgentM.

SDK data types are defined here. Domain-specific types live in their
canonical locations under ``scenarios/``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from agentm.models.enums import AgentRunStatus


# --- Compression (SDK) ---


@dataclass
class CompressionRef:
    """Reference from a compressed state to its source checkpoint range."""

    from_checkpoint_id: str
    to_checkpoint_id: str
    layer: Literal["sub_agent", "orchestrator"]
    step_count: int
    reason: str


@dataclass
class SubAgentMessageSummary:
    """Compressed summary of Sub-Agent message history."""

    summary: str
    latest_raw_data: dict = field(default_factory=dict)


# --- Phase Definition (SDK) ---


@dataclass
class PhaseDefinition:
    """Definition of a phase in a system's workflow."""

    name: str
    description: str
    handler: object  # Callable, kept as object for stub phase
    next_phases: list[str] = field(default_factory=list)
    on_enter: Optional[object] = None
    on_exit: Optional[object] = None


# --- Scenario Tool Bundle (SDK) ---


@dataclass
class ScenarioToolBundle:
    """Tools and overrides returned by a strategy's ``create_scenario_tools``.

    Allows each scenario to supply its own orchestrator tools, worker tools,
    and format_context override without the builder needing scenario-specific
    knowledge.
    """

    orchestrator_tools: dict[str, Any] = field(default_factory=dict)
    worker_tools: list[Any] = field(default_factory=list)
    format_context_override: Optional[Callable[..., str]] = None


# --- TaskManager (SDK) ---


@dataclass
class ManagedTask:
    """A single Sub-Agent execution managed by TaskManager."""

    task_id: str
    agent_id: str
    instruction: str
    hypothesis_id: Optional[str] = None
    status: AgentRunStatus = AgentRunStatus.RUNNING
    current_step: int = 0
    max_steps: Optional[int] = None
    timeout: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    result: Optional[dict] = None
    error_summary: Optional[str] = None
    asyncio_task: Optional[asyncio.Task] = field(default=None, repr=False)  # type: ignore[type-arg]
    events_buffer: list[dict] = field(default_factory=list)
    subgraph_config: Optional[dict] = field(default=None, repr=False)
    reported: bool = False
    trajectory_self_reported: bool = (
        False  # True when the subgraph records its own trajectory events
    )
    pending_instructions: list[str] = field(default_factory=list)
    parent_thread_id: Optional[str] = None
    parent_dispatch_step: Optional[int] = None
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    last_tool_call: Optional[dict] = None
    llm_call_count: int = 0


# --- Orchestrator Hooks (SDK) ---


@dataclass
class OrchestratorHooks:
    """Orchestrator behavior customization points returned by strategy.

    Strategies return an instance of this dataclass from ``orchestrator_hooks()``
    to control think-stall detection, context injection policy, and synthesize
    retry behavior.  Default values provide sensible generic behavior.
    """

    # Think-stall detection
    think_stall_enabled: bool = True
    think_stall_limit: int = 3
    think_stall_warning: str = (
        "THINK-STALL WARNING: You have called only `think` for the "
        "last {streak} rounds without taking any action. "
        "You MUST call an action tool NOW. "
        "Do NOT call think again until you have taken an action."
    )

    # Context injection policy
    skip_context_on_think_only: bool = False

    # Synthesize retries
    synthesize_max_retries: int = 2


# --- Trajectory (SDK) ---


@dataclass
class TaskTraceRef:
    """Links a Sub-Agent's checkpoint chain to the Orchestrator's timeline."""

    task_id: str
    agent_id: str
    agent_thread_id: str
    parent_thread_id: str
    parent_dispatch_step: int
    hypothesis_id: Optional[str] = None
