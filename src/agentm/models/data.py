"""Data structures (dataclasses) for AgentM.

SDK data types are defined here. Domain-specific types live in their
canonical locations under ``scenarios/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    metadata: dict = field(default_factory=dict)
