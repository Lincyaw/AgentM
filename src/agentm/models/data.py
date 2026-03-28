"""Data structures (dataclasses) for AgentM.

SDK data types are defined here. Domain-specific types live in their
canonical locations under ``scenarios/``.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- Orchestrator Hooks (SDK) ---


@dataclass
class OrchestratorHooks:
    """Orchestrator behavior customization points returned by strategy.

    Strategies return an instance of this dataclass from ``orchestrator_hooks()``
    to control think-stall detection and synthesize retry behavior.
    Default values provide sensible generic behavior.
    """

    # Think-stall detection
    think_stall_enabled: bool = True

    # Synthesize retries
    synthesize_max_retries: int = 2
