"""State schema registry for system-type-specific state classes.

Default state schemas are lazily loaded from ``scenarios/`` on first access.
"""

from __future__ import annotations

STATE_SCHEMAS: dict[str, type] = {}
_defaults_loaded = False


def _ensure_defaults() -> None:
    """Lazily import and register default state schemas from scenarios."""
    global _defaults_loaded
    if _defaults_loaded:
        return
    _defaults_loaded = True

    from agentm.scenarios.rca.state import HypothesisDrivenState
    from agentm.scenarios.memory_extraction.state import MemoryExtractionState
    from agentm.models.state import DecisionTreeState, SequentialDiagnosisState

    STATE_SCHEMAS.setdefault("hypothesis_driven", HypothesisDrivenState)
    STATE_SCHEMAS.setdefault("sequential", SequentialDiagnosisState)
    STATE_SCHEMAS.setdefault("memory_extraction", MemoryExtractionState)
    STATE_SCHEMAS.setdefault("decision_tree", DecisionTreeState)


def get_state_schema(system_type: str) -> type:
    """Look up the state schema for a system type.

    Raises ValueError if the system type is not registered.
    """
    _ensure_defaults()
    if system_type not in STATE_SCHEMAS:
        available = list(STATE_SCHEMAS.keys())
        raise ValueError(f"Unknown system type: {system_type}. Available: {available}")
    return STATE_SCHEMAS[system_type]
