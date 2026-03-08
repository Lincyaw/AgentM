"""State schema registry for system-type-specific state classes.

The registry mapping is fully implemented (value object).
The lookup function is fully implemented.
"""

from __future__ import annotations

from agentm.models.state import (
    DecisionTreeState,
    HypothesisDrivenState,
    MemoryExtractionState,
    SequentialDiagnosisState,
)

STATE_SCHEMAS: dict[str, type] = {
    "hypothesis_driven": HypothesisDrivenState,
    "sequential": SequentialDiagnosisState,
    "memory_extraction": MemoryExtractionState,
    "decision_tree": DecisionTreeState,
}


def get_state_schema(system_type: str) -> type:
    """Look up the state schema for a system type.

    Raises ValueError if the system type is not registered.
    """
    if system_type not in STATE_SCHEMAS:
        available = list(STATE_SCHEMAS.keys())
        raise ValueError(f"Unknown system type: {system_type}. Available: {available}")
    return STATE_SCHEMAS[system_type]
