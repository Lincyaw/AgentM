"""State schema registry for system-type-specific state classes.

Registries are populated by scenario ``register()`` functions called
via ``agentm.scenarios.discover()``.  The SDK core never imports from
``scenarios/`` directly.

SDK-only state schemas (SequentialDiagnosisState, DecisionTreeState)
are registered at module load time since they live in ``models/state.py``.
"""

from __future__ import annotations

from agentm.models.state import DecisionTreeState, SequentialDiagnosisState

STATE_SCHEMAS: dict[str, type] = {
    "sequential": SequentialDiagnosisState,
    "decision_tree": DecisionTreeState,
}


def register_state(system_type: str, state_schema: type) -> None:
    """Register a state schema for a system type.

    Called by scenario ``register()`` functions.  Uses ``setdefault``
    semantics — first registration wins.
    """
    STATE_SCHEMAS.setdefault(system_type, state_schema)


def get_state_schema(system_type: str) -> type:
    """Look up the state schema for a system type.

    Raises ValueError if the system type is not registered.
    """
    if system_type not in STATE_SCHEMAS:
        available = list(STATE_SCHEMAS.keys())
        raise ValueError(f"Unknown system type: {system_type}. Available: {available}")
    return STATE_SCHEMAS[system_type]
