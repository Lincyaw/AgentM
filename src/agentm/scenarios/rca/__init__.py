"""Hypothesis-driven Root Cause Analysis scenario."""


def register() -> None:
    """Register RCA types with SDK registries."""
    from agentm.core.strategy_registry import register_strategy
    from agentm.core.state_registry import STATE_SCHEMAS, _ensure_defaults as _ensure_state_defaults
    from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy
    from agentm.scenarios.rca.state import HypothesisDrivenState

    # Ensure defaults are loaded first, then register/overwrite
    _ensure_state_defaults()
    STATE_SCHEMAS["hypothesis_driven"] = HypothesisDrivenState
    register_strategy("hypothesis_driven", HypothesisDrivenStrategy())
