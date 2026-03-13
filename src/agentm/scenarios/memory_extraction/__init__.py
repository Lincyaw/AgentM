"""Cross-task knowledge extraction scenario."""


def register() -> None:
    """Register memory-extraction types with SDK registries."""
    from agentm.core.strategy_registry import register_strategy
    from agentm.core.state_registry import STATE_SCHEMAS, _ensure_defaults as _ensure_state_defaults
    from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy
    from agentm.scenarios.memory_extraction.state import MemoryExtractionState

    # Ensure defaults are loaded first, then register/overwrite
    _ensure_state_defaults()
    STATE_SCHEMAS["memory_extraction"] = MemoryExtractionState
    register_strategy("memory_extraction", MemoryExtractionStrategy())
