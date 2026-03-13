"""Strategy registry — maps system-type names to ReasoningStrategy instances.

Default strategies are lazily loaded from ``scenarios/`` on first access.
"""

from __future__ import annotations

from typing import Any

from agentm.core.strategy import ReasoningStrategy

# Singleton instances — initially empty, populated lazily.
_STRATEGY_INSTANCES: dict[str, ReasoningStrategy[Any]] = {}
_defaults_loaded = False


def _ensure_defaults() -> None:
    """Lazily import and register default strategies from scenarios."""
    global _defaults_loaded
    if _defaults_loaded:
        return
    _defaults_loaded = True

    from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy
    from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy

    _STRATEGY_INSTANCES.setdefault(
        "hypothesis_driven", HypothesisDrivenStrategy()
    )
    _STRATEGY_INSTANCES.setdefault(
        "memory_extraction", MemoryExtractionStrategy()
    )


def get_strategy(system_type: str) -> ReasoningStrategy[Any]:
    """Look up the strategy for a system type.

    Raises ``ValueError`` if the system type has no registered strategy.
    """
    _ensure_defaults()
    strategy = _STRATEGY_INSTANCES.get(system_type)
    if strategy is None:
        available = list(_STRATEGY_INSTANCES.keys())
        raise ValueError(
            f"No strategy registered for system type: {system_type!r}. "
            f"Available: {available}"
        )
    return strategy


def register_strategy(
    system_type: str, strategy: ReasoningStrategy[Any]
) -> None:
    """Register a custom strategy for a system type.

    Overwrites any existing registration for the same key.
    """
    _STRATEGY_INSTANCES[system_type] = strategy


def list_strategies() -> list[str]:
    """Return all registered system-type names."""
    _ensure_defaults()
    return list(_STRATEGY_INSTANCES.keys())
