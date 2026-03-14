"""Strategy registry — maps system-type names to ReasoningStrategy instances.

Registries are populated by scenario ``register()`` functions called
via ``agentm.scenarios.discover()``.  The SDK core never imports from
``scenarios/`` directly.
"""

from __future__ import annotations

from typing import Any

from agentm.core.strategy import ReasoningStrategy

# Singleton instances — initially empty, populated by scenario register().
_STRATEGY_INSTANCES: dict[str, ReasoningStrategy[Any]] = {}


def get_strategy(system_type: str) -> ReasoningStrategy[Any]:
    """Look up the strategy for a system type.

    Raises ``ValueError`` if the system type has no registered strategy.
    """
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
    return list(_STRATEGY_INSTANCES.keys())
