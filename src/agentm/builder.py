"""AgentSystem and AgentSystemBuilder — unified entry point for all agent systems.

All methods are stubs — raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from agentm.config.schema import ScenarioConfig


class AgentSystem:
    """Unified interface for all agent systems."""

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent system with the given input. Returns final state."""
        raise NotImplementedError

    async def stream(self, input_data: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream events from the agent system execution."""
        raise NotImplementedError


class AgentSystemBuilder:
    """Unified entry point for building any agent system.

    Internally selects the appropriate architecture based on system_type:
    - ReAct-based (create_react_agent): For exploratory, non-linear scenarios like RCA
    - StateGraph-based (custom graph with phase nodes): For linear, deterministic scenarios
    """

    @staticmethod
    def build(system_type: str, config: ScenarioConfig) -> AgentSystem:
        """Build an AgentSystem from a system type and scenario config."""
        raise NotImplementedError
