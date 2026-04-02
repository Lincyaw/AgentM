"""Agent Harness SDK — core types, protocols, and runtime for agent management."""
from __future__ import annotations

from agentm.harness.adapters import TrajectoryEventAdapter
from agentm.harness.handle import AgentHandle
from agentm.harness.protocols import (
    AgentLoop,
    CheckpointStore,
    EventHandler,
)
from agentm.harness.runtime import AgentRuntime
from agentm.harness.scenario import (
    Scenario,
    ScenarioWiring,
    SetupContext,
    clear_scenarios,
    get_scenario,
    list_scenarios,
    register_scenario,
)
from agentm.core.tool import Tool, tool, tool_from_function
from agentm.harness.types import (
    AgentEvent,
    AgentInfo,
    AgentResult,
    AgentStatus,
    LoopContext,
    RunConfig,
)

__all__ = [
    # Enums
    "AgentStatus",
    # Data types
    "RunConfig",
    "AgentResult",
    "AgentEvent",
    "AgentInfo",
    "LoopContext",
    # Tool
    "Tool",
    "tool",
    "tool_from_function",
    # Scenario
    "Scenario",
    "SetupContext",
    "ScenarioWiring",
    "register_scenario",
    "get_scenario",
    "list_scenarios",
    "clear_scenarios",
    # Protocols
    "AgentLoop",
    "CheckpointStore",
    "EventHandler",
    # Classes
    "AgentHandle",
    "AgentRuntime",
    # Adapters
    "TrajectoryEventAdapter",
]
