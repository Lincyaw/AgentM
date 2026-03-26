"""Adapters for bridging AgentM subsystems to the Harness protocols."""
from __future__ import annotations

from typing import Any

from agentm.harness.types import AgentEvent


class TrajectoryEventAdapter:
    """Adapt AgentRuntime events to TrajectoryCollector.record() calls.

    Implements the EventHandler protocol so it can be passed directly
    to AgentRuntime(event_handler=adapter).
    """

    def __init__(self, trajectory: Any) -> None:
        self._trajectory = trajectory

    async def on_event(self, event: AgentEvent) -> None:
        """Forward an AgentEvent to the TrajectoryCollector."""
        await self._trajectory.record(
            event_type=event.type,
            agent_path=["orchestrator", event.agent_id],
            data=event.data,
            node_name=event.data.get("node_name", ""),
        )
