"""Adapters for bridging AgentM subsystems to the Harness protocols."""
from __future__ import annotations

from typing import TYPE_CHECKING

from agentm.harness.types import AgentEvent

if TYPE_CHECKING:
    from agentm.core.trajectory import TrajectoryCollector


class TrajectoryEventAdapter:
    """Adapt AgentRuntime events to TrajectoryCollector.record() calls.

    Implements the EventHandler protocol so it can be passed directly
    to AgentRuntime(event_handler=adapter).
    """

    def __init__(
        self,
        trajectory: TrajectoryCollector,
        root_agent_id: str = "orchestrator",
    ) -> None:
        self._trajectory = trajectory
        self._root_agent_id = root_agent_id

    async def on_event(self, event: AgentEvent) -> None:
        """Forward an AgentEvent to the TrajectoryCollector."""
        await self._trajectory.record(
            event_type=event.type,
            agent_path=[self._root_agent_id, event.agent_id],
            data=event.data,
            node_name=event.data.get("node_name", ""),
        )
