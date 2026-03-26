"""AgentHandle — convenience wrapper for interacting with a spawned agent."""
from __future__ import annotations

from typing import TYPE_CHECKING

from agentm.harness.types import AgentResult, AgentStatus

if TYPE_CHECKING:
    from agentm.harness.runtime import AgentRuntime


class AgentHandle:
    """Reference to a running agent within the runtime.

    Convenience wrapper -- all operations delegate to AgentRuntime.
    """

    def __init__(self, runtime: AgentRuntime, agent_id: str) -> None:
        self._runtime = runtime
        self.agent_id = agent_id

    @property
    def status(self) -> AgentStatus:
        """Current status of the agent."""
        info = self._runtime.get_status().get(self.agent_id)
        if info is None:
            raise ValueError(f"Agent '{self.agent_id}' not found in runtime")
        return info.status

    @property
    def result(self) -> AgentResult | None:
        """Result of the agent, or None if still running."""
        return self._runtime.get_result(self.agent_id)

    async def wait(self, *, timeout: float | None = None) -> AgentResult:
        """Block until the agent completes. Raises TimeoutError if exceeded."""
        return await self._runtime.wait(self.agent_id, timeout=timeout)

    async def send(self, message: str) -> None:
        """Send a message to this agent's inbox."""
        await self._runtime.send(self.agent_id, message)

    async def abort(self, reason: str) -> bool:
        """Abort this agent. Returns False if already stopped."""
        return await self._runtime.abort(self.agent_id, reason=reason)
