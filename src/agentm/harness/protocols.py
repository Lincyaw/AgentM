"""Protocol definitions for the Agent Harness SDK."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from agentm.harness.types import AgentEvent, AgentResult, Message, RunConfig


@runtime_checkable
class AgentLoop(Protocol):
    """Protocol for a single agent's conversation loop.

    An AgentLoop encapsulates the cycle:
        receive input -> call LLM -> execute tools -> repeat -> return result

    Implementations may use LangGraph, a simple while-loop, or any other
    mechanism. The SDK does not care -- it only interacts via this interface.

    stream() is the primary method; run() is a convenience wrapper that
    iterates the stream and returns the final result.
    """

    async def run(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AgentResult:
        """Run the agent loop to completion.

        Default implementation iterates stream() and extracts the result
        from the final "complete" event. Implementations may override for
        efficiency.
        """
        ...

    def stream(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent loop, yielding events as they occur.

        This is the primary execution method. The runtime always uses
        stream() internally to capture events for EventHandler forwarding.

        The last event MUST be type="complete" with data={"result": AgentResult}.

        Yields AgentEvent instances for: llm_start, llm_end, tool_start,
        tool_end, inject, error, complete.
        """
        ...

    def inject(self, message: str) -> None:
        """Inject a message into the agent's inbox.

        The message will be consumed before the next LLM call.
        Can be called concurrently while the agent loop is running
        (safe under asyncio single-thread model).
        """
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Persist and recover agent conversation state.

    Implementations: MemoryCheckpointStore, SQLiteCheckpointStore,
    or LangGraphCheckpointAdapter (wraps LangGraph's BaseCheckpointSaver).
    """

    async def save(self, agent_id: str, state: dict[str, Any]) -> str:
        """Save a state snapshot. Returns checkpoint_id."""
        ...

    async def load(
        self, agent_id: str, *, checkpoint_id: str | None = None
    ) -> dict[str, Any] | None:
        """Load state. None checkpoint_id = latest. Returns None if not found."""
        ...

    async def list_checkpoints(
        self, agent_id: str
    ) -> list[dict[str, Any]]:
        """List available checkpoints for an agent (newest first)."""
        ...


@runtime_checkable
class EventHandler(Protocol):
    """Receive streaming events from all agents in the runtime.

    Implementations: TrajectoryCollector adapter, WebSocket broadcaster,
    debug console renderer.
    """

    async def on_event(self, event: AgentEvent) -> None:
        """Called for every event emitted by any agent in the runtime."""
        ...


@runtime_checkable
class NoteReader(Protocol):
    """Protocol for reading notes/skills from a store.

    Decouples middleware (e.g. SkillMiddleware) from concrete vault
    implementations such as MarkdownVault.
    """

    def read(self, path: str) -> dict[str, Any] | None: ...
