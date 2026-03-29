"""AgentRuntime — the Harness for managing multiple agent loops."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from agentm.harness.handle import AgentHandle
from agentm.harness.protocols import (
    AgentLoop,
    CheckpointStore,
    EventHandler,
)
from agentm.harness.types import (
    AgentInfo,
    AgentResult,
    AgentStatus,
    RunConfig,
)


@dataclass
class _AgentEntry:
    """Internal bookkeeping for a spawned agent."""

    agent_id: str
    loop: AgentLoop
    task: asyncio.Task[None]
    parent_id: str | None = None
    status: AgentStatus = AgentStatus.RUNNING
    result: AgentResult | None = None
    current_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


class AgentRuntime:
    """Manages multiple AgentLoops: lifecycle, messaging, coordination.

    This is the Harness. It replaces TaskManager + AgentPool with a
    unified interface for dynamic agent management.
    """

    def __init__(
        self,
        *,
        checkpoint_store: CheckpointStore | None = None,
        event_handler: EventHandler | None = None,
        trajectory: Any | None = None,
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._event_handler = event_handler
        self._trajectory = trajectory
        self._agents: dict[str, _AgentEntry] = {}

    # --- Lifecycle ---

    async def spawn(
        self,
        agent_id: str,
        *,
        loop: AgentLoop,
        input: str,
        parent_id: str | None = None,
        config: RunConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentHandle:
        """Spawn a new agent and start its loop as an asyncio.Task."""
        if agent_id in self._agents:
            raise ValueError(f"Agent '{agent_id}' already exists")

        config = config or RunConfig()
        # Ensure agent_id is in config metadata so the loop can access it
        config.metadata.setdefault("agent_id", agent_id)

        entry = _AgentEntry(
            agent_id=agent_id,
            loop=loop,
            task=None,  # type: ignore[arg-type]  # set below
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._agents[agent_id] = entry

        # Record task_dispatch trajectory event
        if self._trajectory is not None:
            original_agent_id = (metadata or {}).get("original_agent_id", agent_id)
            self._trajectory.record_sync(
                event_type="task_dispatch",
                agent_path=[original_agent_id],
                data={
                    "task_id": agent_id,
                    "agent_id": original_agent_id,
                    "task_type": (metadata or {}).get("task_type", "unknown"),
                    "instruction": input,
                    "metadata": metadata or {},
                },
                task_id=agent_id,
            )

        task = asyncio.create_task(
            self._run_agent(entry, input, config),
            name=f"agent-{agent_id}",
        )
        entry.task = task
        return AgentHandle(self, agent_id)

    async def _run_agent(
        self, entry: _AgentEntry, input: str, config: RunConfig
    ) -> None:
        """Run an agent's stream, forwarding events and handling completion."""
        start_time = time.monotonic()
        original_agent_id = entry.metadata.get("original_agent_id", entry.agent_id)
        try:
            async for event in entry.loop.stream(input, config=config):  # type: ignore[attr-defined]
                entry.current_step = event.step
                if self._event_handler is not None:
                    await self._event_handler.on_event(event)
                if event.type == "complete":
                    result = event.data.get("result")
                    if isinstance(result, AgentResult):
                        entry.result = result
                        entry.status = result.status
                    else:
                        entry.status = AgentStatus.COMPLETED
                        entry.result = AgentResult(
                            agent_id=entry.agent_id,
                            status=AgentStatus.COMPLETED,
                            output=result,
                        )
            # Record task_complete
            duration = time.monotonic() - start_time
            if entry.result is not None:
                entry.result.duration_seconds = duration
            if self._trajectory is not None:
                self._trajectory.record_sync(
                    event_type="task_complete",
                    agent_path=[original_agent_id],
                    data={
                        "task_id": entry.agent_id,
                        "agent_id": original_agent_id,
                        "duration_seconds": duration,
                        "result": entry.result.output if entry.result else None,
                    },
                    task_id=entry.agent_id,
                )
        except asyncio.CancelledError:
            if entry.status == AgentStatus.RUNNING:
                entry.status = AgentStatus.ABORTED
                duration = time.monotonic() - start_time
                entry.result = AgentResult(
                    agent_id=entry.agent_id,
                    status=AgentStatus.ABORTED,
                    error="cancelled",
                    duration_seconds=duration,
                )
                if self._trajectory is not None:
                    self._trajectory.record_sync(
                        event_type="task_abort",
                        agent_path=[original_agent_id],
                        data={
                            "task_id": entry.agent_id,
                            "agent_id": original_agent_id,
                            "reason": "cancelled",
                        },
                        task_id=entry.agent_id,
                    )
        except Exception as exc:
            duration = time.monotonic() - start_time
            entry.status = AgentStatus.FAILED
            entry.result = AgentResult(
                agent_id=entry.agent_id,
                status=AgentStatus.FAILED,
                error=str(exc),
                duration_seconds=duration,
            )
            if self._trajectory is not None:
                self._trajectory.record_sync(
                    event_type="task_fail",
                    agent_path=[original_agent_id],
                    data={
                        "task_id": entry.agent_id,
                        "agent_id": original_agent_id,
                        "error_summary": str(exc),
                    },
                    task_id=entry.agent_id,
                )
        finally:
            entry.done_event.set()
            # Cascade: abort running children when parent terminates
            await self._cascade_children(entry.agent_id)

    async def _cascade_children(self, parent_id: str) -> None:
        """Abort all running children of the given parent."""
        children = [
            e
            for e in self._agents.values()
            if e.parent_id == parent_id and e.status == AgentStatus.RUNNING
        ]
        for child in children:
            await self.abort(child.agent_id, reason=f"parent '{parent_id}' terminated")

    async def abort(self, agent_id: str, reason: str) -> bool:
        """Abort a running agent. Returns False if already stopped.

        Cascades: also aborts all running children recursively.
        """
        entry = self._agents.get(agent_id)
        if entry is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        if entry.status != AgentStatus.RUNNING:
            return False

        entry.status = AgentStatus.ABORTED
        entry.result = AgentResult(
            agent_id=agent_id,
            status=AgentStatus.ABORTED,
            error=reason,
        )
        entry.task.cancel()
        # Wait briefly for the task to handle cancellation
        try:
            await asyncio.wait_for(asyncio.shield(entry.task), timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
        entry.done_event.set()
        # Cascade to children
        await self._cascade_children(agent_id)
        return True

    async def wait(
        self, agent_id: str, *, timeout: float | None = None
    ) -> AgentResult:
        """Block until an agent completes. Raises TimeoutError if exceeded."""
        entry = self._agents.get(agent_id)
        if entry is None:
            raise ValueError(f"Agent '{agent_id}' not found")

        if timeout is not None:
            try:
                await asyncio.wait_for(entry.done_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Agent '{agent_id}' did not complete within {timeout}s"
                )
        else:
            await entry.done_event.wait()

        assert entry.result is not None
        return entry.result

    async def wait_any(
        self,
        agent_ids: list[str] | None = None,
        *,
        timeout: float | None = None,
    ) -> list[str]:
        """Wait for any agent to complete. Returns newly completed agent_ids.

        Args:
            agent_ids: If given, only watch these agents. None = all running.
        """
        if agent_ids is None:
            targets = [
                e
                for e in self._agents.values()
                if e.status == AgentStatus.RUNNING
            ]
        else:
            targets = [
                self._agents[aid]
                for aid in agent_ids
                if aid in self._agents
            ]

        if not targets:
            return []

        # Check if any are already done
        already_done = [e.agent_id for e in targets if e.status != AgentStatus.RUNNING]
        if already_done:
            return already_done

        # Create futures that complete when any target's done_event is set
        done_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

        async def _watch(entry: _AgentEntry) -> None:
            await entry.done_event.wait()
            if not done_future.done():
                done_future.set_result(entry.agent_id)

        watchers = [asyncio.create_task(_watch(e)) for e in targets]

        try:
            if timeout is not None:
                first_id = await asyncio.wait_for(done_future, timeout=timeout)
            else:
                first_id = await done_future
        finally:
            for w in watchers:
                w.cancel()

        # Collect all completed targets (there might be more than one by now)
        completed = [
            e.agent_id
            for e in targets
            if e.status != AgentStatus.RUNNING
        ]
        # Ensure the first one is at least included
        if first_id not in completed:
            completed.append(first_id)
        return completed

    # --- Communication ---

    async def send(self, to: str, message: str) -> None:
        """Send a message to a running agent's inbox."""
        entry = self._agents.get(to)
        if entry is None:
            raise ValueError(f"Agent '{to}' not found")
        entry.loop.inject(message)

    # --- Status ---

    def get_status(self) -> dict[str, AgentInfo]:
        """Snapshot of all agents: status, duration, step count, result."""
        result: dict[str, AgentInfo] = {}
        for agent_id, entry in self._agents.items():
            result[agent_id] = AgentInfo(
                agent_id=agent_id,
                status=entry.status,
                parent_id=entry.parent_id,
                current_step=entry.current_step,
                metadata=entry.metadata,
                result=entry.result,
            )
        return result

    def get_result(self, agent_id: str) -> AgentResult | None:
        """Get result of a completed agent, or None if still running."""
        entry = self._agents.get(agent_id)
        if entry is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        if entry.status == AgentStatus.RUNNING:
            return None
        return entry.result

    def get_running_ids(self) -> list[str]:
        """List agent_ids of currently running agents."""
        return [
            agent_id
            for agent_id, entry in self._agents.items()
            if entry.status == AgentStatus.RUNNING
        ]
