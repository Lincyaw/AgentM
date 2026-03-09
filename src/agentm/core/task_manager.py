"""TaskManager for async Sub-Agent lifecycle management."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage

from agentm.config.schema import RetryConfig
from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus


class TaskManager:
    """Manages asynchronously executing Sub-Agent subgraphs.

    Provides the backing implementation for Orchestrator tools:
    - dispatch_agent -> submit()
    - check_tasks   -> get_all_status()
    - inject_instruction -> inject()
    - abort_task    -> abort()

    A shared ``_completion_event`` (asyncio.Event) is fired whenever any task
    finishes, fails, or is aborted.  ``get_all_status`` and ``wait_for_task``
    await this event instead of polling, giving precise wake-up on completion.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ManagedTask] = {}
        self._broadcast_callback: Callable[..., Any] | None = None
        self._trajectory: Any | None = None  # TrajectoryCollector (avoid circular import)
        self._completion_event = asyncio.Event()

    def set_trajectory(self, trajectory: Any) -> None:
        """Set the TrajectoryCollector for event recording."""
        self._trajectory = trajectory

    def set_broadcast_callback(self, callback: Callable[..., Any]) -> None:
        """Set the WebSocket broadcast callback."""
        self._broadcast_callback = callback

    @property
    def trajectory(self) -> Any | None:
        """Access the TrajectoryCollector (read-only)."""
        return self._trajectory

    async def submit(
        self,
        agent_id: str,
        instruction: str,
        task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
        hypothesis_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a new task for a Sub-Agent. Returns task_id."""
        task_id = str(uuid.uuid4())

        subgraph = kwargs.get("subgraph")
        config: dict[str, Any] = kwargs.get("config", {})
        retry_config: RetryConfig = kwargs.get("retry_config", RetryConfig())

        managed = ManagedTask(
            task_id=task_id,
            agent_id=agent_id,
            instruction=instruction,
            hypothesis_id=hypothesis_id,
            started_at=datetime.now().isoformat(),
            subgraph_config=config,
        )

        if subgraph is not None:
            managed.asyncio_task = asyncio.create_task(
                self._execute_agent(managed, subgraph, config, retry_config)
            )

        self._tasks[task_id] = managed

        if self._trajectory is not None:
            await self._trajectory.record(
                event_type="task_dispatch",
                agent_path=[agent_id],
                data={
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "task_type": task_type,
                    "instruction_preview": instruction[:200],
                    "hypothesis_id": hypothesis_id,
                },
                task_id=task_id,
            )

        return task_id

    async def get_all_status(self, wait_seconds: float = 0) -> dict[str, Any]:
        """Get status of all managed tasks.

        If *wait_seconds* > 0 and all tasks are still running, blocks until
        the completion event fires (a task finishes) or the timeout expires.
        """
        if wait_seconds > 0 and all(
            t.status == AgentRunStatus.RUNNING for t in self._tasks.values()
        ):
            self._completion_event.clear()
            try:
                await asyncio.wait_for(
                    self._completion_event.wait(), timeout=wait_seconds
                )
            except asyncio.TimeoutError:
                pass

        result: dict[str, Any] = {"running": [], "completed": [], "failed": []}
        for task_id, task in self._tasks.items():
            entry: dict[str, Any] = {
                "task_id": task_id,
                "agent_id": task.agent_id,
                "hypothesis_id": task.hypothesis_id,
            }
            if task.status == AgentRunStatus.RUNNING:
                entry["step"] = task.current_step
                entry["max_steps"] = task.max_steps
                result["running"].append(entry)
            elif task.status == AgentRunStatus.COMPLETED:
                entry["duration_seconds"] = task.duration_seconds
                entry["result"] = task.result
                result["completed"].append(entry)
            elif task.status == AgentRunStatus.FAILED:
                entry["error_summary"] = task.error_summary
                entry["last_steps"] = task.last_steps
                result["failed"].append(entry)
        return result

    async def wait_for_task(self, task_id: str, timeout: float = 180) -> ManagedTask:
        """Wait for a specific task to leave RUNNING state.

        Uses the shared completion event so it wakes instantly when any task
        finishes, then checks whether the target task is done.
        """
        task = self.get_task(task_id)
        while task.status == AgentRunStatus.RUNNING and task.asyncio_task is not None:
            self._completion_event.clear()
            try:
                await asyncio.wait_for(
                    self._completion_event.wait(), timeout=timeout
                )
            except asyncio.TimeoutError:
                break
        return task

    def get_running_count(self) -> int:
        """Return the number of currently running tasks."""
        return sum(1 for t in self._tasks.values() if t.status == AgentRunStatus.RUNNING)

    async def inject(self, task_id: str, instruction: str) -> None:
        """Inject a new instruction into a running Sub-Agent."""
        task = self.get_task(task_id)
        if task.status != AgentRunStatus.RUNNING:
            raise ValueError(
                f"Task {task_id!r} is not running (status={task.status!r})"
            )
        task.pending_instructions.append(instruction)

    async def abort(self, task_id: str, reason: str) -> None:
        """Abort a running Sub-Agent task."""
        task = self.get_task(task_id)
        if task.status != AgentRunStatus.RUNNING:
            raise ValueError(
                f"Task {task_id!r} is not running (status={task.status!r})"
            )
        if task.asyncio_task is not None:
            task.asyncio_task.cancel()
        task.status = AgentRunStatus.FAILED
        task.error_summary = f"Aborted: {reason}"
        self._completion_event.set()
        if self._trajectory is not None:
            await self._trajectory.record(
                event_type="task_abort",
                agent_path=[task.agent_id],
                data={"task_id": task_id, "agent_id": task.agent_id, "reason": reason},
                task_id=task_id,
            )

    def consume_instructions(self, task_id: str) -> list[str]:
        """Dequeue and return all pending instructions for a task.

        Called by the instruction hook before each LLM invocation.
        Returns the list of pending instructions and clears them from the task.
        """
        task = self._tasks[task_id]
        instructions = list(task.pending_instructions)
        task.pending_instructions.clear()
        return instructions

    def get_task(self, task_id: str) -> ManagedTask:
        """Look up a single managed task by ID.

        Raises ValueError if task_id is not found.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise ValueError(
                f"Task {task_id!r} not found. "
                f"Known task IDs: {list(self._tasks.keys())}"
            ) from None

    async def _record_subagent_event(
        self, managed: ManagedTask, data: dict[str, Any]
    ) -> None:
        """Extract and record tool_call events from sub-agent stream data."""
        messages = data.get("messages", [])
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            if role == "ai":
                for tc in getattr(msg, "tool_calls", []):
                    await self._trajectory.record(
                        event_type="tool_call",
                        agent_path=["orchestrator", managed.agent_id],
                        data={
                            "tool_name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        },
                        task_id=managed.task_id,
                    )

    async def _execute_agent(
        self,
        managed: ManagedTask,
        subgraph: Any,
        config: dict[str, Any],
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Core async execution loop for a Sub-Agent subgraph.

        Streams events from the subgraph, updates managed task status,
        and stores results upon completion.  Retries on failure with
        exponential backoff as specified by *retry_config*.
        """
        retry_config = retry_config or RetryConfig()
        input_data = {"messages": [HumanMessage(content=managed.instruction)]}
        last_error: Exception | None = None

        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                managed.events_buffer = []
                async for namespace, mode, data in subgraph.astream(
                    input_data,
                    config,
                    stream_mode=["updates", "custom"],
                    subgraphs=True,
                ):
                    managed.events_buffer.append(data)
                    if len(managed.events_buffer) > 20:
                        managed.events_buffer = managed.events_buffer[-20:]

                    # Record sub-agent tool calls to trajectory
                    if self._trajectory is not None and isinstance(data, dict):
                        await self._record_subagent_event(managed, data)

                    if self._broadcast_callback is not None:
                        await self._broadcast_callback({
                            "agent_path": [managed.agent_id],
                            "mode": mode if isinstance(mode, str) else "updates",
                            "data": data,
                            "timestamp": datetime.now().isoformat(),
                        })

                managed.status = AgentRunStatus.COMPLETED
                managed.result = {"events": managed.events_buffer}
                managed.completed_at = datetime.now().isoformat()
                if managed.started_at:
                    started = datetime.fromisoformat(managed.started_at)
                    managed.duration_seconds = (
                        datetime.now() - started
                    ).total_seconds()
                self._completion_event.set()

                if self._trajectory is not None:
                    await self._trajectory.record(
                        event_type="task_complete",
                        agent_path=[managed.agent_id],
                        data={
                            "task_id": managed.task_id,
                            "agent_id": managed.agent_id,
                            "duration_seconds": managed.duration_seconds,
                        },
                        task_id=managed.task_id,
                    )
                return

            except asyncio.CancelledError:
                managed.completed_at = datetime.now().isoformat()
                self._completion_event.set()
                raise

            except Exception as e:
                last_error = e
                if attempt < retry_config.max_attempts:
                    delay = retry_config.initial_interval * (
                        retry_config.backoff_factor ** (attempt - 1)
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        managed.status = AgentRunStatus.FAILED
        managed.error_summary = str(last_error)
        managed.last_steps = managed.events_buffer[-5:]
        managed.completed_at = datetime.now().isoformat()
        self._completion_event.set()

        if self._trajectory is not None:
            await self._trajectory.record(
                event_type="task_fail",
                agent_path=[managed.agent_id],
                data={
                    "task_id": managed.task_id,
                    "agent_id": managed.agent_id,
                    "error_summary": str(last_error),
                },
                task_id=managed.task_id,
            )
