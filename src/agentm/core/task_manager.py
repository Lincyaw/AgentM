"""TaskManager for async Sub-Agent lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from agentm.core.trajectory import TrajectoryCollector
from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus
from agentm.models.types import TaskType

logger = logging.getLogger("agentm.core.task_manager")


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
        self._trajectory: TrajectoryCollector | None = None
        self._completion_event = asyncio.Event()
        # Smart waiting strategy for check_tasks
        self._smart_wait_strategy = SmartWaitStrategy()

    def set_trajectory(self, trajectory: TrajectoryCollector) -> None:
        """Set the TrajectoryCollector for event recording."""
        self._trajectory = trajectory

    def set_broadcast_callback(self, callback: Callable[..., Any]) -> None:
        """Set the WebSocket broadcast callback."""
        self._broadcast_callback = callback

    @property
    def trajectory(self) -> TrajectoryCollector | None:
        """Access the TrajectoryCollector (read-only)."""
        return self._trajectory

    async def submit(
        self,
        agent_id: str,
        instruction: str,
        task_type: TaskType = "scout",
        hypothesis_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a new task for a Sub-Agent. Returns task_id.

        If ``task_id`` is provided in *kwargs* it is used as-is; otherwise
        a new UUID is generated.  This allows the caller to pre-generate the
        ID so that the same value can be baked into the compiled subgraph
        hooks before the task is submitted.
        """
        task_id = kwargs.pop("task_id", None) or str(uuid.uuid4())
        logger.info("Task %s submitted: agent=%s type=%s", task_id, agent_id, task_type)
        trajectory_self_reported: bool = kwargs.pop("trajectory_self_reported", False)
        max_steps: int | None = kwargs.pop("max_steps", None)
        timeout: int | None = kwargs.pop("timeout", None)

        subgraph = kwargs.get("subgraph")
        config: dict[str, Any] = kwargs.get("config", {})

        managed = ManagedTask(
            task_id=task_id,
            agent_id=agent_id,
            instruction=instruction,
            hypothesis_id=hypothesis_id,
            max_steps=max_steps,
            timeout=timeout,
            started_at=datetime.now().isoformat(),
            subgraph_config=config,
            trajectory_self_reported=trajectory_self_reported,
        )

        if subgraph is not None:
            managed.asyncio_task = asyncio.create_task(
                self._execute_agent(managed, subgraph, config)
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
                    "instruction": instruction,
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
        running = [
            t for t in self._tasks.values() if t.status == AgentRunStatus.RUNNING
        ]
        completed_unreported = [
            t
            for t in self._tasks.values()
            if t.status in (AgentRunStatus.COMPLETED, AgentRunStatus.FAILED)
            and not t.reported
        ]
        if wait_seconds > 0 and running and not completed_unreported:
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
                entry["tool_call_counts"] = task.tool_call_counts
                entry["total_tool_calls"] = sum(task.tool_call_counts.values())
                entry["llm_call_count"] = task.llm_call_count
                entry["last_tool_call"] = task.last_tool_call
                result["running"].append(entry)
            elif task.status == AgentRunStatus.COMPLETED and not task.reported:
                entry["duration_seconds"] = task.duration_seconds
                entry["result"] = task.result
                result["completed"].append(entry)
                task.reported = True
            elif task.status == AgentRunStatus.FAILED and not task.reported:
                entry["error_summary"] = task.error_summary
                if task.result:
                    entry["partial_result"] = task.result
                result["failed"].append(entry)
                task.reported = True
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
                await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                break
        return task

    def get_running_tasks_info(self) -> list[dict[str, Any]]:
        """Get information about all running tasks for smart waiting strategy."""
        running_info = []
        for task_id, task in self._tasks.items():
            if task.status == AgentRunStatus.RUNNING:
                elapsed = 0.0
                if task.started_at:
                    started = datetime.fromisoformat(task.started_at)
                    elapsed = (datetime.now() - started).total_seconds()

                running_info.append(
                    {
                        "task_id": task_id,
                        "agent_id": task.agent_id,
                        "task_type": getattr(task, "task_type", "scout"),
                        "elapsed_seconds": elapsed,
                        "current_step": task.current_step,
                        "max_steps": task.max_steps,
                    }
                )
        return running_info

    def get_running_count(self) -> int:
        """Return the number of currently running tasks."""
        return sum(
            1 for t in self._tasks.values() if t.status == AgentRunStatus.RUNNING
        )

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
        """Extract and record all message events from sub-agent stream data.

        Handles both flat format (``{"messages": [...]}``) and node-keyed
        format (``{"agent": {"messages": [...]}}``), as produced by
        ``create_react_agent`` with ``stream_mode="updates"``.
        """
        trajectory = self._trajectory
        if trajectory is None:
            return

        messages = data.get("messages", [])
        if not messages:
            for node_key, node_data in data.items():
                # pre_model_hook replays the full message history — skip it
                if node_key == "pre_model_hook":
                    continue
                if isinstance(node_data, dict) and "messages" in node_data:
                    messages = node_data["messages"]
                    break
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            if role == "ai":
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name", "unknown")
                        managed.tool_call_counts[name] = managed.tool_call_counts.get(name, 0) + 1
                        managed.last_tool_call = {"name": name, "args": tc.get("args", {})}
                        await trajectory.record(
                            event_type="tool_call",
                            agent_path=["orchestrator", managed.agent_id],
                            data={
                                "tool_name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                            },
                            task_id=managed.task_id,
                        )
                if content:
                    managed.llm_call_count += 1
                    await trajectory.record(
                        event_type="llm_end",
                        agent_path=["orchestrator", managed.agent_id],
                        data={"content": content},
                        task_id=managed.task_id,
                    )
            elif role == "tool":
                tool_name = getattr(msg, "name", "unknown")
                content = getattr(msg, "content", "")
                await trajectory.record(
                    event_type="tool_result",
                    agent_path=["orchestrator", managed.agent_id],
                    data={
                        "tool_name": tool_name,
                        "result": content if content else "",
                    },
                    task_id=managed.task_id,
                )

    async def _execute_agent(
        self,
        managed: ManagedTask,
        subgraph: Any,
        config: dict[str, Any],
    ) -> None:
        """Core async execution loop for a Sub-Agent subgraph.

        Streams events from the subgraph, updates managed task status,
        and stores results upon completion.  No task-level retry —
        transient failures (network, rate-limit) are retried at the
        LLM request level by ChatOpenAI's built-in retry.

        If ``managed.timeout`` is set, the streaming loop is wrapped in
        ``asyncio.wait_for`` — exceeding the limit marks the task as
        FAILED with partial results salvaged from events collected so far.
        """
        input_data = {"messages": [HumanMessage(content=managed.instruction)]}
        worker_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": managed.task_id,
            },
        }

        try:
            managed.events_buffer = []

            async def _stream() -> None:
                async for namespace, mode, data in subgraph.astream(
                    input_data,
                    worker_config,
                    stream_mode=["updates", "custom"],
                    subgraphs=True,
                ):
                    managed.events_buffer.append(data)

                    # Track step progress from llm_call node outputs
                    if isinstance(data, dict):
                        llm_data = data.get("llm_call")
                        if isinstance(llm_data, dict):
                            for msg in llm_data.get("messages", []):
                                if getattr(msg, "type", "") == "ai":
                                    managed.current_step += 1

                    if self._trajectory is not None and isinstance(data, dict):
                        if not managed.trajectory_self_reported:
                            await self._record_subagent_event(managed, data)

            if managed.timeout and managed.timeout > 0:
                await asyncio.wait_for(_stream(), timeout=managed.timeout)
            else:
                await _stream()

            managed.status = AgentRunStatus.COMPLETED
            managed.result = _extract_structured_response(managed.events_buffer)
            managed.completed_at = datetime.now().isoformat()
            logger.info(
                "Task %s completed: status=%s", managed.task_id, managed.status.value
            )
            if managed.started_at:
                started = datetime.fromisoformat(managed.started_at)
                managed.duration_seconds = (datetime.now() - started).total_seconds()
            self._completion_event.set()

            if self._trajectory is not None:
                await self._trajectory.record(
                    event_type="task_complete",
                    agent_path=[managed.agent_id],
                    data={
                        "task_id": managed.task_id,
                        "agent_id": managed.agent_id,
                        "duration_seconds": managed.duration_seconds,
                        "result": managed.result,
                    },
                    task_id=managed.task_id,
                )

        except asyncio.TimeoutError:
            managed.status = AgentRunStatus.FAILED
            managed.error_summary = f"Timed out after {managed.timeout}s"
            partial = _extract_structured_response(managed.events_buffer or [])
            if partial:
                managed.result = {
                    **partial,
                    "_partial": True,
                    "_error": managed.error_summary,
                }
            managed.completed_at = datetime.now().isoformat()
            logger.warning(
                "Task %s timed out after %ds", managed.task_id, managed.timeout
            )
            self._completion_event.set()

            if self._trajectory is not None:
                await self._trajectory.record(
                    event_type="task_timeout",
                    agent_path=[managed.agent_id],
                    data={
                        "task_id": managed.task_id,
                        "agent_id": managed.agent_id,
                        "timeout_seconds": managed.timeout,
                        "steps_completed": managed.current_step,
                        "error_summary": managed.error_summary,
                    },
                    task_id=managed.task_id,
                )

        except asyncio.CancelledError:
            managed.completed_at = datetime.now().isoformat()
            self._completion_event.set()
            raise

        except Exception as e:
            managed.status = AgentRunStatus.FAILED
            managed.error_summary = str(e)
            # Salvage partial findings from events collected before failure
            partial = _extract_structured_response(managed.events_buffer or [])
            if partial:
                managed.result = {**partial, "_partial": True, "_error": str(e)}
            managed.completed_at = datetime.now().isoformat()
            logger.warning("Task %s failed: %s", managed.task_id, e)
            self._completion_event.set()

            if self._trajectory is not None:
                await self._trajectory.record(
                    event_type="task_fail",
                    agent_path=[managed.agent_id],
                    data={
                        "task_id": managed.task_id,
                        "agent_id": managed.agent_id,
                        "error_summary": str(e),
                    },
                    task_id=managed.task_id,
                )


class SmartWaitStrategy:
    """Linear-backoff waiting strategy for check_tasks.

    Starts at 180 s, adds 60 s after each poll, caps at 300 s.
    """

    _BASE: float = 180.0
    _INCREMENT: float = 60.0
    _MAX: float = 300.0

    def __init__(self) -> None:
        self._poll_counts: dict[str, int] = {}  # task_id -> polls so far

    def calculate_wait_time(
        self,
        task_id: str,
        agent_id: str,
        task_type: str,
        elapsed_seconds: float,
        current_step: int,
        max_steps: int | None,
    ) -> float:
        """Return seconds to wait before the next check_tasks call."""
        n = self._poll_counts.get(task_id, 0)
        wait = min(self._BASE + n * self._INCREMENT, self._MAX)
        self._poll_counts[task_id] = n + 1
        return wait

    def record_completion(self, task_id: str, duration_seconds: float) -> None:
        """Clean up state for a finished task."""
        self._poll_counts.pop(task_id, None)


# Global smart wait strategy instance
_smart_wait_strategy = SmartWaitStrategy()


def _extract_structured_response(events_buffer: list[Any]) -> dict[str, Any] | None:
    """Extract the Sub-Agent's structured response from its event stream.

    When response_format is set on create_react_agent, the graph appends a
    generate_structured_response node whose update event contains the key
    ``structured_response``.  In ``stream_mode="updates"`` this arrives as::

        {"generate_structured_response": {"structured_response": SubAgentAnswer(...)}}

    With ``subgraphs=True`` the outer tuple is unpacked by the caller; the
    *data* dict handed here is the inner update payload.
    """
    for event in reversed(events_buffer):
        if not isinstance(event, dict):
            continue
        sr = event.get("structured_response")
        if sr is not None:
            if hasattr(sr, "model_dump"):
                return sr.model_dump()
            return sr
        # Handle the nested node-keyed form produced by stream_mode="updates".
        # react layer: node is "generate_structured_response"
        # node layer:  node is "collect_and_compress"
        for node_key in ("generate_structured_response", "collect_and_compress"):
            node_data = event.get(node_key)
            if isinstance(node_data, dict) and "structured_response" in node_data:
                sr = node_data["structured_response"]
                if hasattr(sr, "model_dump"):
                    return sr.model_dump()
                return sr

    return None
