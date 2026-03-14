"""TaskManager for async Sub-Agent lifecycle management."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from agentm.core.trajectory import TrajectoryCollector
from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus
from agentm.models.types import TaskType


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
        trajectory_self_reported: bool = kwargs.pop("trajectory_self_reported", False)
        max_steps: int | None = kwargs.pop("max_steps", None)

        subgraph = kwargs.get("subgraph")
        config: dict[str, Any] = kwargs.get("config", {})

        managed = ManagedTask(
            task_id=task_id,
            agent_id=agent_id,
            instruction=instruction,
            hypothesis_id=hypothesis_id,
            max_steps=max_steps,
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
                result["running"].append(entry)
            elif task.status == AgentRunStatus.COMPLETED and not task.reported:
                entry["duration_seconds"] = task.duration_seconds
                entry["result"] = task.result
                result["completed"].append(entry)
                task.reported = True
            elif task.status == AgentRunStatus.FAILED and not task.reported:
                entry["error_summary"] = task.error_summary
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

            managed.status = AgentRunStatus.COMPLETED
            managed.result = _extract_structured_response(managed.events_buffer)
            managed.completed_at = datetime.now().isoformat()
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

        except asyncio.CancelledError:
            managed.completed_at = datetime.now().isoformat()
            self._completion_event.set()
            raise

        except Exception as e:
            managed.status = AgentRunStatus.FAILED
            managed.error_summary = str(e)
            managed.completed_at = datetime.now().isoformat()
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
