"""TaskManager for async Sub-Agent lifecycle management."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Literal, Optional

from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus


class TaskManager:
    """Manages asynchronously executing Sub-Agent subgraphs.

    Provides the backing implementation for Orchestrator tools:
    - dispatch_agent -> submit()
    - check_tasks   -> get_all_status()
    - inject_instruction -> inject()
    - abort_task    -> abort()
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ManagedTask] = {}
        self._broadcast_callback: Callable[..., Any] | None = None
        self._trajectory: Any | None = None  # Set by builder

    async def submit(
        self,
        agent_id: str,
        instruction: str,
        task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
        hypothesis_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Submit a new task for a Sub-Agent. Returns task_id."""
        task_id = str(uuid.uuid4())

        subgraph = kwargs.get("subgraph")
        config: dict[str, Any] = kwargs.get("config", {})

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
                self._execute_agent(managed, subgraph, config)
            )

        self._tasks[task_id] = managed

        # Record task dispatch to trajectory
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
        """Get status of all managed tasks. Optionally wait if all are still running."""
        if wait_seconds > 0 and all(
            t.status == AgentRunStatus.RUNNING for t in self._tasks.values()
        ):
            done, _ = await asyncio.wait(
                [t.asyncio_task for t in self._tasks.values() if t.asyncio_task],
                timeout=wait_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )

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

    async def inject(self, task_id: str, instruction: str) -> None:
        """Inject a new instruction into a running Sub-Agent."""
        task = self._tasks[task_id]
        if task.status != AgentRunStatus.RUNNING:
            raise ValueError(
                f"Task {task_id!r} is not running (status={task.status!r})"
            )
        task.pending_instructions.append(instruction)

    async def abort(self, task_id: str, reason: str) -> None:
        """Abort a running Sub-Agent task."""
        task = self._tasks[task_id]
        if task.status != AgentRunStatus.RUNNING:
            raise ValueError(
                f"Task {task_id!r} is not running (status={task.status!r})"
            )
        if task.asyncio_task is not None:
            task.asyncio_task.cancel()
        task.status = AgentRunStatus.FAILED
        task.error_summary = f"Aborted: {reason}"

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

        Raises KeyError if task_id is not found.
        """
        return self._tasks[task_id]

    async def _execute_agent(
        self,
        managed: ManagedTask,
        subgraph: Any,
        config: dict[str, Any],
    ) -> None:
        """Core async execution loop for a Sub-Agent subgraph.

        Streams events from the subgraph, updates managed task status,
        and stores results upon completion.
        """
        from langchain_core.messages import HumanMessage

        input_data = {"messages": [HumanMessage(content=managed.instruction)]}

        try:
            async for namespace, mode, data in subgraph.astream(
                input_data,
                config,
                stream_mode=["updates", "custom"],
                subgraphs=True,
            ):
                managed.events_buffer.append(data)
                if len(managed.events_buffer) > 20:
                    managed.events_buffer = managed.events_buffer[-20:]

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

        except asyncio.CancelledError:
            managed.completed_at = datetime.now().isoformat()
            raise

        except Exception as e:
            managed.status = AgentRunStatus.FAILED
            managed.error_summary = str(e)
            managed.last_steps = managed.events_buffer[-5:]
            managed.completed_at = datetime.now().isoformat()

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
