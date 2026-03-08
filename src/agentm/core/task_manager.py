"""TaskManager for async Sub-Agent lifecycle management.

All methods are stubs — raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from agentm.models.data import ManagedTask


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

    async def submit(
        self,
        agent_id: str,
        instruction: str,
        task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
        hypothesis_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Submit a new task for a Sub-Agent. Returns task_id."""
        raise NotImplementedError

    async def get_all_status(self, wait_seconds: float = 0) -> dict[str, Any]:
        """Get status of all managed tasks. Optionally wait if all are still running."""
        raise NotImplementedError

    async def inject(self, task_id: str, instruction: str) -> None:
        """Inject a new instruction into a running Sub-Agent."""
        raise NotImplementedError

    async def abort(self, task_id: str, reason: str) -> None:
        """Abort a running Sub-Agent task."""
        raise NotImplementedError

    def consume_instructions(self, task_id: str) -> list[str]:
        """Dequeue and return all pending instructions for a task.

        Called by the instruction hook before each LLM invocation.
        Returns the list of pending instructions and clears them from the task.
        """
        raise NotImplementedError

    def get_task(self, task_id: str) -> ManagedTask:
        """Look up a single managed task by ID.

        Raises KeyError if task_id is not found.
        """
        raise NotImplementedError

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
        raise NotImplementedError
