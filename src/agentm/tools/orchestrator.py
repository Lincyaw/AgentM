"""Orchestrator tools for task dispatch and monitoring.

These tools wrap AgentRuntime to provide the LLM-callable interface
that the orchestrator agent uses to manage worker agents.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Callable, Protocol, runtime_checkable

from agentm.harness.protocols import AgentLoop
from agentm.harness.runtime import AgentRuntime
from agentm.harness.types import AgentStatus, RunConfig
from agentm.models.types import TaskType


@runtime_checkable
class WorkerFactory(Protocol):
    """Creates worker AgentLoop instances for a given task type."""

    def create_worker(self, agent_id: str, task_type: TaskType) -> AgentLoop: ...


def create_orchestrator_tools(
    runtime: AgentRuntime,
    worker_factory: WorkerFactory,
    *,
    max_concurrent_workers: int | None = None,
    check_tasks_wait_seconds: float = 600.0,
) -> dict[str, Callable[..., Any]]:
    """Factory that creates orchestrator tool functions backed by AgentRuntime.

    Returns a dict mapping tool name to bound tool function. The runtime and
    worker_factory are captured in closures.

    Args:
        runtime: AgentRuntime instance for agent lifecycle management.
        worker_factory: Factory that creates AgentLoop instances per task type.
        max_concurrent_workers: Max parallel workers. None = unlimited.
    """
    _worker_semaphore: asyncio.Semaphore | None = (
        asyncio.Semaphore(max_concurrent_workers)
        if max_concurrent_workers is not None
        else None
    )

    async def dispatch_agent(
        agent_id: str,
        task: str,
        task_type: TaskType,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Launch a Sub-Agent. Auto-blocks when this is the only running task.

        Single-worker: waits for completion and returns result directly,
        saving an LLM roundtrip through check_tasks.
        Multi-worker: returns immediately with status "running".

        When max_concurrent_workers is configured, blocks until a slot
        is available before spawning.

        Args:
            agent_id: Which agent to dispatch.
            task: Natural language instruction for the agent.
            task_type: Task type key (e.g. "scout", "verify", "execute").
                Determines the answer schema and prompt overlay.
            metadata: Optional scenario-specific key-value pairs
                (e.g. {"hypothesis_id": "H1"} for RCA).
        """
        if _worker_semaphore is not None:
            await _worker_semaphore.acquire()

        unique_id = f"{agent_id}-{uuid.uuid4().hex[:8]}"
        loop = worker_factory.create_worker(agent_id, task_type)

        run_metadata = {"task_type": task_type, "original_agent_id": agent_id}
        if metadata:
            run_metadata.update(metadata)

        handle = await runtime.spawn(
            unique_id,
            loop=loop,
            input=task,
            parent_id="orchestrator",
            config=RunConfig(metadata=run_metadata),
            metadata=run_metadata,
        )

        # Release semaphore when worker finishes
        if _worker_semaphore is not None:

            async def _release_on_done(aid: str) -> None:
                try:
                    await runtime.wait(aid)
                finally:
                    _worker_semaphore.release()

            asyncio.create_task(_release_on_done(handle.agent_id))

        # Auto-block: if this is the only running worker, wait for completion
        running_ids = runtime.get_running_ids()
        if len(running_ids) == 1 and running_ids[0] == handle.agent_id:
            result = await runtime.wait(handle.agent_id)
            content = json.dumps(
                {
                    "task_id": handle.agent_id,
                    "agent_id": agent_id,
                    "status": result.status.value,
                    "result": result.output,
                    "error_summary": result.error,
                    "duration_seconds": result.duration_seconds,
                },
                default=str,
            )
        else:
            content = json.dumps(
                {
                    "task_id": handle.agent_id,
                    "agent_id": agent_id,
                    "status": "running",
                }
            )

        return content

    async def check_tasks(
        request: str,
    ) -> str:
        """Check status of all dispatched tasks and collect completed results.

        Waits briefly for any running agent to complete before returning,
        so the orchestrator gets fresh results without busy-polling.

        Args:
            request: Description of what we're waiting for.
        """
        _ = request

        running_ids = runtime.get_running_ids()
        wait_seconds = _calculate_wait(running_ids, check_tasks_wait_seconds)

        # Wait for any agent to complete (or timeout)
        if running_ids and wait_seconds > 0:
            try:
                await runtime.wait_any(running_ids, timeout=wait_seconds)
            except (TimeoutError, asyncio.TimeoutError):
                pass

        # Build response from runtime status
        status_map = runtime.get_status()

        running: list[dict[str, Any]] = []
        completed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        for agent_id, info in status_map.items():
            # Skip the orchestrator itself
            if info.parent_id != "orchestrator":
                continue

            entry: dict[str, Any] = {
                "task_id": agent_id,
                "agent_id": info.metadata.get("original_agent_id", agent_id),
                "task_type": info.metadata.get("task_type", "unknown"),
                "current_step": info.current_step,
            }

            if info.status == AgentStatus.RUNNING:
                running.append(entry)
            elif info.status == AgentStatus.COMPLETED:
                if info.result:
                    entry["result"] = info.result.output
                    entry["duration_seconds"] = info.result.duration_seconds
                completed.append(entry)
            else:
                # FAILED or ABORTED
                entry["status"] = info.status.value
                if info.result:
                    entry["error_summary"] = info.result.error
                    entry["duration_seconds"] = info.result.duration_seconds
                failed.append(entry)

        results: dict[str, Any] = {
            "running": running,
            "completed": completed,
            "failed": failed,
            "waited_seconds": wait_seconds,
            "running_count": len(running),
            "completed_count": len(completed),
            "failed_count": len(failed),
        }

        return json.dumps(results, default=str)

    async def inject_instruction(task_id: str, instruction: str) -> str:
        """Inject a new instruction into a running Sub-Agent."""
        try:
            await runtime.send(task_id, instruction)
        except ValueError:
            return f"Task {task_id} not found -- instruction not injected. Use check_tasks to collect its result."
        return f"Instruction injected into task {task_id}"

    async def abort_task(task_id: str, reason: str) -> str:
        """Abort a running Sub-Agent task."""
        try:
            ok = await runtime.abort(task_id, reason)
        except ValueError:
            return f"Task {task_id} not found -- cannot abort. Use check_tasks to collect its result."
        if not ok:
            return f"Task {task_id} is no longer running -- cannot abort. Use check_tasks to collect its result."
        return f"Task {task_id} aborted: {reason}"

    tools: dict[str, Callable[..., Any]] = {
        "dispatch_agent": dispatch_agent,
        "check_tasks": check_tasks,
        "inject_instruction": inject_instruction,
        "abort_task": abort_task,
    }
    return tools


def _calculate_wait(running_ids: list[str], default_wait: float = 600.0) -> float:
    """Return *default_wait* when agents are running, else 0."""
    return default_wait if running_ids else 0.0
