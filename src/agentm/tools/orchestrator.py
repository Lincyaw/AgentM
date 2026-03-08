"""Orchestrator tools for task dispatch, monitoring, hypothesis management, and recall."""

from __future__ import annotations

import json
from typing import Annotated, Any, Callable, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

from agentm.models.enums import HypothesisStatus


def create_orchestrator_tools(
    task_manager: Any,
    agent_pool: Any,
) -> dict[str, Callable[..., Any]]:
    """Factory that creates orchestrator tool functions with injected dependencies.

    Returns a dict mapping tool name to bound tool function. The task_manager and
    agent_pool are captured in closures — no module-level globals needed.
    """

    async def dispatch_agent(
        agent_id: str,
        task: str,
        task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
        hypothesis_id: Optional[str] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Launch a background Sub-Agent. Returns Command to update graph state."""
        task_id = await task_manager.submit(
            agent_id,
            task,
            task_type,
            hypothesis_id,
            subgraph=None,
        )
        content = json.dumps({
            "task_id": task_id,
            "agent_id": agent_id,
            "status": "running",
        })
        return Command(
            update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
        )

    async def check_tasks(
        wait_seconds: float = 10,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Check status of all dispatched tasks and collect completed results."""
        results = await task_manager.get_all_status(wait_seconds=wait_seconds)
        return Command(
            update={
                "messages": [
                    ToolMessage(content=json.dumps(results), tool_call_id=tool_call_id)
                ]
            }
        )

    def inject_instruction(task_id: str, instruction: str) -> str:
        """Inject a new instruction into a running Sub-Agent."""
        managed = task_manager.get_task(task_id)
        managed.pending_instructions.append(instruction)
        return f"Instruction injected into task {task_id}"

    def abort_task(task_id: str, reason: str) -> str:
        """Abort a running Sub-Agent task."""
        from agentm.models.enums import AgentRunStatus

        managed = task_manager.get_task(task_id)
        if managed.asyncio_task is not None:
            managed.asyncio_task.cancel()
        managed.status = AgentRunStatus.FAILED
        managed.error_summary = f"Aborted: {reason}"
        return f"Task {task_id} aborted: {reason}"

    return {
        "dispatch_agent": dispatch_agent,
        "check_tasks": check_tasks,
        "inject_instruction": inject_instruction,
        "abort_task": abort_task,
    }


# --- Tools that do NOT require injected dependencies ---


async def update_hypothesis(
    id: str,
    description: str,
    status: Literal[
        "formed",
        "investigating",
        "confirmed",
        "rejected",
        "refined",
        "inconclusive",
    ] = "formed",
    evidence_summary: Optional[str] = None,
    parent_id: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Create or update a hypothesis in the DiagnosticNotebook."""
    HypothesisStatus(status)  # validate the value matches enum
    content = f"Hypothesis {id} updated: {status} — {description}"
    return Command(
        update={
            "messages": [
                ToolMessage(content=content, tool_call_id=tool_call_id)
            ]
        }
    )


async def remove_hypothesis(
    id: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Remove a hypothesis from the DiagnosticNotebook."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Hypothesis {id} removed",
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


def recall_history(
    query: str,
    scope: Literal["current_compression", "all_compressions"] = "current_compression",
) -> str:
    """Search pre-compression history for detailed information."""
    return "No compression history available yet."
