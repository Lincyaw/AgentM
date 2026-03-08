"""Orchestrator tools for task dispatch, monitoring, hypothesis management, and recall.

All functions are stubs — raise NotImplementedError.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command


async def dispatch_agent(
    agent_id: str,
    task: str,
    task_type: Literal["scout", "verify", "deep_analyze"] = "scout",
    hypothesis_id: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Launch a background Sub-Agent. Returns Command to update graph state."""
    raise NotImplementedError


async def check_tasks(
    wait_seconds: float = 10,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Check status of all dispatched tasks and collect completed results."""
    raise NotImplementedError


def inject_instruction(task_id: str, instruction: str) -> str:
    """Inject a new instruction into a running Sub-Agent."""
    raise NotImplementedError


def abort_task(task_id: str, reason: str) -> str:
    """Abort a running Sub-Agent task."""
    raise NotImplementedError


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
    raise NotImplementedError


async def remove_hypothesis(
    id: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Remove a hypothesis from the DiagnosticNotebook."""
    raise NotImplementedError


def recall_history(
    query: str,
    scope: Literal["current_compression", "all_compressions"] = "current_compression",
) -> str:
    """Search pre-compression history for detailed information."""
    raise NotImplementedError
