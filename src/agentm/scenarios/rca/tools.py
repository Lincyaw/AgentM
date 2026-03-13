"""RCA-specific orchestrator tools (hypothesis management)."""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

from agentm.scenarios.rca.enums import HypothesisStatus


def create_rca_tools(
    trajectory: Any | None = None,
) -> dict[str, Callable[..., Any]]:
    """Create RCA-specific orchestrator tools.

    Returns a dict of tool name -> callable for hypothesis management.
    """

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

        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": status,
                    "description": description,
                    "evidence_summary": evidence_summary,
                    "parent_id": parent_id,
                },
                hypothesis_id=id,
            )

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            }
        )

    async def remove_hypothesis(
        id: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Remove a hypothesis from the DiagnosticNotebook."""
        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": "removed",
                    "description": "",
                },
                hypothesis_id=id,
            )

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

    return {
        "update_hypothesis": update_hypothesis,
        "remove_hypothesis": remove_hypothesis,
    }
