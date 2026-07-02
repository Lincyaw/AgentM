"""Observer agent tools: trajectory inspection + structured divergence report.

Mounted on a short-lived AgentSession that investigates why an RCA case
failed. The agent uses ``get_turn`` to drill into specific trajectory
turns, ``get_gt_info`` to see the ground truth, and
``submit_divergence_report`` to emit its structured findings.
"""

from __future__ import annotations

from loguru import logger

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)

# ---------------------------------------------------------------------------
# get_turn — drill into one trajectory turn
# ---------------------------------------------------------------------------

class GetTurnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idx: int = Field(description="0-based turn index in the trajectory.")

def build_get_turn_tool(snapshot: list[dict[str, Any]]) -> FunctionTool:
    """Read one turn from the pre-serialized parent trajectory."""

    async def _fn(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = GetTurnArgs.model_validate(args)
        except Exception as exc:
            logger.debug("observer_tools: get_turn rejected: {}", exc)
            return ToolResult(
                content=[TextContent(type="text", text=f"get_turn rejected: {exc}")],
                is_error=True,
            )
        if parsed.idx < 0 or parsed.idx >= len(snapshot):
            return ToolResult(
                content=[TextContent(type="text", text=f"idx {parsed.idx} out of range [0, {len(snapshot)})")],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(snapshot[parsed.idx], ensure_ascii=False))],
        )

    return FunctionTool(
        name="get_turn",
        description=(
            "Fetch a specific turn from the RCA agent's trajectory by index. "
            "Returns the serialized turn (role, content, tool_calls, tool_results). "
            "Use to inspect what the agent did at a particular step."
        ),
        parameters=GetTurnArgs,
        fn=_fn,
    )

# ---------------------------------------------------------------------------
# get_gt_info — ground truth for the case
# ---------------------------------------------------------------------------

class GetGtInfoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

def build_get_gt_info_tool(gt_info: dict[str, Any]) -> FunctionTool:
    """Return the ground truth root causes and fault types."""
    serialized = json.dumps(gt_info, ensure_ascii=False)

    async def _fn(args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=[TextContent(type="text", text=serialized)],
        )

    return FunctionTool(
        name="get_gt_info",
        description=(
            "Get the ground truth for this case: correct root cause services, "
            "fault types, and what the agent concluded. Call this first to "
            "understand what the correct answer was."
        ),
        parameters=GetGtInfoArgs,
        fn=_fn,
    )

# ---------------------------------------------------------------------------
# get_trajectory_summary — high-level overview before drilling down
# ---------------------------------------------------------------------------

class GetTrajectorySummaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

def build_get_trajectory_summary_tool(summary: str, total_turns: int) -> FunctionTool:
    """Return a condensed list of all tool calls in the trajectory."""

    async def _fn(args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=[TextContent(type="text", text=f"Total turns: {total_turns}\n\n{summary}")],
        )

    return FunctionTool(
        name="get_trajectory_summary",
        description=(
            "Get a high-level summary of all tool calls the RCA agent made, "
            "with turn indices. Use this to get an overview before drilling "
            "into specific turns with get_turn."
        ),
        parameters=GetTrajectorySummaryArgs,
        fn=_fn,
    )

# ---------------------------------------------------------------------------
# submit_divergence_report — terminal tool
# ---------------------------------------------------------------------------

class DivergencePointModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_index: int = Field(description="Approximate turn number where the error occurred.")
    description: str = Field(description="What the agent did wrong at this point.")
    should_have_done: str = Field(description="What the correct action would have been.")
    category: str = Field(
        description=(
            "Error category: missed_metric, red_herring, premature_conclusion, "
            "wrong_service_focus, insufficient_evidence, correlation_confusion, "
            "ignored_anomaly, or another descriptive category."
        ),
    )

class SubmitDivergenceReportArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    divergence_points: list[DivergencePointModel] = Field(
        description="List of points where the investigation diverged from the correct path.",
        min_length=1,
    )
    key_lesson: str = Field(
        description="One-sentence takeaway about the failure pattern.",
    )

def build_submit_divergence_report_tool() -> FunctionTool:
    """Terminal tool — submit structured divergence findings."""

    async def _fn(args: dict[str, Any]) -> ToolTerminate:
        try:
            parsed = SubmitDivergenceReportArgs.model_validate(args)
        except Exception as exc:
            logger.debug("observer_tools: submit_divergence_report rejected: {}", exc)
            return ToolResult(  # type: ignore[return-value]
                content=[TextContent(type="text", text=f"submit_divergence_report rejected: {exc}")],
                is_error=True,
            )
        return ToolTerminate(
            result=ToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(parsed.model_dump(), ensure_ascii=False),
                )],
            ),
            reason="evolution:divergence-report-submitted",
        )

    return FunctionTool(
        name="submit_divergence_report",
        description=(
            "Submit your analysis of where the RCA investigation went wrong. "
            "Call this after you have investigated the trajectory and identified "
            "the divergence points. This terminates the session."
        ),
        parameters=SubmitDivergenceReportArgs,
        fn=_fn,
        metadata={"terminates": True},
    )
