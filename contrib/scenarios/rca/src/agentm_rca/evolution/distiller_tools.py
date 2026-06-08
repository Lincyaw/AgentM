"""Distiller agent tools: browse failure reports + submit a SKILL.md."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.tool import ToolTerminate
from agentm.core.lib import pydantic_to_openai_tool_schema


# ---------------------------------------------------------------------------
# browse_reports — read divergence reports
# ---------------------------------------------------------------------------

class BrowseReportsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idx: int = Field(description="0-based index of the failure report to read.")


def build_browse_reports_tool(reports: list[dict[str, Any]]) -> FunctionTool:
    """Read one failure report by index."""

    async def _fn(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = BrowseReportsArgs.model_validate(args)
        except Exception as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"browse_reports rejected: {exc}")],
                is_error=True,
            )
        if parsed.idx < 0 or parsed.idx >= len(reports):
            return ToolResult(
                content=[TextContent(type="text", text=f"idx {parsed.idx} out of range [0, {len(reports)})")],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(reports[parsed.idx], ensure_ascii=False))],
        )

    return FunctionTool(
        name="browse_reports",
        description=(
            f"Read a failure analysis report by index (0 to {len(reports) - 1}). "
            "Each report contains: case_id, root_causes_gt, root_causes_agent, "
            "divergence_points (with category, description, should_have_done), "
            "and key_lesson."
        ),
        parameters=pydantic_to_openai_tool_schema(BrowseReportsArgs),
        fn=_fn,
    )


# ---------------------------------------------------------------------------
# get_report_summary — overview of all reports
# ---------------------------------------------------------------------------

class GetReportSummaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_get_report_summary_tool(summary: str) -> FunctionTool:
    """Return aggregated stats of all failure reports."""

    async def _fn(args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=[TextContent(type="text", text=summary)],
        )

    return FunctionTool(
        name="get_report_summary",
        description=(
            "Get an overview of all failure reports: total count, category "
            "frequencies, and per-case summaries. Call this first to identify "
            "the dominant failure pattern before browsing individual reports."
        ),
        parameters=pydantic_to_openai_tool_schema(GetReportSummaryArgs),
        fn=_fn,
    )


# ---------------------------------------------------------------------------
# submit_skill — terminal tool
# ---------------------------------------------------------------------------

class SubmitSkillArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Kebab-case skill name, e.g. 'verify-before-conclude'.")
    description: str = Field(description="One-line description of what the skill addresses.")
    tags: list[str] = Field(description="Tags for the skill, e.g. ['rca', 'pod_failure'].")
    trigger_patterns: list[str] = Field(description="Situations that should activate this skill.")
    body: str = Field(description="Markdown body of the SKILL.md (≤300 words, actionable guidance).")


def build_submit_skill_tool() -> FunctionTool:
    """Terminal tool — submit a distilled SKILL.md."""

    async def _fn(args: dict[str, Any]) -> ToolTerminate:
        try:
            parsed = SubmitSkillArgs.model_validate(args)
        except Exception as exc:
            return ToolResult(  # type: ignore[return-value]
                content=[TextContent(type="text", text=f"submit_skill rejected: {exc}")],
                is_error=True,
            )
        return ToolTerminate(
            result=ToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(parsed.model_dump(), ensure_ascii=False),
                )],
            ),
            reason="evolution:skill-submitted",
        )

    return FunctionTool(
        name="submit_skill",
        description=(
            "Submit a SKILL.md that addresses the dominant failure pattern. "
            "The skill should be actionable, specific, and ≤300 words. "
            "This terminates the session."
        ),
        parameters=pydantic_to_openai_tool_schema(SubmitSkillArgs),
        fn=_fn,
        metadata={"terminates": True},
    )
