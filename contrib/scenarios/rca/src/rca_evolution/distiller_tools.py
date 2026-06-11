"""Distiller agent tools: browse failure reports + manage skills."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)

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
        parameters=BrowseReportsArgs,
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
        parameters=GetReportSummaryArgs,
        fn=_fn,
    )

# ---------------------------------------------------------------------------
# get_existing_skills — list current evolved skills
# ---------------------------------------------------------------------------

class GetExistingSkillsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

def build_get_existing_skills_tool(
    existing_skills: list[dict[str, Any]],
) -> FunctionTool:
    """List all currently evolved skills with their content."""

    async def _fn(args: dict[str, Any]) -> ToolResult:
        if not existing_skills:
            text = "(no existing evolved skills)"
        else:
            text = json.dumps(existing_skills, ensure_ascii=False, indent=2)
        return ToolResult(content=[TextContent(type="text", text=text)])

    return FunctionTool(
        name="get_existing_skills",
        description=(
            f"List all {len(existing_skills)} currently evolved skills. "
            "Each entry has: name, description, body, evidence. "
            "Review these before deciding whether to create a new skill, "
            "update an existing one, or retire one that is redundant."
        ),
        parameters=GetExistingSkillsArgs,
        fn=_fn,
    )

# ---------------------------------------------------------------------------
# submit_skill — terminal tool (create / update / retire)
# ---------------------------------------------------------------------------

class SubmitSkillArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["create", "update", "retire"] = Field(
        description=(
            "'create' = new skill; "
            "'update' = refine an existing skill (set 'name' to the existing skill name); "
            "'retire' = remove a skill that is redundant or harmful (only 'name' + 'reason' needed)."
        ),
    )
    name: str = Field(description="Kebab-case skill name. For update/retire, must match an existing skill.")
    description: str = Field(default="", description="One-line description (required for create/update).")
    tags: list[str] = Field(default_factory=list, description="Tags for the skill.")
    trigger_patterns: list[str] = Field(default_factory=list, description="Situations that should activate this skill.")
    body: str = Field(default="", description="Markdown body of the SKILL.md (≤300 words, required for create/update).")
    reason: str = Field(default="", description="Why this action (especially important for update/retire).")

def build_submit_skill_tool() -> FunctionTool:
    """Terminal tool — create, update, or retire a skill."""

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
            reason=f"evolution:skill-{parsed.action}",
        )

    return FunctionTool(
        name="submit_skill",
        description=(
            "Submit a skill action. Actions:\n"
            "- 'create': write a new SKILL.md\n"
            "- 'update': refine an existing skill with new evidence\n"
            "- 'retire': remove a skill that is redundant or harmful\n"
            "Prefer updating over creating when the failure pattern overlaps "
            "an existing skill. This terminates the session."
        ),
        parameters=SubmitSkillArgs,
        fn=_fn,
        metadata={"terminates": True},
    )
