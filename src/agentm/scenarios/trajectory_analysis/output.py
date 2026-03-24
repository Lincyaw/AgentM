"""Trajectory-analysis structured output schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisReport(BaseModel):
    """Final output of a trajectory analysis run."""

    skill: str = Field(description="Which analysis skill was applied")
    source_count: int = Field(description="Number of trajectories analyzed")
    findings: list[dict] = Field(
        description="Key findings from the analysis (skill-specific shape)"
    )
    artifacts: list[str] = Field(
        description="Paths of artifacts created (vault entries, reports, etc.)"
    )
    summary: str = Field(description="Prose summary of the analysis")
