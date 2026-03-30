"""Data models for trajectory judger scenario.

Defines the TrajectoryLabel structured output for trajectory classification.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """A single piece of evidence supporting the classification."""

    step: int = Field(description="Step number in the trajectory where this evidence was observed")
    description: str = Field(description="What happened at this step")
    relevance: str = Field(default="", description="Why this evidence matters for classification")


class TrajectoryStats(BaseModel):
    """Query statistics extracted from the trajectory."""

    total_steps: int = Field(default=0, description="Total number of agent steps")
    total_tool_calls: int = Field(default=0, description="Total tool invocations")
    unique_services_queried: int = Field(default=0, description="Distinct services investigated")
    root_cause_first_mentioned_step: int = Field(default=-1, description="Step when actual root cause first appeared (-1 if never)")


class TrajectoryLabel(BaseModel):
    """Trajectory analysis label.

    Single-pass classification of an RCA agent trajectory according to the
    decision-tree taxonomy: success, lucky_hit, exploration_fail,
    confirmation_fail, or judgment_fail.
    """

    # Basic info
    trajectory_id: str = Field(
        ...,
        description="Unique identifier for the trajectory",
    )
    case_id: str = Field(
        ...,
        description="Case identifier associated with this trajectory",
    )

    # Contrast
    agent_conclusion: list[str] = Field(
        ...,
        description="Services identified by the agent as root cause",
    )
    ground_truth: list[str] = Field(
        ...,
        description="Actual root-cause services",
    )
    is_correct: bool = Field(
        ...,
        description="Whether the agent conclusion fully matches ground truth",
    )
    is_partial: bool = Field(
        ...,
        description="Whether the agent found some but not all root causes",
    )

    # Classification result
    category: Literal[
        "success",
        "lucky_hit",
        "exploration_fail",
        "confirmation_fail",
        "judgment_fail",
    ] = Field(
        ...,
        description="High-level classification category from decision tree",
    )
    sub_type: str | None = Field(
        default=None,
        description="Detailed sub-type within the category (e.g., 'confirmation_bias')",
    )

    # Analysis details
    reasoning: str = Field(
        ...,
        description="Detailed justification of the classification (200+ characters)",
        min_length=10,
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence supporting the label (step numbers, queries, statements)",
    )

    # Key step locations
    key_steps: dict[str, int] = Field(
        default_factory=dict,
        description="Step numbers for key events (e.g., first_query, pivot_point)",
    )

    # Statistics
    stats: TrajectoryStats = Field(
        default_factory=TrajectoryStats,
        description="Query statistics from the trajectory",
    )

    # Metadata
    analyzed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when analysis was performed",
    )
    analyzer_version: str = Field(
        default="1.0.0",
        description="Version of the analyzer that produced this label",
    )
