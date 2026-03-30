"""Data models for trajectory judger scenario.

Defines Pydantic models for trajectory classification including the main
TrajectoryLabel output, AnalyzeTask input, and BatchReport for batch analysis.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
    evidence: list[dict] = Field(
        default_factory=list,
        description="Evidence supporting the label (step numbers, queries, statements)",
    )

    # Key step locations
    key_steps: dict = Field(
        default_factory=dict,
        description="Step numbers for key events",
    )

    # Statistics
    stats: dict = Field(
        default_factory=dict,
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


class AnalyzeTask(BaseModel):
    """Input task for single trajectory analysis.

    Contains all data needed for the trajectory judger to classify
    a single agent execution.
    """

    trajectory_id: str = Field(
        ...,
        description="Unique identifier for the trajectory to analyze",
    )
    trajectory_data: dict = Field(
        ...,
        description="The trajectory JSON data containing agent execution steps",
    )
    case_id: str = Field(
        ...,
        description="Case identifier for grouping related trajectories",
    )
    ground_truth: list[str] = Field(
        ...,
        description="List of actual root-cause service names",
    )


class BatchReport(BaseModel):
    """Output for batch analysis results.

    Aggregated statistics across multiple trajectory analyses including
    distribution by category and sub-type.
    """

    total: int = Field(
        ...,
        description="Total number of trajectories analyzed",
        ge=0,
    )
    by_category: dict[str, int] = Field(
        default_factory=dict,
        description="Count of trajectories per category",
    )
    by_sub_type: dict[str, int] = Field(
        default_factory=dict,
        description="Count of trajectories per sub-type",
    )
    confidence_distribution: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Confidence scores distribution by category",
    )
    detailed_results: list[TrajectoryLabel] = Field(
        default_factory=list,
        description="Individual labels for all analyzed trajectories",
    )
