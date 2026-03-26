"""Trajectory-analysis answer schemas for Sub-Agent task types."""

from __future__ import annotations

from pydantic import Field

from agentm.models.base_answer import _BaseAnswer


class AnalyzeAnswer(_BaseAnswer):
    """Generic trajectory analysis worker result."""

    leads: list[str] = Field(
        description="Aspects that need further investigation or follow-up.",
        default_factory=list,
    )


class CritiqueAnswer(_BaseAnswer):
    """Critic worker review of analysis completeness."""

    phase_gaps: list[str] = Field(
        description=(
            "Missing analysis steps per phase. Each entry: "
            "'[phase]: [what was missing] -> [what you found]'"
        ),
        default_factory=list,
    )
    unverified_claims: list[str] = Field(
        description=(
            "Claims from orchestrator that need verification. Each entry: "
            "'[claim] -> [verification result: confirmed/contradicted/partial]'"
        ),
        default_factory=list,
    )
    blind_spots: list[str] = Field(
        description=(
            "Unexplored directions that could change conclusions. Each entry: "
            "'[direction] -> [what a query reveals or why it matters]'"
        ),
        default_factory=list,
    )
    recommended_actions: list[str] = Field(
        description="Specific follow-up actions before writing vault entries.",
        default_factory=list,
    )
