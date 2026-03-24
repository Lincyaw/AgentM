"""Trajectory-analysis answer schemas for Sub-Agent task types."""

from __future__ import annotations

from pydantic import Field

from agentm.models.answer_schemas import _BaseAnswer


class AnalyzeAnswer(_BaseAnswer):
    """Generic trajectory analysis worker result."""

    leads: list[str] = Field(
        description="Aspects that need further investigation or follow-up.",
        default_factory=list,
    )
