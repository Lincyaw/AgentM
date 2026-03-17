"""General-purpose answer schema for Sub-Agent task types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GeneralAnswer(BaseModel):
    """General-purpose worker output: a plain-text report."""

    answer: str = Field(
        description="Complete report of findings and conclusions from the task.",
    )
