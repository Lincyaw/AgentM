"""Memory-extraction-specific structured output schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeSummary(BaseModel):
    """Final output of a memory_extraction run."""

    entries_created: int = Field(description="Number of new knowledge entries written.")
    entries_updated: int = Field(description="Number of existing entries updated.")
    entries_skipped: int = Field(
        description="Number of entries skipped (duplicate or low confidence)."
    )
    categories: list[str] = Field(
        description="Distinct knowledge categories populated during this run."
    )
    summary: str = Field(
        description="One-paragraph prose summary of what was learned and stored."
    )
