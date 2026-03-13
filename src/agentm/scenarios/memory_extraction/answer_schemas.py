"""Memory-extraction-specific answer schemas for Sub-Agent task types."""

from __future__ import annotations

from pydantic import Field

from agentm.models.answer_schemas import _BaseAnswer


class CollectAnswer(_BaseAnswer):
    """Trajectory collection result."""

    trajectories_loaded: list[str] = Field(
        description="Thread IDs of trajectories successfully read and summarised.",
        default_factory=list,
    )
    patterns_observed: list[str] = Field(
        description="Preliminary patterns noticed during collection (1-sentence each).",
        default_factory=list,
    )


class AnalyzeAnswer(_BaseAnswer):
    """Trajectory analysis result."""

    patterns: list[dict] = Field(
        description=(
            "Extracted patterns, each a dict with keys: "
            "pattern_type, description, evidence (list of supporting snippets)."
        ),
        default_factory=list,
    )
    leads: list[str] = Field(
        description="Aspects that need deeper extraction in a follow-up pass.",
        default_factory=list,
    )


class ExtractAnswer(_BaseAnswer):
    """Knowledge entry extraction result."""

    knowledge_entries: list[dict] = Field(
        description=(
            "KnowledgeEntry-shaped dicts ready to be written to the store. "
            "Each dict should contain at minimum: title, description, category."
        ),
        default_factory=list,
    )


class RefineAnswer(_BaseAnswer):
    """Knowledge refinement result."""

    updated_entries: list[str] = Field(
        description="Paths of knowledge entries written or updated.",
        default_factory=list,
    )
    skipped_entries: list[str] = Field(
        description="Paths skipped (duplicate content or confidence below threshold).",
        default_factory=list,
    )
