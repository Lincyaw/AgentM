"""Consolidated answer schema registry for all Sub-Agent task types.

Both react/sub_agent.py and node/worker.py import from this module.
Adding a new scenario only requires adding schemas here and extending ANSWER_SCHEMA.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _BaseAnswer(BaseModel):
    """Shared fields across all sub-agent answer types."""

    findings: str = Field(
        description=(
            "Structured findings following the <output> format in your system "
            "prompt. Exact service names in backticks. No reasoning or caveats."
        ),
    )


# ---------------------------------------------------------------------------
# RCA / hypothesis-driven task types
# ---------------------------------------------------------------------------


class ScoutAnswer(_BaseAnswer):
    """Scout agent output: structural map of the incident + investigation leads."""

    leads: list[str] = Field(
        description=(
            "3-6 divergent investigation directions. Each lead is one sentence: "
            "'[service/component] may [cause] because [evidence]'. "
            "Cover different fault domains (network, resource, dependency, config, code)."
        ),
    )


class DeepAnalyzeAnswer(_BaseAnswer):
    """Deep-analyze agent output: causal mechanism + refined hypotheses."""

    leads: list[str] = Field(
        description=(
            "1-3 refined hypotheses about specific causal mechanisms. Narrow "
            "and evidence-heavy — only include leads that scout-level analysis "
            "could NOT have produced."
        ),
    )


class VerifyAnswer(_BaseAnswer):
    """Verify agent output: adversarial test verdict with tagged (+)/(-) evidence."""

    verdict: str = Field(
        description=(
            "SUPPORTED, CONTRADICTED, or INCONCLUSIVE — followed by exactly "
            "one sentence citing the strongest piece of evidence. SUPPORTED "
            "means the hypothesis survived active disproof attempts."
        ),
    )


# ---------------------------------------------------------------------------
# Memory-extraction task types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Backward-compatible alias (used by task_manager docstrings)
# ---------------------------------------------------------------------------

SubAgentAnswer = ScoutAnswer | DeepAnalyzeAnswer | VerifyAnswer

# ---------------------------------------------------------------------------
# Master registry — imported by both react/sub_agent.py and node/worker.py
# ---------------------------------------------------------------------------

ANSWER_SCHEMA: dict[str, type[BaseModel]] = {
    # RCA task types
    "scout": ScoutAnswer,
    "deep_analyze": DeepAnalyzeAnswer,
    "verify": VerifyAnswer,
    # Memory-extraction task types
    "collect": CollectAnswer,
    "analyze": AnalyzeAnswer,
    "extract": ExtractAnswer,
    "refine": RefineAnswer,
}
