"""Consolidated answer schema registry for all Sub-Agent task types.

Both react/sub_agent.py and node/worker.py import from this module.
Adding a new scenario only requires registering schemas via scenario init.

SDK base class ``_BaseAnswer`` is defined here. Domain-specific schemas
live in their canonical locations under ``scenarios/``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared base (SDK)
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
# Master registry — imported by both react/sub_agent.py and node/worker.py
# ---------------------------------------------------------------------------

ANSWER_SCHEMA: dict[str, type[BaseModel]] = {}
_defaults_loaded = False


def _ensure_defaults() -> None:
    """Lazily import and register default answer schemas from scenarios."""
    global _defaults_loaded
    if _defaults_loaded:
        return
    _defaults_loaded = True

    from agentm.scenarios.rca.answer_schemas import (
        ScoutAnswer,
        DeepAnalyzeAnswer,
        VerifyAnswer,
    )
    from agentm.scenarios.memory_extraction.answer_schemas import (
        CollectAnswer,
        AnalyzeAnswer,
        ExtractAnswer,
        RefineAnswer,
    )

    ANSWER_SCHEMA.setdefault("scout", ScoutAnswer)
    ANSWER_SCHEMA.setdefault("deep_analyze", DeepAnalyzeAnswer)
    ANSWER_SCHEMA.setdefault("verify", VerifyAnswer)
    ANSWER_SCHEMA.setdefault("collect", CollectAnswer)
    ANSWER_SCHEMA.setdefault("analyze", AnalyzeAnswer)
    ANSWER_SCHEMA.setdefault("extract", ExtractAnswer)
    ANSWER_SCHEMA.setdefault("refine", RefineAnswer)


def get_answer_schema(task_type: str) -> type[BaseModel] | None:
    """Look up an answer schema by task type, loading defaults lazily."""
    _ensure_defaults()
    return ANSWER_SCHEMA.get(task_type)
