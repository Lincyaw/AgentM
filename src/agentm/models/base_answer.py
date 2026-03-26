"""Shared base class for all sub-agent answer schemas.

Moved from ``models/answer_schemas.py`` during Phase 3A cleanup.
Domain-specific answer schemas live in their canonical locations
under ``scenarios/``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class _BaseAnswer(BaseModel):
    """Shared fields across all sub-agent answer types."""

    findings: str = Field(
        description=(
            "Structured findings following the <output> format in your system "
            "prompt. Exact service names in backticks. No reasoning or caveats."
        ),
    )
