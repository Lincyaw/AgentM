"""General-purpose-specific answer schemas for Sub-Agent task types."""

from __future__ import annotations

from pydantic import Field

from agentm.models.answer_schemas import _BaseAnswer


class GeneralAnswer(_BaseAnswer):
    """General-purpose worker output: findings + optional action items."""

    action_items: list[str] = Field(
        description="Concrete next steps or action items identified during the task.",
        default_factory=list,
    )
