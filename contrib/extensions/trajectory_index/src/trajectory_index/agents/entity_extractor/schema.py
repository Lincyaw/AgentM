"""Extraction result schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedSymbol(BaseModel):
    name: str = Field(description="Canonical symbol name")
    kind: str = Field(description="Symbol kind from the vocabulary")
    summary: str | None = Field(default=None, description="One-sentence description")
    aliases: list[str] = Field(default_factory=list, description="Alternative surface forms")
    entity_class: str = Field(
        default="identifier",
        description=(
            "Which world this entity lives in — decide from meaning, not spelling:\n"
            "- 'identifier': a rigid name that denotes the same thing every time it "
            "appears (file path, table, id, endpoint, function name, error code, a "
            "proper noun like a place or person). Its value is its own existence.\n"
            "- 'value': a slot whose bound value can change over the trajectory "
            "(a metric like cpu usage, a status, a price, an answer, user.tier).\n"
            "- 'unknown': a vague or anaphoric surface whose referent is unclear on "
            "its own ('the previous result', 'the customer', 'it', 'this approach')."
        ),
    )


class ExtractionResult(BaseModel):
    symbols: list[ExtractedSymbol] = Field(description="All extracted symbols")
