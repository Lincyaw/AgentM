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
            "The name/value axis, independent of `kind`. Decide by what the symbol "
            "denotes. Decisive test: could a tool report DIFFERENT content for this "
            "while it stays the same thing?\n"
            "- 'identifier' (test: no): the symbol IS the name of a thing referred "
            "to or operated on; the string simply is the thing, with no separate "
            "content that could change.\n"
            "- 'value' (test: yes): the symbol is content something holds or a "
            "check/computation produced; the same slot could hold different content "
            "later.\n"
            "- 'unknown': a vague or anaphoric surface with no clear referent on its "
            "own. Never put an entity_class word in the `kind` field."
        ),
    )


class ExtractionResult(BaseModel):
    symbols: list[ExtractedSymbol] = Field(description="All extracted symbols")
