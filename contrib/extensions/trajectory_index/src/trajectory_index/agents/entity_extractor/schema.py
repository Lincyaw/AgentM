"""Extraction result schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedSymbol(BaseModel):
    name: str = Field(description="Canonical symbol name")
    kind: str = Field(description="Symbol kind from the vocabulary")
    summary: str | None = Field(
        default=None,
        description="Short phrase for disambiguation when the name alone is ambiguous. Omit when the name is self-explanatory.",
    )
    aliases: list[str] = Field(default_factory=list, description="Alternative surface forms")
    entity_class: str = Field(
        default="identifier",
        description=(
            "The name/value axis, independent of `kind`. "
            "Could a tool report different content for this entity while it stays the same thing?\n"
            "- 'identifier': NO — the string IS the entity (a service name, file path, tool name). Most symbols.\n"
            "- 'value': YES — a tracked quantity the agent monitors across turns; the same measurement can have different values at different times.\n"
            "- 'unknown': a vague or anaphoric surface with no clear referent on its own."
        ),
    )


class ExtractionResult(BaseModel):
    symbols: list[ExtractedSymbol] = Field(description="All extracted symbols")
