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
            "The name/value axis, independent of `kind`. Decide: is the symbol itself "
            "a piece of data, or the name of a resource?\n"
            "- 'identifier': it names a resource you refer to or operate on; the "
            "string simply is the thing.\n"
            "- 'value': it is data — a value, measurement, status, verdict, answer, or "
            "result, including an expression or formula (which stands for the value it "
            "computes to).\n"
            "- 'unknown': a vague or anaphoric surface with no clear referent on its "
            "own. Never put an entity_class word in the `kind` field."
        ),
    )


class ExtractionResult(BaseModel):
    symbols: list[ExtractedSymbol] = Field(description="All extracted symbols")
