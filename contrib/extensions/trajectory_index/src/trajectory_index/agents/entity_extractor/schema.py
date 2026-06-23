"""Extraction result schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedSymbol(BaseModel):
    name: str = Field(description="Canonical symbol name")
    kind: str = Field(description="One of: variable, object, concept, tool, file, api, state_field")
    summary: str | None = Field(default=None, description="One-sentence description")
    aliases: list[str] = Field(default_factory=list, description="Alternative surface forms")


class ExtractedReference(BaseModel):
    symbol_name: str = Field(description="Must match a symbol name exactly")
    turn_id: str = Field(description="ID of the message where this reference appears")
    text: str = Field(description="Short phrase (< 50 chars)")
    kind: str = Field(
        default="use",
        description="One of: define, use, read, write, mutate, question, answer, tool_input, tool_output",
    )


class ExtractedRelation(BaseModel):
    from_symbol: str = Field(description="Source symbol name")
    to_symbol: str = Field(description="Target symbol name")
    relation_type: str = Field(
        description="One of: uses, defines, updates, derived_from, input_to, output_of, mentions, explains",
    )
    turn_id: str = Field(description="ID of the message where this relation was observed")


class ExtractionResult(BaseModel):
    symbols: list[ExtractedSymbol] = Field(description="All extracted symbols")
    references: list[ExtractedReference] = Field(default_factory=list, description="Symbol references")
    relations: list[ExtractedRelation] = Field(default_factory=list, description="Relations between symbols")
