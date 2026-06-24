"""Extraction result schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedSymbol(BaseModel):
    name: str = Field(description="Canonical symbol name")
    kind: str = Field(description="Symbol kind from the vocabulary")
    summary: str | None = Field(default=None, description="One-sentence description")
    aliases: list[str] = Field(default_factory=list, description="Alternative surface forms")


class ExtractionResult(BaseModel):
    symbols: list[ExtractedSymbol] = Field(description="All extracted symbols")
