"""Extraction result schema.

The extractor outputs a structured list of found symbols, claims,
observation regions, and constraints — no full-text re-emission. Code
locates each in the original messages by name/substring matching.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedSymbol(BaseModel):
    name: str = Field(description="Canonical symbol name")
    kind: str = Field(description="Symbol kind from the vocabulary")
    aliases: list[str] = Field(default_factory=list, description="Alternative surface forms")
    entity_class: str = Field(
        default="identifier",
        description=(
            "identifier: the string IS the entity (file path, function name). "
            "value: a tracked quantity the agent monitors across turns. "
            "unknown: vague/anaphoric surface."
        ),
    )


class ExtractedClaim(BaseModel):
    head: str = Field(description="Verbatim head anchor of the claim")
    tail: str = Field(default="", description="Verbatim tail anchor")
    role: str = Field(
        default="",
        description="Empty for ordinary claims; 'commit' for the agent's final answer",
    )


class ExtractedObs(BaseModel):
    head: str = Field(
        description="Verbatim head anchor of the retrieved/environment region",
    )
    tail: str = Field(
        default="",
        description="Verbatim tail anchor (omit if head is the whole region)",
    )


class ExtractedConstraint(BaseModel):
    head: str = Field(description="Verbatim head anchor of the requirement")
    tail: str = Field(default="", description="Verbatim tail anchor")


class ExtractedValue(BaseModel):
    sym: str = Field(description="Symbol name this value belongs to")
    value: str = Field(description="The concrete value text")


class ExtractionResult(BaseModel):
    symbols: list[ExtractedSymbol] = Field(
        default_factory=list,
        description="Named entities found in the trajectory chunk",
    )
    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="Settled-fact assertions by the agent",
    )
    observations: list[ExtractedObs] = Field(
        default_factory=list,
        description="Retrieved/environment regions in assistant steps (not tool_result steps)",
    )
    constraints: list[ExtractedConstraint] = Field(
        default_factory=list,
        description="Task requirements from the user's question/instructions",
    )
    values: list[ExtractedValue] = Field(
        default_factory=list,
        description="Concrete values read from tool results or written in tool calls",
    )

    def parsed_symbols(self) -> list[ExtractedSymbol]:
        """Symbols from the result, deduplicated by canonical name."""
        by_name: dict[str, ExtractedSymbol] = {}
        for s in self.symbols:
            canonical = s.name.strip()
            if not canonical:
                continue
            key = canonical.lower()
            prior = by_name.get(key)
            if prior is None:
                by_name[key] = s
            else:
                for alias in s.aliases:
                    if alias not in prior.aliases:
                        prior.aliases.append(alias)
        return list(by_name.values())
