"""Extraction result schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedSymbol(BaseModel):
    name: str = Field(description="Canonical symbol name")
    kind: str = Field(description="Symbol kind from the vocabulary")
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


class ExtractedClaim(BaseModel):
    """A settled-fact assertion by the agent, extracted verbatim.

    Claims are first-class extraction output alongside symbols: the
    trajectory is visited once and every downstream pass (source-claim
    consistency, constraint linkage, commitment detection) consumes the
    same extracted claims instead of re-scanning the trajectory.
    """

    message_id: str = Field(description="Message id the claim sentence appears in")
    text: str = Field(description="The claim sentence, copied verbatim")


class ExtractionResult(BaseModel):
    symbols: list[ExtractedSymbol] = Field(description="All extracted symbols")
    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="Verification/sourcing assertions the agent presents as settled",
    )
