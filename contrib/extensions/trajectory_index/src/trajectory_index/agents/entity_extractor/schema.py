"""Extraction result schema.

Pass 1 output is the unified annotation markup (see ``markup.py``): the
extractor re-emits each annotated message body verbatim with
``⟦tag attrs|content⟧`` spans inserted. One output channel carries every
node kind — symbols (``sym``), provenance segments (``obs``), claims
(``claim``) — and code verifies the whole message by strip-and-compare,
which makes every span offset exact. Messages without annotations are
omitted (nothing to re-emit).

``ExtractedSymbol`` remains as the shape of code-side structural prescan
symbols and of symbols parsed back out of the markup.
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
            "The name/value axis, independent of `kind`. "
            "Could a tool report different content for this entity while it stays the same thing?\n"
            "- 'identifier': NO — the string IS the entity (a service name, file path, tool name). Most symbols.\n"
            "- 'value': YES — a tracked quantity the agent monitors across turns; the same measurement can have different values at different times.\n"
            "- 'unknown': a vague or anaphoric surface with no clear referent on its own."
        ),
    )


class AnnotatedMessage(BaseModel):
    """One message body re-emitted verbatim with annotations inserted."""

    message_id: str = Field(description="Message id from the [id|role] header")
    text: str = Field(
        description=(
            "The message body, copied EXACTLY as given, with "
            "⟦tag attrs|content⟧ annotations inserted. Stripping the "
            "annotations must reproduce the original body character for "
            "character."
        ),
    )


class ExtractionResult(BaseModel):
    annotated: list[AnnotatedMessage] = Field(
        default_factory=list,
        description="Messages that carry at least one annotation",
    )
