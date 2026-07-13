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

    def parsed_symbols(self) -> list[ExtractedSymbol]:
        """Symbols declared in the markup, deduplicated by canonical name.

        Registry/bookkeeping view only — index population re-parses with
        offset verification. Malformed messages are skipped here (populate
        records them properly).
        """
        from trajectory_index.markup import MarkupError, parse

        by_name: dict[str, ExtractedSymbol] = {}
        for am in self.annotated:
            try:
                plain, annotations = parse(am.text)
            except MarkupError:
                continue
            for a in annotations:
                if a.tag != "sym":
                    continue
                surface = plain[a.start:a.end].strip()
                canonical = (a.attrs.get("name") or surface).strip()
                if not canonical:
                    continue
                alias = [surface] if surface and surface.lower() != canonical.lower() else []
                prior = by_name.get(canonical.lower())
                if prior is None:
                    by_name[canonical.lower()] = ExtractedSymbol(
                        name=canonical,
                        kind=a.attrs.get("kind", "unknown"),
                        aliases=alias,
                        entity_class=a.attrs.get("class", "identifier"),
                    )
                else:
                    for al in alias:
                        if al not in prior.aliases:
                            prior.aliases.append(al)
        return list(by_name.values())
