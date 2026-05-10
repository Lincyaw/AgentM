"""Pure-Python witness validation for extractor edges (design §4.b, §4.f).

This module is mechanism, not an atom. ``normalize`` lowercases and
collapses runs of whitespace to a single space; ``witness_data`` and
``witness_ref`` use it to verify that every cited entity / quote
appears in BOTH the source-turn text and the destination-turn text of
an edge. Validation is deterministic and reproducible without an LLM.

Returns ``None`` on success and a structured error string on the first
failure, so the caller (``extractor.tools.add_edge``) can echo the
message to the LLM as a tool-result error and let it retry.
"""

from __future__ import annotations

import re

_WS_RUN = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lowercase + collapse runs of whitespace to a single space.

    Leading and trailing whitespace are stripped after collapse so that
    ``"  Hello  World "`` and ``"hello world"`` compare equal. No
    edit-distance, no stemming — see design §4.b.
    """

    return _WS_RUN.sub(" ", s.lower()).strip()


def witness_data(
    cited_entities: list[str],
    src_text: str,
    dst_text: str,
) -> str | None:
    """Verify every entity in ``cited_entities`` appears in both src and dst.

    Returns ``None`` on pass, a structured error string on first failure.
    Empty ``cited_entities`` is rejected — callers must enforce
    non-emptiness before calling so the error is precise.
    """

    if not cited_entities:
        return "witness/data: cited_entities must be non-empty for kind='data'"
    src_norm = normalize(src_text)
    dst_norm = normalize(dst_text)
    for entity in cited_entities:
        ent_norm = normalize(entity)
        if not ent_norm:
            return "witness/data: cited entity is empty after normalization"
        if ent_norm not in src_norm:
            return f"witness/data: cited entity {entity!r} not found in normalized src_turns text"
        if ent_norm not in dst_norm:
            return f"witness/data: cited entity {entity!r} not found in normalized dst_turns text"
    return None


def witness_ref(
    cited_quote: str,
    src_text: str,
    dst_text: str,
) -> str | None:
    """Verify ``cited_quote`` appears verbatim (mod normalize) in both texts.

    Returns ``None`` on pass, a structured error string on first failure.
    Empty quote is rejected up front.
    """

    if not cited_quote:
        return "witness/ref: cited_quote must be non-empty for kind='ref'"
    quote_norm = normalize(cited_quote)
    if not quote_norm:
        return "witness/ref: cited_quote is empty after normalization"
    src_norm = normalize(src_text)
    dst_norm = normalize(dst_text)
    if quote_norm not in src_norm:
        return f"witness/ref: cited_quote {cited_quote!r} not found in normalized src_turns text"
    if quote_norm not in dst_norm:
        return f"witness/ref: cited_quote {cited_quote!r} not found in normalized dst_turns text"
    return None


__all__ = ["normalize", "witness_data", "witness_ref"]
