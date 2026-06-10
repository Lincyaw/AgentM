"""Witness validation for extractor edges.

Deterministic substring checks — no LLM needed. Returns None on success,
a structured error string on the first failure.
"""

from __future__ import annotations

import re

_WS_RUN = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lowercase + collapse runs of whitespace to a single space."""
    return _WS_RUN.sub(" ", s.lower()).strip()


def witness_data(
    cited_entities: list[str],
    src_text: str,
    dst_text: str,
) -> str | None:
    """Verify every entity in cited_entities appears in src OR dst."""
    if not cited_entities:
        return "witness/data: cited_entities must be non-empty for kind='data'"
    src_norm = normalize(src_text)
    dst_norm = normalize(dst_text)
    for entity in cited_entities:
        ent_norm = normalize(entity)
        if not ent_norm:
            return "witness/data: cited entity is empty after normalization"
        if ent_norm not in src_norm and ent_norm not in dst_norm:
            return (
                f"witness/data: cited entity {entity!r} not found in normalized "
                "src_turns OR dst_turns text"
            )
    return None


def witness_ref(
    cited_quote: str,
    src_text: str,
    dst_text: str,
) -> str | None:
    """Verify cited_quote appears verbatim (mod normalize) in src OR dst."""
    if not cited_quote:
        return "witness/ref: cited_quote must be non-empty for kind='ref'"
    quote_norm = normalize(cited_quote)
    if not quote_norm:
        return "witness/ref: cited_quote is empty after normalization"
    src_norm = normalize(src_text)
    dst_norm = normalize(dst_text)
    if quote_norm not in src_norm and quote_norm not in dst_norm:
        return (
            f"witness/ref: cited_quote {cited_quote!r} not found in normalized "
            "src_turns OR dst_turns text"
        )
    return None


__all__ = ["normalize", "witness_data", "witness_ref"]
