"""Word-level robust string matching.

Matches by word sequences (``\\w+``), making results invariant under
whitespace, markdown formatting, and punctuation differences between
the query and the content.

    >>> from trajectory_index.locate import locate
    >>> locate("at most 12 passes", "budget: at most **12** full passes.")
    (8, 34)
"""

from __future__ import annotations

import re
from typing import Final

_WORD_RE: Final = re.compile(r"\w+", re.UNICODE)
_INLINE_TAG_RE: Final = re.compile(r"⟦[^|⟧]*\|([^⟧]*)⟧")


def strip_tags(text: str) -> str:
    """Remove inline annotation tags, keeping only their content."""
    return _INLINE_TAG_RE.sub(r"\1", text)


def _build_pattern(text: str) -> re.Pattern[str]:
    words = _WORD_RE.findall(text)
    if not words:
        return re.compile(r"(?!)")  # never matches
    return re.compile(
        r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b",
        re.IGNORECASE | re.DOTALL,
    )


def locate(query: str, content: str, *, after: int = 0) -> tuple[int, int] | None:
    """Find *query* in *content* by word-sequence matching.

    Returns ``(char_start, char_end)`` in *content*, or ``None``.
    Only considers matches starting at or after *after*.

    Supports the ``…`` wildcard: ``"head…tail"`` matches the head
    word-sequence, then the tail word-sequence appearing after it.
    The returned span covers from the start of the head match to the
    end of the tail match.
    """
    query = strip_tags(query)
    parts = query.split("…")
    if len(parts) == 1:
        m = _build_pattern(query).search(content, after)
        return (m.start(), m.end()) if m else None

    head_m = _build_pattern(parts[0]).search(content, after)
    if not head_m:
        return None
    tail_m = _build_pattern(parts[-1]).search(content, head_m.end())
    if not tail_m:
        return None
    return head_m.start(), tail_m.end()
