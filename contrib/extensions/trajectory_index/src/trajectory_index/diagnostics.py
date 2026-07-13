"""The one diagnostics sink every analysis pass writes into.

``transcript`` is the oracle-tuple record (P3): one row per model
judgment, so pass output is a deterministic function of (facts,
transcript). ``prune_log`` records every code-side pruning decision
(P2: no silent false negatives).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_MIN_LEXICAL_TOKEN = 4        # lexical token length floor
_STOPWORDS = frozenset({
    "the", "and", "that", "with", "from", "this", "have", "been", "were",
    "was", "for", "are", "not", "his", "her", "their", "its", "who", "than",
    "then", "when", "what", "which", "where", "before", "after", "during",
    "into", "about", "there", "they", "them", "also", "some", "same",
})


def content_tokens(text: str) -> set[str]:
    """Content-bearing lowercase tokens — the shared lexical primitive."""
    return {
        t for t in re.split(r"\W+", text.lower())
        if len(t) >= _MIN_LEXICAL_TOKEN and t not in _STOPWORDS
    }


@dataclass(slots=True)
class Diagnostics:
    transcript: list[dict[str, Any]] = field(default_factory=list)
    prune_log: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self, relation: str, key: str, verdict: Any,
        confidence: float, detail: str = "",
    ) -> None:
        self.transcript.append({
            "relation": relation, "key": key, "verdict": verdict,
            "confidence": round(confidence, 3), "detail": detail,
        })

    def prune(self, stage: str, what: str, why: str) -> None:
        self.prune_log.append({"stage": stage, "what": what, "why": why})
