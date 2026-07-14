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

_MIN_LEXICAL_TOKEN = 4        # lexical token length floor (alphabetic scripts)
_STOPWORDS = frozenset({
    "the", "and", "that", "with", "from", "this", "have", "been", "were",
    "was", "for", "are", "not", "his", "her", "their", "its", "who", "than",
    "then", "when", "what", "which", "where", "before", "after", "during",
    "into", "about", "there", "they", "them", "also", "some", "same",
})

# CJK scripts don't whitespace-segment: a run like 服务网关超时 arrives as ONE
# \w+ token and would never overlap another text's tokens, silently
# disabling every lexical guard for Chinese/Japanese/Korean content.
# Character bigrams give CJK the overlap signal words give alphabetic text.
_CJK_RE = re.compile(r"[㐀-鿿぀-ヿ가-힯]")


def content_tokens(text: str) -> set[str]:
    """Content-bearing lowercase tokens — the shared lexical primitive.

    Alphabetic tokens: length floor + stopword filter. CJK runs:
    character bigrams (plus the lone character for length-1 runs).
    Deterministic; no semantics — this feeds NEGATIVE-only lexical guards.
    """
    tokens: set[str] = set()
    for t in re.split(r"\W+", text.lower()):
        if not t:
            continue
        if _CJK_RE.search(t):
            if len(t) == 1:
                tokens.add(t)
            tokens.update(t[i:i + 2] for i in range(len(t) - 1))
        elif len(t) >= _MIN_LEXICAL_TOKEN and t not in _STOPWORDS:
            tokens.add(t)
    return tokens


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
