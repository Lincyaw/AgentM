"""The one diagnostics sink every analysis pass writes into.

``transcript`` is the oracle-tuple record (P3): one row per model
judgment, so pass output is a deterministic function of (facts,
transcript). ``prune_log`` records every code-side pruning decision
(P2: no silent false negatives).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
