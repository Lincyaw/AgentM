"""Hypothesis Store — thread-safe, run-scoped hypothesis management.

Same pattern as ServiceProfileStore: closure-injected into tools,
independent of LangGraph state. The orchestrator reads the formatted
output via format_context each round.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class HypothesisEntry:
    """A single hypothesis tracked by the orchestrator."""

    id: str
    description: str
    status: str = "formed"  # formed|investigating|confirmed|rejected|refined|inconclusive
    evidence: tuple[str, ...] = ()
    counter_evidence: tuple[str, ...] = ()
    parent_id: str | None = None


class HypothesisStore:
    """Thread-safe, run-scoped hypothesis store.

    Mirrors ServiceProfileStore's design — tools hold a closure reference,
    no LangGraph state injection needed.
    """

    _VALID_STATUSES = frozenset(
        {"formed", "investigating", "confirmed", "rejected", "refined", "inconclusive"}
    )

    def __init__(self) -> None:
        self._hypotheses: dict[str, HypothesisEntry] = {}
        self._confirmed_id: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(
        self,
        id: str,
        description: str,
        status: str = "formed",
        evidence_summary: str | None = None,
        parent_id: str | None = None,
    ) -> HypothesisEntry:
        """Create or update a hypothesis. Returns the new entry."""
        if status not in self._VALID_STATUSES:
            raise ValueError(f"Invalid status: {status!r}")

        with self._lock:
            existing = self._hypotheses.get(id)
            if existing is None:
                entry = HypothesisEntry(
                    id=id,
                    description=description,
                    status=status,
                    evidence=(evidence_summary,) if evidence_summary else (),
                    parent_id=parent_id,
                )
            else:
                new_evidence = existing.evidence
                if evidence_summary:
                    new_evidence = (*existing.evidence, evidence_summary)
                entry = HypothesisEntry(
                    id=id,
                    description=description,
                    status=status,
                    evidence=new_evidence,
                    counter_evidence=existing.counter_evidence,
                    parent_id=parent_id or existing.parent_id,
                )
            self._hypotheses[id] = entry

            if status == "confirmed":
                self._confirmed_id = id

            return entry

    def remove(self, id: str) -> bool:
        """Remove a hypothesis. Returns True if it existed."""
        with self._lock:
            if id in self._hypotheses:
                del self._hypotheses[id]
                if self._confirmed_id == id:
                    self._confirmed_id = None
                return True
            return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, id: str) -> HypothesisEntry | None:
        with self._lock:
            return self._hypotheses.get(id)

    def get_all(self) -> dict[str, HypothesisEntry]:
        with self._lock:
            return dict(self._hypotheses)

    @property
    def confirmed_id(self) -> str | None:
        with self._lock:
            return self._confirmed_id

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_llm(self) -> str:
        """Format all hypotheses for the LLM context message."""
        with self._lock:
            hypotheses = list(self._hypotheses.values())
            confirmed = self._confirmed_id

        if not hypotheses:
            return ""

        lines: list[str] = ["## Hypotheses"]
        for h in hypotheses:
            lines.append(f"### [{h.status.upper()}] {h.id}: {h.description}")
            if h.evidence:
                lines.append("Evidence:")
                for e in h.evidence:
                    lines.append(f"  + {e}")
            if h.counter_evidence:
                lines.append("Counter-evidence:")
                for ce in h.counter_evidence:
                    lines.append(f"  - {ce}")
            if h.parent_id:
                lines.append(f"  (refined from {h.parent_id})")

        if confirmed:
            lines.append("")
            lines.append(f"**Confirmed Root Cause: {confirmed}**")

        return "\n".join(lines)
