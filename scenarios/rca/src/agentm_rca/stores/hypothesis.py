"""Hypothesis Store -- thread-safe, run-scoped hypothesis management."""

from __future__ import annotations

from dataclasses import dataclass

from ._threadsafe import ThreadSafeStore


@dataclass(frozen=True)
class HypothesisEntry:
    """A single hypothesis tracked by the orchestrator."""

    id: str
    description: str
    status: str = "formed"
    evidence: tuple[str, ...] = ()
    counter_evidence: tuple[str, ...] = ()
    parent_id: str | None = None


class HypothesisStore(ThreadSafeStore[str, HypothesisEntry]):
    """Thread-safe, run-scoped hypothesis store."""

    _VALID_STATUSES = frozenset(
        {"formed", "investigating", "confirmed", "rejected", "refined", "inconclusive"}
    )

    def __init__(self) -> None:
        super().__init__()
        self._confirmed_id: str | None = None

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
            existing = self._data.get(id)
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
            self._data[id] = entry

            if status == "confirmed":
                self._confirmed_id = id

            return entry

    def remove(self, id: str) -> bool:
        """Remove a hypothesis. Returns True if it existed."""
        with self._lock:
            if id in self._data:
                del self._data[id]
                if self._confirmed_id == id:
                    self._confirmed_id = None
                return True
            return False

    def get(self, id: str) -> HypothesisEntry | None:
        return super().get(id)

    def get_all(self) -> dict[str, HypothesisEntry]:
        return super().get_all()

    @property
    def confirmed_id(self) -> str | None:
        with self._lock:
            return self._confirmed_id

    def format_for_llm(self) -> str:
        """Format all hypotheses for the LLM context message."""
        with self._lock:
            hypotheses = list(self._data.values())
            confirmed = self._confirmed_id

        if not hypotheses:
            return ""

        lines: list[str] = ["## Hypotheses"]
        for hypothesis in hypotheses:
            lines.append(
                f"### [{hypothesis.status.upper()}] {hypothesis.id}: {hypothesis.description}"
            )
            if hypothesis.evidence:
                lines.append("Evidence:")
                for evidence in hypothesis.evidence:
                    lines.append(f"  + {evidence}")
            if hypothesis.counter_evidence:
                lines.append("Counter-evidence:")
                for counter_evidence in hypothesis.counter_evidence:
                    lines.append(f"  - {counter_evidence}")
            if hypothesis.parent_id:
                lines.append(f"  (refined from {hypothesis.parent_id})")

        if confirmed:
            lines.append("")
            lines.append(f"**Confirmed Root Cause: {confirmed}**")

        return "\n".join(lines)
