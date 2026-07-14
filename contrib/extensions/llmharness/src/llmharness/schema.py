"""Shared data types and entry-type constants for the cognitive-audit pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Verdict:
    """Auditor verdict. surface_reminder=True triggers reminder injection."""

    surface_reminder: bool
    reminder_text: str = ""
    evidence: list[str] = field(default_factory=list)
    continuation_notes: list[str] = field(default_factory=list)
    matched_event_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_reminder": self.surface_reminder,
            "reminder_text": self.reminder_text,
            "evidence": list(self.evidence),
            "continuation_notes": list(self.continuation_notes),
            "matched_event_ids": list(self.matched_event_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Verdict:
        return cls(
            surface_reminder=bool(data.get("surface_reminder", False)),
            reminder_text=str(data.get("reminder_text", "")),
            evidence=[str(e) for e in (data.get("evidence") or [])],
            continuation_notes=[str(n) for n in (data.get("continuation_notes") or [])],
            matched_event_ids=list(data.get("matched_event_ids") or []),
        )


@dataclass(frozen=True, slots=True)
class Reminder:
    """A pending reminder waiting to be injected on the next user prompt."""

    text: str


# ---------------------------------------------------------------------------
# Entry-type constants for session entries
# ---------------------------------------------------------------------------

VERDICT = "llmharness.verdict"
EXTRACTOR_CURSOR = "llmharness.extractor_cursor"
REMINDER_DELIVERED = "llmharness.reminder_delivered"
REMINDER_SUPPRESSED = "llmharness.reminder_suppressed"

RECENT_VERDICTS_FOR_AUDITOR = 5

__all__ = [
    "EXTRACTOR_CURSOR",
    "RECENT_VERDICTS_FOR_AUDITOR",
    "REMINDER_DELIVERED",
    "REMINDER_SUPPRESSED",
    "VERDICT",
    "Reminder",
    "Verdict",
]
