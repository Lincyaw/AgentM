"""Typed payloads for the cognitive-audit pipeline.

These dataclasses describe the verdict / event / reminder shapes the audit
child session emits and the adapter consumes. Persistence lives on the
session entry tree (``api.session.append_entry``); this module is just
the typed contract between the audit prompt schemas, the phase parsers
(``audit.extractor.RawExtractorOutput`` /
``audit.auditor.RawVerdictOutput``), and the adapter.

V2 breaking changes (issue #134, 2026-05-10):
- ``DriftType`` enum removed (preset enum for a subjective dimension —
  violates the project's no-preset-enum rule; free-text
  ``reminder_text`` + ``continuation_notes`` carry the information with
  better fidelity).
- ``Verdict`` shape changed to V2 (design §6.2): ``surface_reminder``,
  ``reminder_text``, ``continuation_notes``, ``matched_event_ids``,
  ``cited_cards``. Fields ``drift``, ``type``, ``reminder``, and
  ``downstream_reaction`` are removed.
- ``Reminder.type`` field removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    TASK = "task"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    DECISION = "decision"
    ACTION = "action"
    REFLECTION = "reflection"
    CONCLUSION = "conclusion"


@dataclass(frozen=True)
class Event:
    """A compressed semantic event extracted from one or more turns."""

    id: int
    kind: EventKind
    summary: str
    refs: list[int] = field(default_factory=list)
    source_turns: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        return cls(
            id=int(data["id"]),
            kind=EventKind(data["kind"]),
            summary=data.get("summary", ""),
            refs=list(data.get("refs") or []),
            source_turns=list(data.get("source_turns") or []),
        )


@dataclass(frozen=True)
class Verdict:
    """Auditor output (V2 shape — design §6.2).

    ``surface_reminder=False`` means stay silent. When ``True``,
    ``reminder_text`` must be non-empty and will be injected before the
    next agent turn.
    """

    surface_reminder: bool
    reminder_text: str = ""
    continuation_notes: list[str] = field(default_factory=list)
    matched_event_ids: list[int] = field(default_factory=list)
    cited_cards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_reminder": self.surface_reminder,
            "reminder_text": self.reminder_text,
            "continuation_notes": list(self.continuation_notes),
            "matched_event_ids": list(self.matched_event_ids),
            "cited_cards": list(self.cited_cards),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Verdict:
        return cls(
            surface_reminder=bool(data.get("surface_reminder", False)),
            reminder_text=str(data.get("reminder_text", "")),
            continuation_notes=[str(n) for n in (data.get("continuation_notes") or [])],
            matched_event_ids=list(data.get("matched_event_ids") or []),
            cited_cards=[str(c) for c in (data.get("cited_cards") or [])],
        )


@dataclass(frozen=True)
class Reminder:
    """A pending reminder waiting to be injected on the next user prompt."""

    text: str


__all__ = [
    "Event",
    "EventKind",
    "Reminder",
    "Verdict",
]
