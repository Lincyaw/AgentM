"""Typed payloads for the cognitive-audit pipeline.

These dataclasses describe the verdict / event / reminder shapes the audit
child session emits and the adapter consumes. Persistence lives on the
session entry tree (``api.session.append_entry``); this module is just
the typed contract between the audit prompt schemas, the phase parsers
(``audit.extractor.RawExtractorOutput`` /
``audit.auditor.RawVerdictOutput``), and the adapter.
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


class DriftType(str, Enum):
    TASK_DRIFT = "task_drift"
    EVIDENCE_IGNORED = "evidence_ignored"
    PREMATURE_CONCLUSION = "premature_conclusion"
    STUCK_LOOP = "stuck_loop"


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
    """Drift detector output. ``drift=False`` means stay silent."""

    drift: bool
    type: DriftType | None = None
    reminder: str = ""
    matched_event_ids: list[int] = field(default_factory=list)
    cited_cards: list[str] = field(default_factory=list)
    downstream_reaction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift": self.drift,
            "type": self.type.value if self.type is not None else None,
            "reminder": self.reminder,
            "matched_event_ids": list(self.matched_event_ids),
            "cited_cards": list(self.cited_cards),
            "downstream_reaction": self.downstream_reaction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Verdict:
        type_str = data.get("type")
        return cls(
            drift=bool(data.get("drift", False)),
            type=DriftType(type_str) if type_str else None,
            reminder=str(data.get("reminder", "")),
            matched_event_ids=list(data.get("matched_event_ids") or []),
            cited_cards=list(data.get("cited_cards") or []),
            downstream_reaction=data.get("downstream_reaction"),
        )


@dataclass(frozen=True)
class Reminder:
    """A pending reminder waiting to be injected on the next user prompt."""

    type: DriftType
    text: str
