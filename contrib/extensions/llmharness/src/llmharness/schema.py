"""Typed payloads for the cognitive-audit pipeline.

These dataclasses describe the verdict / event / edge / reminder / finding
shapes the audit child session emits and the adapter consumes.
Persistence lives on the session entry tree
(``api.session.append_entry``); this module is just the typed contract
between the audit prompt schemas, the phase parsers
(``audit.extractor.RawExtractorOutput`` /
``audit.auditor.RawVerdictOutput``), and the adapter.

V3 breaking changes (issue #134, 2026-05-10):
- ``EventKind`` short-form values: ``task``, ``hyp``, ``evid``, ``act``,
  ``dec``, ``concl``. The earlier long forms (``hypothesis``,
  ``evidence``, ``decision``, ``action``, ``conclusion``) are gone, and
  the v2 ``REFLECTION`` member is dropped — design §3 lists six kinds.
- ``Event.refs`` is removed. Edges are now first-class records: the
  extractor emits ``Edge`` instances via ``add_edge`` (design §7.1) and
  the adapter persists them as ``llmharness.audit_edge`` entries.
- New ``Edge`` + ``EdgeKind`` dataclass / enum for those records, with
  witness fields (``cited_entities``, ``cited_quote``) and per-side
  source-turn tuples — see design §4.c, §7.1.
- New ``Finding`` dataclass — output shape for scenario-registered
  audit checks (see ``audit/registry.py``, design §4.c).
- ``Verdict`` shape is unchanged from v2 (design §6.2 / decision #9):
  ``surface_reminder``, ``reminder_text``, ``continuation_notes``,
  ``matched_event_ids``, ``cited_cards``.

V2 breaking changes carried forward unchanged:
- ``DriftType`` enum stays removed. ``Reminder.type`` stays removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """Action-signature classification of an extracted event (design §3)."""

    TASK = "task"
    HYP = "hyp"
    EVID = "evid"
    ACT = "act"
    DEC = "dec"
    CONCL = "concl"


class EdgeKind(str, Enum):
    """Kind of edge between two events (design §4.c, §7.1).

    ``DATA`` — content/data flow (e.g. evidence supports a hypothesis).
    ``REF`` — referential mention (e.g. a decision references a prior task).
    """

    DATA = "data"
    REF = "ref"


@dataclass(frozen=True)
class Event:
    """A compressed semantic event extracted from one or more turns.

    Edges are no longer stored on the event; they are emitted as
    separate :class:`Edge` records. Use the audit graph
    (events + edges, see ``audit.registry.CheckContext``) to traverse
    references.
    """

    id: int
    kind: EventKind
    summary: str
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
            source_turns=list(data.get("source_turns") or []),
        )


@dataclass(frozen=True)
class Edge:
    """A directed witness-bearing edge between two events (design §7.1).

    Mirrors the ``add_edge`` tool-call schema: the extractor must back
    every edge with a citation — entities and/or a verbatim quote — that
    the witness layer can verify against the source turns. The adapter
    persists each accepted edge as a single ``llmharness.audit_edge``
    entry whose payload is :meth:`to_dict`.
    """

    src: int
    dst: int
    kind: EdgeKind
    reason: str
    src_turns: tuple[int, ...]
    dst_turns: tuple[int, ...]
    cited_entities: tuple[str, ...] = ()
    cited_quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind.value,
            "reason": self.reason,
            "src_turns": list(self.src_turns),
            "dst_turns": list(self.dst_turns),
            "cited_entities": list(self.cited_entities),
            "cited_quote": self.cited_quote,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        return cls(
            src=int(data["src"]),
            dst=int(data["dst"]),
            kind=EdgeKind(data["kind"]),
            reason=str(data.get("reason", "")),
            src_turns=tuple(int(t) for t in (data.get("src_turns") or [])),
            dst_turns=tuple(int(t) for t in (data.get("dst_turns") or [])),
            cited_entities=tuple(str(e) for e in (data.get("cited_entities") or [])),
            cited_quote=str(data.get("cited_quote", "")),
        )


@dataclass(frozen=True)
class Finding:
    """Advisory finding emitted by a scenario-registered audit check.

    Per design §4.c, registered checks (see ``audit/registry.py``) run
    over a frozen :class:`~llmharness.audit.registry.CheckContext`
    snapshot at auditor firing time and return a list of
    :class:`Finding` records. The auditor folds these into its prompt
    block as advisory signals — it may ignore, contradict, or extend
    them.
    """

    category: str
    description: str
    related_event_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "related_event_ids": list(self.related_event_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        return cls(
            category=str(data.get("category", "")),
            description=str(data.get("description", "")),
            related_event_ids=tuple(int(i) for i in (data.get("related_event_ids") or [])),
        )


@dataclass(frozen=True)
class Verdict:
    """Auditor output (V2 shape — design §6.2; preserved in v3).

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
    "Edge",
    "EdgeKind",
    "Event",
    "EventKind",
    "Finding",
    "Reminder",
    "Verdict",
]
