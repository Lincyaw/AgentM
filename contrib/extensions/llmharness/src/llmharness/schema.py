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
- ``Event.refs`` is removed at the schema level. Edges are first-class
  records persisted as ``llmharness.audit_edge`` entries. (V3.1 lets
  the extractor LLM submit events with embedded ``refs[]`` in a single
  ``submit_events`` call, but those refs are validated and unrolled
  into ``Edge`` instances inside ``ExtractionState.commit`` — the
  schema-level wire format remains ``Event`` + separate ``Edge``.)
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

from dataclasses import dataclass, field
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
class ExternalRef:
    """A reference from a new event back into ``recent_graph`` — i.e. to
    an event extracted by a PRIOR firing.

    Stored on :class:`Event` (not as a free-standing ``Edge``) because the
    referenced event has only a local id from its own firing at the time
    the extractor submits. The aggregator resolves these to real edges
    in the cumulative global id space (see ``aggregate.collector``).

    Witnesses are validated by the live harness, same rules as in-firing
    refs: ``data`` requires non-empty ``cited_entities``; ``ref`` requires
    a non-empty ``cited_quote``; both anchors must appear in BOTH the
    source event's source-turns text and this event's source-turns text.
    """

    to_recent_event_id: int  # global event id from a recent_graph[i].id
    kind: EdgeKind
    reason: str = ""
    cited_entities: tuple[str, ...] = ()
    cited_quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "to_recent_event_id": self.to_recent_event_id,
            "kind": self.kind.value,
            "reason": self.reason,
            "cited_entities": list(self.cited_entities),
            "cited_quote": self.cited_quote,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalRef:
        return cls(
            to_recent_event_id=int(data["to_recent_event_id"]),
            kind=EdgeKind(data["kind"]),
            reason=str(data.get("reason", "")),
            cited_entities=tuple(str(e) for e in (data.get("cited_entities") or [])),
            cited_quote=str(data.get("cited_quote", "")),
        )


@dataclass(frozen=True)
class Event:
    """A compressed semantic event extracted from one or more turns.

    In-firing refs are emitted as separate :class:`Edge` records (see
    ``audit.registry.CheckContext``). Cross-firing refs live on the
    event itself as ``external_refs`` and are resolved into edges by
    the offline aggregator (`_accumulate_graph`).
    """

    id: int
    kind: EventKind
    summary: str
    source_turns: list[int] = field(default_factory=list)
    external_refs: tuple[ExternalRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "summary": self.summary,
            "source_turns": list(self.source_turns),
            "external_refs": [r.to_dict() for r in self.external_refs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        return cls(
            id=int(data["id"]),
            kind=EventKind(data["kind"]),
            summary=data.get("summary", ""),
            source_turns=list(data.get("source_turns") or []),
            external_refs=tuple(
                ExternalRef.from_dict(r) for r in (data.get("external_refs") or [])
            ),
        )


@dataclass(frozen=True)
class Edge:
    """A directed witness-bearing edge between two events (design §7.1).

    Mirrors the witness-bearing ref the extractor LLM emits inside an
    event's ``refs[]`` (V3.1 ``submit_events`` payload): each ref
    carries entities and/or a verbatim quote that the witness layer
    verifies against the source turns. The adapter persists each
    accepted edge as a single ``llmharness.audit_edge`` entry whose
    payload is :meth:`to_dict`.
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


@dataclass(frozen=True)
class Phase:
    """A merged "basic block" over consecutive raw events.

    Raw events are extracted one-per-turn; phases group consecutive
    ``act`` / ``evid`` events into a single block while keeping
    ``task`` / ``hyp`` / ``dec`` / ``concl`` as singleton phases. The
    adapter persists phases as ``llmharness.audit_phase`` entries
    alongside raw events; the auditor reads the phase view for
    high-level reasoning and drills back to raw events via
    ``get_event_detail`` when needed.

    ``id`` is per-firing fresh-numbered (1, 2, 3, ...) — same convention
    as raw event ids.

    ``kind`` is one of the :class:`EventKind` values for singleton
    phases (``task`` / ``hyp`` / ``dec`` / ``concl`` / ``act`` / ``evid``
    when only one such event is in the run), plus the merged-run
    sentinel ``act_evid_run`` when two or more ``act`` / ``evid`` events
    are coalesced. Free-text rather than enum so future merge rules can
    introduce new run types without a schema bump.
    """

    id: int
    kind: str
    member_event_ids: tuple[int, ...]
    source_turns: tuple[int, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "member_event_ids": list(self.member_event_ids),
            "source_turns": list(self.source_turns),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Phase:
        return cls(
            id=int(data["id"]),
            kind=str(data["kind"]),
            member_event_ids=tuple(int(i) for i in (data.get("member_event_ids") or [])),
            source_turns=tuple(int(t) for t in (data.get("source_turns") or [])),
            summary=str(data.get("summary", "")),
        )


__all__ = [
    "Edge",
    "EdgeKind",
    "Event",
    "EventKind",
    "ExternalRef",
    "Finding",
    "Phase",
    "Reminder",
    "Verdict",
]
