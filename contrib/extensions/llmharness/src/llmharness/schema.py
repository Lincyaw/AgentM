"""Shared data types and entry-type constants for the cognitive-audit pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """Action-signature classification of an extracted event."""

    TASK = "task"
    HYP = "hyp"
    ACT = "act"
    DEC = "dec"
    CONCL = "concl"


class CommitmentStatus(str, Enum):
    """How committed the agent is to a hyp/dec/concl claim."""

    EXPLORATORY = "exploratory"
    TENTATIVE = "tentative"
    COMMITTED = "committed"
    FINALIZED = "finalized"


class EdgeKind(str, Enum):
    """Kind of edge between two events."""

    DATA = "data"
    REF = "ref"


class EdgeRole(str, Enum):
    """Causal role of an edge between two events."""

    SUPPORTS = "supports"
    WEAKENS = "weakens"
    DEPENDS = "depends"
    NARROWS = "narrows"


@dataclass(frozen=True)
class ExternalRef:
    """Cross-firing reference from this event to a prior-firing event."""

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
    """A semantic event extracted from one or more turns."""

    id: int
    kind: EventKind
    summary: str
    source_turns: list[int] = field(default_factory=list)
    external_refs: tuple[ExternalRef, ...] = ()
    status: CommitmentStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind.value,
            "summary": self.summary,
            "source_turns": list(self.source_turns),
            "external_refs": [r.to_dict() for r in self.external_refs],
        }
        if self.status is not None:
            d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        raw_status = data.get("status")
        return cls(
            id=int(data["id"]),
            kind=EventKind(data["kind"]),
            summary=data.get("summary", ""),
            source_turns=list(data.get("source_turns") or []),
            external_refs=tuple(
                ExternalRef.from_dict(r) for r in (data.get("external_refs") or [])
            ),
            status=CommitmentStatus(raw_status) if raw_status is not None else None,
        )


@dataclass(frozen=True)
class Edge:
    """A directed witness-bearing edge between two events."""

    src: int
    dst: int
    kind: EdgeKind
    reason: str
    src_turns: tuple[int, ...]
    dst_turns: tuple[int, ...]
    cited_entities: tuple[str, ...] = ()
    cited_quote: str = ""
    role: EdgeRole | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind.value,
            "reason": self.reason,
            "src_turns": list(self.src_turns),
            "dst_turns": list(self.dst_turns),
            "cited_entities": list(self.cited_entities),
            "cited_quote": self.cited_quote,
        }
        if self.role is not None:
            d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        raw_role = data.get("role")
        return cls(
            src=int(data["src"]),
            dst=int(data["dst"]),
            kind=EdgeKind(data["kind"]),
            reason=str(data.get("reason", "")),
            src_turns=tuple(int(t) for t in (data.get("src_turns") or [])),
            dst_turns=tuple(int(t) for t in (data.get("dst_turns") or [])),
            cited_entities=tuple(str(e) for e in (data.get("cited_entities") or [])),
            cited_quote=str(data.get("cited_quote", "")),
            role=EdgeRole(raw_role) if raw_role is not None else None,
        )


@dataclass(frozen=True)
class Finding:
    """Advisory finding from a scenario-registered audit check."""

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
    """Auditor verdict. surface_reminder=True triggers reminder injection."""

    surface_reminder: bool
    reminder_text: str = ""
    continuation_notes: list[str] = field(default_factory=list)
    matched_event_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_reminder": self.surface_reminder,
            "reminder_text": self.reminder_text,
            "continuation_notes": list(self.continuation_notes),
            "matched_event_ids": list(self.matched_event_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Verdict:
        return cls(
            surface_reminder=bool(data.get("surface_reminder", False)),
            reminder_text=str(data.get("reminder_text", "")),
            continuation_notes=[str(n) for n in (data.get("continuation_notes") or [])],
            matched_event_ids=list(data.get("matched_event_ids") or []),
        )


@dataclass(frozen=True)
class Reminder:
    """A pending reminder waiting to be injected on the next user prompt."""

    text: str


@dataclass(frozen=True)
class Phase:
    """A merged basic block over consecutive raw events."""

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


# ---------------------------------------------------------------------------
# Entry-type constants for session entries
# ---------------------------------------------------------------------------

AUDIT_GRAPH_OP = "llmharness.audit_graph_op"
VERDICT = "llmharness.verdict"
EXTRACTOR_CURSOR = "llmharness.extractor_cursor"
REMINDER_DELIVERED = "llmharness.reminder_delivered"

EXTRACTOR_NO_CALL = "llmharness.extractor_no_call"
EXTRACTOR_ERROR = "llmharness.extractor_error"
EXTRACTOR_EMPTY = "llmharness.extractor_empty"
EXTRACTOR_PARTIAL = "llmharness.extractor_partial"
AUDIT_NO_CALL = "llmharness.audit_no_call"
AUDIT_ERROR = "llmharness.audit_error"

MESSAGE = "message"

RECENT_VERDICTS_FOR_AUDITOR = 5

__all__ = [
    "AUDIT_ERROR",
    "AUDIT_GRAPH_OP",
    "AUDIT_NO_CALL",
    "EXTRACTOR_CURSOR",
    "EXTRACTOR_EMPTY",
    "EXTRACTOR_ERROR",
    "EXTRACTOR_NO_CALL",
    "EXTRACTOR_PARTIAL",
    "MESSAGE",
    "RECENT_VERDICTS_FOR_AUDITOR",
    "VERDICT",
    "CommitmentStatus",
    "Edge",
    "EdgeKind",
    "EdgeRole",
    "Event",
    "EventKind",
    "ExternalRef",
    "Finding",
    "Phase",
    "Reminder",
    "Verdict",
]
