"""Shared data types and entry-type constants for the cognitive-audit pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


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


TurnKind = Literal["user", "assistant", "tool_call", "tool_result", "system", "reminder"]
EntityType = Literal[
    "service",
    "endpoint",
    "edge",
    "metric",
    "log_pattern",
    "fault_kind",
    "tool",
    "schema_field",
    "unknown",
]
ObservationPolarity = Literal["supports", "weakens", "neutral", "unknown"]
ObservationSignal = Literal[
    "missing_or_normal_only",
    "volume_or_count_drop",
    "volume_or_count_increase",
    "latency_delta",
    "error_delta",
    "resource_delta",
    "weak_or_no_error",
    "schema_or_output_failure",
]
AttentionKind = Literal[
    "competing_observation_cluster",
    "weak_candidate_signal",
    "local_signal_on_disappeared_entity",
]
ClaimKind = Literal["hypothesis", "decision", "demotion", "conclusion", "final_answer"]
CandidateState = Literal["mentioned", "investigated", "retained", "demoted", "finalized"]
ObligationSource = Literal["agent_plan", "methodology", "tool_contract"]
ObligationState = Literal["open", "satisfied", "abandoned", "unknown"]
ContractStatus = Literal["rejected", "empty", "malformed", "validation_failed", "repaired"]
LinkKind = Literal["mentions", "cites", "near", "follows", "same_entity", "derives_from"]


@dataclass(frozen=True)
class TurnRef:
    turn_index: int
    role: str
    kind: TurnKind
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "kind": self.kind,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class EntityRef:
    id: str
    name: str
    type: EntityType
    turns: tuple[int, ...]
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "turns": list(self.turns),
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class ObservationRef:
    id: str
    turns: tuple[int, ...]
    source: str
    summary: str
    entities: tuple[str, ...]
    values: tuple[str, ...] = ()
    polarity: ObservationPolarity = "unknown"
    signals: tuple[ObservationSignal, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turns": list(self.turns),
            "source": self.source,
            "summary": self.summary,
            "entities": list(self.entities),
            "values": list(self.values),
            "polarity": self.polarity,
            "signals": list(self.signals),
        }


@dataclass(frozen=True)
class ClaimRef:
    id: str
    turns: tuple[int, ...]
    text: str
    kind: ClaimKind
    status: str
    entities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turns": list(self.turns),
            "text": self.text,
            "kind": self.kind,
            "status": self.status,
            "entities": list(self.entities),
        }


@dataclass(frozen=True)
class CandidateRef:
    entity_id: str
    first_seen_turn: int | None
    last_seen_turn: int | None
    state: CandidateState
    state_turn: int | None
    reason_claim_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    evidence_tags: tuple[ObservationSignal, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "first_seen_turn": self.first_seen_turn,
            "last_seen_turn": self.last_seen_turn,
            "state": self.state,
            "state_turn": self.state_turn,
            "reason_claim_id": self.reason_claim_id,
            "evidence_ids": list(self.evidence_ids),
            "evidence_tags": list(self.evidence_tags),
        }


@dataclass(frozen=True)
class ObligationRef:
    id: str
    turns: tuple[int, ...]
    source: ObligationSource
    text: str
    entities: tuple[str, ...]
    state: ObligationState = "open"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turns": list(self.turns),
            "source": self.source,
            "text": self.text,
            "entities": list(self.entities),
            "state": self.state,
        }


@dataclass(frozen=True)
class ContractEventRef:
    id: str
    turns: tuple[int, ...]
    tool: str
    status: ContractStatus
    summary: str
    entities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turns": list(self.turns),
            "tool": self.tool,
            "status": self.status,
            "summary": self.summary,
            "entities": list(self.entities),
        }


@dataclass(frozen=True)
class IndexLink:
    src: str
    dst: str
    kind: LinkKind
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AttentionHint:
    id: str
    kind: AttentionKind
    turns: tuple[int, ...]
    summary: str
    entities: tuple[str, ...]
    observation_ids: tuple[str, ...]
    signals: tuple[ObservationSignal, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "turns": list(self.turns),
            "summary": self.summary,
            "entities": list(self.entities),
            "observation_ids": list(self.observation_ids),
            "signals": list(self.signals),
        }


@dataclass(frozen=True)
class ContextIndex:
    turns: tuple[TurnRef, ...]
    entities: tuple[EntityRef, ...]
    observations: tuple[ObservationRef, ...]
    claims: tuple[ClaimRef, ...]
    candidates: tuple[CandidateRef, ...]
    obligations: tuple[ObligationRef, ...]
    contract_events: tuple[ContractEventRef, ...]
    links: tuple[IndexLink, ...]
    attention_hints: tuple[AttentionHint, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "entities": [e.to_dict() for e in self.entities],
            "observations": [o.to_dict() for o in self.observations],
            "claims": [c.to_dict() for c in self.claims],
            "candidates": [c.to_dict() for c in self.candidates],
            "obligations": [o.to_dict() for o in self.obligations],
            "contract_events": [c.to_dict() for c in self.contract_events],
            "links": [link.to_dict() for link in self.links],
            "attention_hints": [hint.to_dict() for hint in self.attention_hints],
        }


@dataclass(frozen=True)
class ExternalRef:
    """Cross-firing reference from this event to a prior-firing event."""

    to_recent_event_id: int  # global event id from a recent_records[i].id
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


# ---------------------------------------------------------------------------
# Entry-type constants for session entries
# ---------------------------------------------------------------------------

AUDIT_INDEX_OP = "llmharness.audit_index_op"
VERDICT = "llmharness.verdict"
EXTRACTOR_CURSOR = "llmharness.extractor_cursor"
REMINDER_DELIVERED = "llmharness.reminder_delivered"

RECENT_VERDICTS_FOR_AUDITOR = 5

__all__ = [
    "AUDIT_INDEX_OP",
    "EXTRACTOR_CURSOR",
    "RECENT_VERDICTS_FOR_AUDITOR",
    "VERDICT",
    "AttentionHint",
    "AttentionKind",
    "CandidateRef",
    "CandidateState",
    "ClaimKind",
    "ClaimRef",
    "CommitmentStatus",
    "ContextIndex",
    "ContractEventRef",
    "ContractStatus",
    "Edge",
    "EdgeKind",
    "EdgeRole",
    "EntityRef",
    "EntityType",
    "Event",
    "EventKind",
    "ExternalRef",
    "IndexLink",
    "LinkKind",
    "ObligationRef",
    "ObligationSource",
    "ObligationState",
    "ObservationPolarity",
    "ObservationRef",
    "ObservationSignal",
    "Reminder",
    "TurnKind",
    "TurnRef",
    "Verdict",
]
