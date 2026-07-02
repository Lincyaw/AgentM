"""Shared data types and entry-type constants for the cognitive-audit pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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

RECENT_VERDICTS_FOR_AUDITOR = 5

__all__ = [
    "EXTRACTOR_CURSOR",
    "RECENT_VERDICTS_FOR_AUDITOR",
    "VERDICT",
    "AttentionHint",
    "AttentionKind",
    "CandidateRef",
    "CandidateState",
    "ClaimKind",
    "ClaimRef",
    "ContextIndex",
    "ContractEventRef",
    "ContractStatus",
    "EntityRef",
    "EntityType",
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
