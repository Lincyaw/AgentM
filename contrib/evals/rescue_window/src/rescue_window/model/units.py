"""Measurement data model for rescue-window experiments.

The ``EvalUnit`` is the doc §6.2 ``z`` — one row per rollout. It is the only
artifact the rollout layer produces and the only input the analysis layer
consumes. Keeping it a plain, fully-serializable record is what lets metrics be
pure functions over rows (DESIGN §1, §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .schema import ActionType, ForkPoint, Intervention


class ContentLevel(StrEnum):
    """The doc §7.2 content ladder, from no-info to full oracle."""

    CONTINUE = "CONTINUE"
    PLACEBO = "PLACEBO"
    GENERIC = "GENERIC"
    TYPE = "TYPE"
    TYPE_TARGET = "TYPE_TARGET"
    EVIDENCE = "EVIDENCE"
    ORACLE_GROUNDED = "ORACLE_GROUNDED"
    ORACLE_DIAG = "ORACLE_DIAG"


class LadderRung(StrEnum):
    """The doc §4.2 Rescuability Ladder rungs."""

    STATE = "state"
    ACTOR = "actor"
    CHANNEL = "channel"
    OBSERVABLE = "observable"
    BOUNDED = "bounded"
    REALIZED = "realized"


@dataclass(frozen=True, slots=True)
class PrefixPoint:
    """A sampled fork point on one trajectory (doc §6.4)."""

    trajectory_id: str
    case_id: str
    repository_id: str
    prefix_id: str
    fork_point: ForkPoint
    turn_index: int
    progress: float
    stratum: str
    event: str | None = None
    weight: float = 1.0
    remaining_budget: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "case_id": self.case_id,
            "repository_id": self.repository_id,
            "prefix_id": self.prefix_id,
            "fork_point": self.fork_point.to_dict(),
            "turn_index": self.turn_index,
            "progress": self.progress,
            "stratum": self.stratum,
            "event": self.event,
            "weight": self.weight,
            "remaining_budget": self.remaining_budget,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrefixPoint":
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            case_id=str(data["case_id"]),
            repository_id=str(data["repository_id"]),
            prefix_id=str(data["prefix_id"]),
            fork_point=ForkPoint.from_dict(dict(data["fork_point"])),
            turn_index=int(data["turn_index"]),
            progress=float(data["progress"]),
            stratum=str(data["stratum"]),
            event=data.get("event"),
            weight=float(data.get("weight", 1.0)),
            remaining_budget=dict(data.get("remaining_budget") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class Treatment:
    """One branch condition: a typed intervention at a content level + rung."""

    treatment_id: str
    content_level: ContentLevel
    action: ActionType
    intervention: Intervention
    rung: LadderRung = LadderRung.CHANNEL
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_continue(self) -> bool:
        return self.content_level is ContentLevel.CONTINUE

    def to_dict(self) -> dict[str, Any]:
        return {
            "treatment_id": self.treatment_id,
            "content_level": self.content_level.value,
            "action": self.action.value,
            "intervention": self.intervention.to_dict(),
            "rung": self.rung.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Treatment":
        return cls(
            treatment_id=str(data["treatment_id"]),
            content_level=ContentLevel(data["content_level"]),
            action=ActionType(data["action"]),
            intervention=Intervention.from_dict(dict(data["intervention"])),
            rung=LadderRung(data.get("rung", LadderRung.CHANNEL.value)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class EvalUnit:
    """One rollout outcome — the doc §6.2 evaluation unit ``z``."""

    # identity / replayable coordinates
    case_id: str
    repository_id: str
    trajectory_id: str
    prefix_id: str
    fork_point: ForkPoint
    progress: float
    # treatment
    treatment_id: str
    content_level: ContentLevel
    action: ActionType
    intervention: Intervention
    rung: LadderRung
    branch_seed: int
    # rollout result
    status: str = "succeeded"  # succeeded | failed | skipped
    fork_session_id: str | None = None
    error: str | None = None
    actor_id: str = ""
    remaining_budget: dict[str, Any] = field(default_factory=dict)
    # outcome (judge)
    binary_success: bool | None = None
    normalized_score: float | None = None
    final_state_hash: str | None = None
    judge_detail: dict[str, Any] = field(default_factory=dict)
    # behavior / cost
    follow_through: bool | None = None
    stale: bool = False
    duplicate: bool = False
    wasted_steps: int | None = None
    critic_latency_ms: int | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    # analysis
    sampling_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cluster_key(self) -> str:
        """All branches + K of one prefix share this cluster (doc §9.4)."""
        return self.prefix_id

    @property
    def cell_key(self) -> tuple[str, str, int]:
        """Resume / dedup key: one row per (prefix, treatment, seed)."""
        return (self.prefix_id, self.treatment_id, self.branch_seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "repository_id": self.repository_id,
            "trajectory_id": self.trajectory_id,
            "prefix_id": self.prefix_id,
            "fork_point": self.fork_point.to_dict(),
            "progress": self.progress,
            "treatment_id": self.treatment_id,
            "content_level": self.content_level.value,
            "action": self.action.value,
            "intervention": self.intervention.to_dict(),
            "rung": self.rung.value,
            "branch_seed": self.branch_seed,
            "status": self.status,
            "fork_session_id": self.fork_session_id,
            "error": self.error,
            "actor_id": self.actor_id,
            "remaining_budget": self.remaining_budget,
            "binary_success": self.binary_success,
            "normalized_score": self.normalized_score,
            "final_state_hash": self.final_state_hash,
            "judge_detail": self.judge_detail,
            "follow_through": self.follow_through,
            "stale": self.stale,
            "duplicate": self.duplicate,
            "wasted_steps": self.wasted_steps,
            "critic_latency_ms": self.critic_latency_ms,
            "cost": self.cost,
            "sampling_weight": self.sampling_weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalUnit":
        return cls(
            case_id=str(data["case_id"]),
            repository_id=str(data["repository_id"]),
            trajectory_id=str(data["trajectory_id"]),
            prefix_id=str(data["prefix_id"]),
            fork_point=ForkPoint.from_dict(dict(data["fork_point"])),
            progress=float(data["progress"]),
            treatment_id=str(data["treatment_id"]),
            content_level=ContentLevel(data["content_level"]),
            action=ActionType(data["action"]),
            intervention=Intervention.from_dict(dict(data["intervention"])),
            rung=LadderRung(data["rung"]),
            branch_seed=int(data["branch_seed"]),
            status=str(data.get("status", "succeeded")),
            fork_session_id=data.get("fork_session_id"),
            error=data.get("error"),
            actor_id=str(data.get("actor_id") or ""),
            remaining_budget=dict(data.get("remaining_budget") or {}),
            binary_success=data.get("binary_success"),
            normalized_score=(
                float(data["normalized_score"])
                if data.get("normalized_score") is not None
                else None
            ),
            final_state_hash=data.get("final_state_hash"),
            judge_detail=dict(data.get("judge_detail") or {}),
            follow_through=data.get("follow_through"),
            stale=bool(data.get("stale", False)),
            duplicate=bool(data.get("duplicate", False)),
            wasted_steps=data.get("wasted_steps"),
            critic_latency_ms=data.get("critic_latency_ms"),
            cost=dict(data.get("cost") or {}),
            sampling_weight=float(data.get("sampling_weight", 1.0)),
            metadata=dict(data.get("metadata") or {}),
        )
