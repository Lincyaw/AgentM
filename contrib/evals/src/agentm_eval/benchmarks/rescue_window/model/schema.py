"""Typed intervention DSL (doc §7.1).

The action vocabulary, fork-point selector, and the ``Intervention`` record
(action / target / evidence / strength / valid_until) that every treatment and
critic decision is expressed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast


class ActionType(StrEnum):
    """Typed intervention actions — each isolates one experimental variable."""

    CONTINUE = "CONTINUE"
    GENERIC = "GENERIC"
    VERIFY = "VERIFY"
    ADVISE = "ADVISE"
    REPLAN = "REPLAN"
    FINAL_AUDIT = "FINAL_AUDIT"


_TARGET_KEYS = {
    "requirement_id",
    "assumption_id",
    "file",
    "API",
    "test",
    "plan_node",
    "service",
    "fault_kind",
}
_EVIDENCE_KEYS = {"trajectory_event_ids"}
_VALID_UNTIL_KEYS = {"state_hash", "step_ttl"}
_STRENGTH_VALUES = {"advisory", "blocking"}


@dataclass(frozen=True, slots=True)
class ForkPoint:
    """A source-session fork selector."""

    message_id: str | None = None
    turn_id: int | None = None
    turn_index: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForkPoint":
        point = cls(
            message_id=_optional_str(data.get("message_id"), "fork_point.message_id"),
            turn_id=_optional_int(data.get("turn_id"), "fork_point.turn_id"),
            turn_index=_optional_int(data.get("turn_index"), "fork_point.turn_index"),
        )
        point.validate()
        return point

    def validate(self) -> None:
        count = sum(
            value is not None for value in (self.message_id, self.turn_id, self.turn_index)
        )
        if count != 1:
            raise ValueError(
                "fork_point must set exactly one of message_id, turn_id, turn_index"
            )

    def to_dict(self) -> dict[str, Any]:
        if self.message_id is not None:
            return {"message_id": self.message_id.removeprefix("message:")}
        if self.turn_id is not None:
            return {"turn_id": self.turn_id}
        return {"turn_index": self.turn_index}


@dataclass(frozen=True, slots=True)
class Intervention:
    """A typed intervention plus its actor-visible channel message."""

    action: ActionType
    condition_id: str
    content_level: str
    message: str = ""
    target: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    strength: str = "advisory"
    valid_until: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intervention":
        action = ActionType(_str_value(data.get("action"), "intervention.action"))
        target = _dict_value(data.get("target") or {}, "intervention.target")
        evidence = _dict_value(data.get("evidence") or {}, "intervention.evidence")
        valid_until = _dict_value(
            data.get("valid_until") or {}, "intervention.valid_until"
        )
        strength = _optional_str(data.get("strength"), "intervention.strength") or "advisory"
        _validate_keys(target, _TARGET_KEYS, "intervention.target")
        _validate_keys(evidence, _EVIDENCE_KEYS, "intervention.evidence")
        _validate_keys(valid_until, _VALID_UNTIL_KEYS, "intervention.valid_until")
        if strength not in _STRENGTH_VALUES:
            raise ValueError(
                "intervention.strength must be one of "
                + ", ".join(sorted(_STRENGTH_VALUES))
            )
        return cls(
            action=action,
            condition_id=_str_value(data.get("condition_id"), "intervention.condition_id"),
            content_level=_str_value(
                data.get("content_level") or data.get("condition_id"),
                "intervention.content_level",
            ),
            message=_optional_str(data.get("message"), "intervention.message") or "",
            target=target,
            evidence=evidence,
            strength=strength,
            valid_until=valid_until,
            metadata=_dict_value(data.get("metadata") or {}, "intervention.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "condition_id": self.condition_id,
            "content_level": self.content_level,
            "message": self.message,
            "target": self.target,
            "evidence": self.evidence,
            "strength": self.strength,
            "valid_until": self.valid_until,
            "metadata": self.metadata,
        }


def _validate_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"{name} has unsupported keys {unknown}; allowed keys: {sorted(allowed)}"
        )


def _dict_value(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _str_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
