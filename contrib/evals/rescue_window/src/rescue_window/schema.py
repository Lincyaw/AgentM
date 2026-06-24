"""Data model for rescue-window branching experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast


class ActionType(StrEnum):
    """Typed intervention actions from the Rescue Window memo."""

    CONTINUE = "CONTINUE"
    VERIFY = "VERIFY"
    CHECKPOINT = "CHECKPOINT"
    REVERT_TO_BEST = "REVERT_TO_BEST"
    REPLAN_SCOPE = "REPLAN_SCOPE"
    FINAL_AUDIT = "FINAL_AUDIT"


_TARGET_KEYS = {
    "requirement_id",
    "assumption_id",
    "file",
    "API",
    "test",
    "plan_node",
}
_EVIDENCE_KEYS = {"trajectory_event_ids"}
_VALID_UNTIL_KEYS = {"state_hash", "step_ttl"}
_STRENGTH_VALUES = {"advisory", "blocking"}


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class InterventionDecision:
    """One policy output for a prefix."""

    policy_id: str
    intervention: Intervention
    should_intervene: bool = True
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "should_intervene": self.should_intervene,
            "reason": self.reason,
            "metadata": self.metadata,
            "intervention": self.intervention.to_dict(),
        }


@dataclass(frozen=True)
class BranchSpec:
    """A single same-prefix branch in an experiment spec."""

    branch_id: str
    source_session_id: str
    fork_point: ForkPoint
    intervention: Intervention
    policy_id: str = "static"
    trajectory_id: str | None = None
    baseline_session_id: str | None = None
    case_id: str | None = None
    max_turns: int | None = None
    max_tool_calls: int | None = None
    cwd: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        defaults: dict[str, Any],
    ) -> "BranchSpec":
        merged = {**defaults, **data}
        return cls(
            branch_id=_str_value(merged.get("branch_id"), "branch.branch_id"),
            source_session_id=_str_value(
                merged.get("source_session_id"), "branch.source_session_id"
            ),
            fork_point=ForkPoint.from_dict(
                _dict_value(merged.get("fork_point") or {}, "branch.fork_point")
            ),
            intervention=Intervention.from_dict(
                _dict_value(merged.get("intervention") or {}, "branch.intervention")
            ),
            policy_id=_optional_str(merged.get("policy_id"), "branch.policy_id")
            or "static",
            trajectory_id=_optional_str(
                merged.get("trajectory_id"), "branch.trajectory_id"
            ),
            baseline_session_id=_optional_str(
                merged.get("baseline_session_id"), "branch.baseline_session_id"
            ),
            case_id=_optional_str(merged.get("case_id"), "branch.case_id"),
            max_turns=_optional_int(merged.get("max_turns"), "branch.max_turns"),
            max_tool_calls=_optional_int(
                merged.get("max_tool_calls"), "branch.max_tool_calls"
            ),
            cwd=_optional_str(merged.get("cwd"), "branch.cwd"),
            metadata=_dict_value(merged.get("metadata") or {}, "branch.metadata"),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: str
    experiment_id: str
    branches: list[BranchSpec]
    description: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        defaults = _dict_value(data.get("defaults") or {}, "defaults")
        raw_branches = data.get("branches")
        if not isinstance(raw_branches, list) or not raw_branches:
            raise ValueError("experiment spec must define a non-empty branches list")
        branches = [
            BranchSpec.from_dict(
                _dict_value(item, f"branches[{idx}]"),
                defaults=defaults,
            )
            for idx, item in enumerate(raw_branches)
        ]
        branch_ids = [branch.branch_id for branch in branches]
        duplicates = sorted(
            {branch_id for branch_id in branch_ids if branch_ids.count(branch_id) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate branch_id values: {', '.join(duplicates)}")
        return cls(
            schema_version=_str_value(data.get("schema_version"), "schema_version"),
            experiment_id=_str_value(data.get("experiment_id"), "experiment_id"),
            branches=branches,
            description=_optional_str(data.get("description"), "description") or "",
            defaults=defaults,
        )


def load_experiment_spec(path: Path) -> ExperimentSpec:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported spec extension {path.suffix!r}")
    return ExperimentSpec.from_dict(_dict_value(data, "experiment"))


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
