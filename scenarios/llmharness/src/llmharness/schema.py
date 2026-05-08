"""Dataclasses for harness turns, events, verdicts, and reminders."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TurnRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


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
class Turn:
    """One transcript turn as ingested from a hook payload."""

    index: int
    role: TurnRole
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        return cls(
            index=int(data["index"]),
            role=TurnRole(data["role"]),
            content=data.get("content", "") or "",
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
        )


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
    """Drift detector output. drift=False means stay silent.

    Cognitive-audit V0 additions (additive, opt-in — see
    ``.claude/designs/llmharness-cognitive-audit.md`` §6.2):

    - ``cited_cards``: AFC IDs the audit chose to cite for the verdict.
      Empty for rule-based detector output.
    - ``downstream_reaction``: free-text note populated by the *next*
      audit firing describing whether the prior reminder was heeded.
      ``None`` until the next firing observes the agent's response.

    Both fields are kept at the END of the dataclass so existing
    positional or keyword constructions in
    ``detector.py`` / ``agentm_bridge.py`` / ``worker.py`` continue
    to produce semantically identical objects with default-empty
    values.
    """

    drift: bool
    type: DriftType | None = None
    confidence: float = 0.0
    reminder: str = ""
    matched_event_ids: list[int] = field(default_factory=list)
    cited_cards: list[str] = field(default_factory=list)
    downstream_reaction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift": self.drift,
            "type": self.type.value if self.type is not None else None,
            "confidence": self.confidence,
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
            confidence=float(data.get("confidence", 0.0)),
            reminder=str(data.get("reminder", "")),
            matched_event_ids=list(data.get("matched_event_ids") or []),
            cited_cards=list(data.get("cited_cards") or []),
            downstream_reaction=data.get("downstream_reaction"),
        )


@dataclass(frozen=True)
class Reminder:
    """A pending reminder waiting to be injected on the next user prompt."""

    session_id: str
    type: DriftType
    confidence: float
    text: str
    created_at_event_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "type": self.type.value,
            "confidence": self.confidence,
            "text": self.text,
            "created_at_event_id": self.created_at_event_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reminder:
        return cls(
            session_id=str(data["session_id"]),
            type=DriftType(data["type"]),
            confidence=float(data["confidence"]),
            text=str(data["text"]),
            created_at_event_id=int(data["created_at_event_id"]),
        )


def dumps_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def loads_jsonl(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]
