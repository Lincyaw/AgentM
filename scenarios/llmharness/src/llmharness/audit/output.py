"""Typed view of the cognitive-audit diagnostic agent's emit JSON.

Both consumers of the audit — the in-process V0 adapter
(:mod:`llmharness.adapters.agentm`, REQ-017) and the dormant P0
subprocess bridge (:mod:`llmharness.agentm_bridge`, REQ-007) — feed a
JSON payload through the LLM and parse a single
``{"events": [...], "verdict": {...}}`` reply back. Sharing the parser
here keeps the schema a one-file edit instead of a search-and-replace
across two consumers, and gives both a real type-checked surface
(no ``dict.get``-chain soup at the JSON boundary).

The schema this parser accepts is documented in
:data:`llmharness.audit.AUDIT_SYSTEM_PROMPT` step 10 — that prompt and
this dataclass MUST move together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema import DriftType, Event, EventKind, Verdict

# ---------------------------------------------------------------------------
# Coercion helpers — JSON values arrive as Any; these clamp them down to the
# concrete types the dataclasses below expect, dropping anything malformed
# silently. The audit is best-effort; raising on a bad LLM emit would block
# every downstream firing.


def _coerce_int_list(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and item.lstrip("-").isdigit():
            out.append(int(item))
    return out


def _coerce_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _coerce_float(raw: Any) -> float:
    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    return 0.0


# ---------------------------------------------------------------------------
# Dataclasses


@dataclass(frozen=True)
class RawAuditEvent:
    """One event entry from the manifest's ``events`` array."""

    kind: EventKind
    summary: str
    source_turns: list[int] = field(default_factory=list)
    refs: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RawAuditEvent | None:
        kind_str = raw.get("kind")
        summary = raw.get("summary")
        if not isinstance(kind_str, str) or not isinstance(summary, str):
            return None
        try:
            kind = EventKind(kind_str)
        except ValueError:
            return None
        return cls(
            kind=kind,
            summary=summary,
            source_turns=_coerce_int_list(raw.get("source_turns")),
            refs=_coerce_int_list(raw.get("refs")),
        )


@dataclass(frozen=True)
class RawAuditOutput:
    """The full ``{events, verdict}`` payload the manifest mandates."""

    drift: bool
    type: DriftType | None = None
    confidence: float = 0.0
    reminder: str = ""
    matched_event_ids: list[int] = field(default_factory=list)
    cited_cards: list[str] = field(default_factory=list)
    downstream_reaction: str | None = None
    events: list[RawAuditEvent] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RawAuditOutput | None:
        verdict_raw = raw.get("verdict")
        if not isinstance(verdict_raw, dict) or "drift" not in verdict_raw:
            return None

        drift = bool(verdict_raw.get("drift", False))
        type_str = verdict_raw.get("type")
        drift_type: DriftType | None = None
        if drift and isinstance(type_str, str) and type_str:
            try:
                drift_type = DriftType(type_str)
            except ValueError:
                drift_type = None
        reminder_raw = verdict_raw.get("reminder")
        reminder = reminder_raw if isinstance(reminder_raw, str) else ""
        downstream_raw = verdict_raw.get("downstream_reaction")
        downstream = downstream_raw if isinstance(downstream_raw, str) else None

        events_raw = raw.get("events")
        events: list[RawAuditEvent] = []
        if isinstance(events_raw, list):
            for item in events_raw:
                if not isinstance(item, dict):
                    continue
                ev = RawAuditEvent.from_dict(item)
                if ev is not None:
                    events.append(ev)

        return cls(
            drift=drift,
            type=drift_type,
            confidence=_coerce_float(verdict_raw.get("confidence")),
            reminder=reminder,
            matched_event_ids=_coerce_int_list(verdict_raw.get("matched_event_ids")),
            cited_cards=_coerce_str_list(verdict_raw.get("cited_cards")),
            downstream_reaction=downstream,
            events=events,
        )

    def to_verdict(self) -> Verdict:
        return Verdict(
            drift=self.drift,
            type=self.type,
            confidence=self.confidence,
            reminder=self.reminder,
            matched_event_ids=list(self.matched_event_ids),
            cited_cards=list(self.cited_cards),
            downstream_reaction=self.downstream_reaction,
        )

    def to_events(self, *, next_id: int) -> list[Event]:
        """Stamp monotonic ids onto stage-A events starting at ``next_id``.

        The LLM never emits ``id`` (manifest forbids it); the consumer owns
        id sequencing. Callers seed ``next_id`` from their own state:

        - V0 audit adapter: ``max(prior_events.id, default=-1) + 1``
        - P0 subprocess bridge: the ``next_event_id`` cursor passed in
        """

        out: list[Event] = []
        for raw in self.events:
            out.append(
                Event(
                    id=next_id,
                    kind=raw.kind,
                    summary=raw.summary,
                    refs=list(raw.refs),
                    source_turns=list(raw.source_turns),
                )
            )
            next_id += 1
        return out


__all__ = ["RawAuditEvent", "RawAuditOutput"]
