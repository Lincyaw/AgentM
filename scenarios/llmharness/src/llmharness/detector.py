"""Rule-based drift detector (P0).

Four patterns, in priority order: stuck_loop > premature_conclusion >
evidence_ignored > task_drift. The first match wins; we never stack multiple
verdicts.

The function returns a Verdict; the caller decides whether confidence clears
the threshold. Stay-silent is the default.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schema import DriftType, Event, EventKind, Verdict

_STUCK_LOOP_WINDOW = 4
_TASK_DRIFT_OVERLAP = 0.15
_PREMATURE_MIN_EVIDENCE = 1
_NEGATIVE_EVIDENCE_MARKERS = (
    "no anomaly",
    "not found",
    "no error",
    "0 results",
    "no match",
    "无异常",
    "未发现",
    "没有",
    "no result",
    "empty",
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", text.lower()) if len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _last_n(events: Iterable[Event], kinds: set[EventKind], n: int) -> list[Event]:
    return [e for e in events if e.kind in kinds][-n:]


def _stuck_loop(events: list[Event]) -> Verdict | None:
    actions = _last_n(events, {EventKind.ACTION}, _STUCK_LOOP_WINDOW)
    if len(actions) < _STUCK_LOOP_WINDOW:
        return None
    # No evidence/reflection event has appeared between the first and last
    # action in the window -> the agent is acting without absorbing feedback.
    first_id = actions[0].id
    last_id = actions[-1].id
    interleaved = [
        e
        for e in events
        if first_id < e.id < last_id and e.kind in {EventKind.EVIDENCE, EventKind.REFLECTION}
    ]
    if interleaved:
        return None
    # Similarity check: at least the last 3 actions share a common token core.
    sims = [
        _jaccard(_tokens(actions[i].summary), _tokens(actions[i + 1].summary))
        for i in range(len(actions) - 1)
    ]
    if not sims or min(sims) < 0.4:
        return None
    return Verdict(
        drift=True,
        type=DriftType.STUCK_LOOP,
        confidence=0.8,
        reminder=(
            "[harness] possible stuck loop — last "
            f"{_STUCK_LOOP_WINDOW} actions look similar with no new evidence. "
            "Step back: what would change your current approach?"
        ),
        matched_event_ids=[a.id for a in actions],
    )


def _premature_conclusion(events: list[Event]) -> Verdict | None:
    if not events or events[-1].kind is not EventKind.CONCLUSION:
        return None
    last = events[-1]
    evidence_count = sum(1 for e in events if e.kind is EventKind.EVIDENCE and e.id < last.id)
    if evidence_count >= _PREMATURE_MIN_EVIDENCE:
        return None
    return Verdict(
        drift=True,
        type=DriftType.PREMATURE_CONCLUSION,
        confidence=0.7,
        reminder=(
            "[harness] possible premature conclusion — a conclusion was drawn "
            "before any evidence event was recorded. Verify before committing."
        ),
        matched_event_ids=[last.id],
    )


def _evidence_ignored(events: list[Event]) -> Verdict | None:
    if not events or events[-1].kind is not EventKind.ACTION:
        return None
    # Look back for the most recent evidence event with a negative marker.
    for prev in reversed(events[:-1]):
        if prev.kind is EventKind.REFLECTION:
            return None  # reflection between negative evidence and next action -> ok
        if prev.kind is EventKind.EVIDENCE:
            lowered = prev.summary.lower()
            if not any(m in lowered for m in _NEGATIVE_EVIDENCE_MARKERS):
                return None
            return Verdict(
                drift=True,
                type=DriftType.EVIDENCE_IGNORED,
                confidence=0.65,
                reminder=(
                    "[harness] possible evidence ignored — the latest evidence "
                    "looked negative but the agent moved to a new action without "
                    "reflecting on it."
                ),
                matched_event_ids=[prev.id, events[-1].id],
            )
    return None


def _task_drift(events: list[Event]) -> Verdict | None:
    task = next((e for e in events if e.kind is EventKind.TASK), None)
    if task is None:
        return None
    recent = _last_n(events, {EventKind.ACTION, EventKind.HYPOTHESIS}, 3)
    if len(recent) < 3:
        return None
    task_tokens = _tokens(task.summary)
    if not task_tokens:
        return None
    overlaps = [_jaccard(task_tokens, _tokens(e.summary)) for e in recent]
    if max(overlaps) >= _TASK_DRIFT_OVERLAP:
        return None
    return Verdict(
        drift=True,
        type=DriftType.TASK_DRIFT,
        confidence=0.6,
        reminder=(
            "[harness] possible task drift — the last few actions/hypotheses "
            "share little vocabulary with the original task. Re-read turn 0."
        ),
        matched_event_ids=[task.id, *(e.id for e in recent)],
    )


def detect_drift(events: list[Event]) -> Verdict:
    """Return the highest-priority verdict, or a silent verdict if nothing matches."""
    for check in (_stuck_loop, _premature_conclusion, _evidence_ignored, _task_drift):
        verdict = check(events)
        if verdict is not None:
            return verdict
    return Verdict(drift=False)
