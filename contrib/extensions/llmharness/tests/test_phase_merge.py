"""Fail-stop tests for the deterministic phase merger.

The merger is the bridge between per-turn raw events and the auditor's
"basic block" view. If it silently merges the wrong things, the
auditor's high-level structural reasoning is corrupted; if it merges
too aggressively, drill-down loses turn-level provenance.
"""

from __future__ import annotations

from llmharness.audit.graph.phase import merge_to_phases
from llmharness.schema import Event, EventKind


def _ev(eid: int, kind: EventKind, summary: str = "", turns: list[int] | None = None) -> Event:
    return Event(
        id=eid,
        kind=kind,
        summary=summary or f"event {eid}",
        source_turns=turns or [eid - 1],
    )


def test_consecutive_act_run_collapses_into_one_phase() -> None:
    events = [
        _ev(1, EventKind.ACT, "tool_call list_tables"),
        _ev(2, EventKind.ACT, "list_tables result"),
        _ev(3, EventKind.ACT, "tool_call query_sql"),
        _ev(4, EventKind.ACT, "query_sql result"),
    ]
    phases = merge_to_phases(events)
    assert len(phases) == 1
    p = phases[0]
    assert p.kind == "act_run"
    assert p.member_event_ids == (1, 2, 3, 4)
    assert p.summary.count(" | ") == 3  # four pieces joined by separator
