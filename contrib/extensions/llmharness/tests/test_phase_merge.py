"""Fail-stop tests for the deterministic phase merger.

The merger is the bridge between per-turn raw events and the auditor's
"basic block" view. If it silently merges the wrong things, the
auditor's high-level structural reasoning is corrupted; if it merges
too aggressively, drill-down loses turn-level provenance.
"""

from __future__ import annotations

from llmharness.audit.phase import merge_to_phases
from llmharness.schema import Event, EventKind


def _ev(eid: int, kind: EventKind, summary: str = "", turns: list[int] | None = None) -> Event:
    return Event(
        id=eid,
        kind=kind,
        summary=summary or f"event {eid}",
        source_turns=turns or [eid - 1],
    )


def test_empty_input_yields_empty_phases() -> None:
    assert merge_to_phases([]) == []


def test_singleton_break_kinds_each_become_their_own_phase() -> None:
    events = [
        _ev(1, EventKind.TASK, "the task"),
        _ev(2, EventKind.HYP, "a hypothesis"),
        _ev(3, EventKind.DEC, "a decision"),
        _ev(4, EventKind.CONCL, "a conclusion"),
    ]
    phases = merge_to_phases(events)
    assert [p.kind for p in phases] == ["task", "hyp", "dec", "concl"]
    assert [p.member_event_ids for p in phases] == [(1,), (2,), (3,), (4,)]
    # IDs are fresh-numbered.
    assert [p.id for p in phases] == [1, 2, 3, 4]


def test_consecutive_act_evid_run_collapses_into_one_phase() -> None:
    events = [
        _ev(1, EventKind.ACT, "tool_call list_tables"),
        _ev(2, EventKind.EVID, "list_tables result"),
        _ev(3, EventKind.ACT, "tool_call query_sql"),
        _ev(4, EventKind.EVID, "query_sql result"),
    ]
    phases = merge_to_phases(events)
    assert len(phases) == 1
    p = phases[0]
    assert p.kind == "act_evid_run"
    assert p.member_event_ids == (1, 2, 3, 4)
    assert p.summary.count(" | ") == 3  # four pieces joined by separator


def test_break_kind_flushes_run_and_starts_singleton() -> None:
    events = [
        _ev(1, EventKind.TASK, "task"),
        _ev(2, EventKind.ACT, "act A"),
        _ev(3, EventKind.EVID, "evid A"),
        _ev(4, EventKind.HYP, "hyp"),
        _ev(5, EventKind.ACT, "act B"),
        _ev(6, EventKind.EVID, "evid B"),
        _ev(7, EventKind.CONCL, "concl"),
    ]
    phases = merge_to_phases(events)
    assert [p.kind for p in phases] == [
        "task",
        "act_evid_run",
        "hyp",
        "act_evid_run",
        "concl",
    ]
    assert [p.member_event_ids for p in phases] == [
        (1,),
        (2, 3),
        (4,),
        (5, 6),
        (7,),
    ]
    assert [p.id for p in phases] == [1, 2, 3, 4, 5]


def test_single_act_or_evid_keeps_its_own_kind_not_run_label() -> None:
    """A run of length 1 must NOT be relabelled ``act_evid_run`` — that
    label means "two or more were merged" and downstream readers may
    branch on it."""
    events = [
        _ev(1, EventKind.HYP, "hyp"),
        _ev(2, EventKind.ACT, "lone act"),
        _ev(3, EventKind.HYP, "hyp 2"),
        _ev(4, EventKind.EVID, "lone evid"),
    ]
    phases = merge_to_phases(events)
    assert [p.kind for p in phases] == ["hyp", "act", "hyp", "evid"]


def test_phase_source_turns_are_union_of_member_turns_sorted_unique() -> None:
    events = [
        _ev(1, EventKind.ACT, "a", turns=[3, 5]),
        _ev(2, EventKind.EVID, "b", turns=[5, 4]),
        _ev(3, EventKind.ACT, "c", turns=[4]),
    ]
    phases = merge_to_phases(events)
    assert len(phases) == 1
    assert phases[0].source_turns == (3, 4, 5)


def test_run_summary_truncated_when_member_summaries_get_long() -> None:
    big = "x" * 800
    events = [
        _ev(1, EventKind.ACT, big),
        _ev(2, EventKind.EVID, big),
    ]
    phases = merge_to_phases(events)
    assert len(phases) == 1
    # 800+800+separator = 1603 chars > 1200 cap → truncated with "..."
    assert phases[0].summary.endswith("...")
    assert len(phases[0].summary) <= 1200
