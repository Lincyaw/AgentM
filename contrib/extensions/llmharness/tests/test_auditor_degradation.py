"""Fail-stop tests for the auditor degradation rule.

The auditor prompt switches from full to degraded payload when the event
count crosses ``audit_summary_threshold`` (default 30). In degraded mode,
witness fields (``cited_entities``, ``cited_quote``) and per-side
source-turn tuples are stripped from the embedded edge JSON, and the
GRAPH section header surfaces a ``degraded — threshold=N, witness fields
stripped`` marker.

Why fail-stop: getting this wrong makes the auditor either (a) overflow
its context budget on large graphs, or (b) silently lose witness
information when degraded — both visible only through prompt inspection.

Drill-down language (``get_event_detail`` etc.) is no longer injected
dynamically; it lives in the chosen framing file. See
``test_auditor_prompt_variants.py`` for prompt-file coverage.
"""

from __future__ import annotations

from llmharness.audit.auditor.prompt import build_auditor_system_prompt
from llmharness.schema import Edge, EdgeKind, Event, EventKind


def _mk_events(n: int) -> tuple[Event, ...]:
    return tuple(
        Event(
            id=i,
            kind=EventKind.ACT,
            summary=f"act-{i}",
            source_turns=[i],
        )
        for i in range(1, n + 1)
    )


def _mk_edge(src: int, dst: int) -> Edge:
    return Edge(
        src=src,
        dst=dst,
        kind=EdgeKind.DATA,
        reason="test reason",
        src_turns=(src,),
        dst_turns=(dst,),
        cited_entities=("widget",),
        cited_quote="the widget",
    )


def test_threshold_off_30_events_full_payload_with_witness_fields() -> None:
    events = _mk_events(30)
    edges = (_mk_edge(1, 2), _mk_edge(2, 3))
    prompt = build_auditor_system_prompt(
        events=events,
        edges=edges,
        findings=[],
        check_errors={},
        continuation_notes=[],
        summary_threshold=30,
    )
    assert '"widget"' in prompt
    assert '"the widget"' in prompt
    assert "degraded" not in prompt.lower()


def test_threshold_on_31_events_degraded_payload_strips_witness_fields() -> None:
    events = _mk_events(31)
    edges = (_mk_edge(1, 2),)
    prompt = build_auditor_system_prompt(
        events=events,
        edges=edges,
        findings=[],
        check_errors={},
        continuation_notes=[],
        summary_threshold=30,
    )
    assert '"widget"' not in prompt
    assert '"the widget"' not in prompt
    assert "degraded" in prompt.lower()
    assert "witness fields stripped" in prompt


def test_configurable_threshold_cuts_in_at_n_plus_one() -> None:
    events_5 = _mk_events(5)
    events_6 = _mk_events(6)
    edge = (_mk_edge(1, 2),)

    prompt_5 = build_auditor_system_prompt(
        events=events_5,
        edges=edge,
        findings=[],
        check_errors={},
        continuation_notes=[],
        summary_threshold=5,
    )
    prompt_6 = build_auditor_system_prompt(
        events=events_6,
        edges=edge,
        findings=[],
        check_errors={},
        continuation_notes=[],
        summary_threshold=5,
    )
    assert "degraded" not in prompt_5.lower()
    assert '"widget"' in prompt_5
    assert "degraded" in prompt_6.lower()
    assert '"widget"' not in prompt_6
