"""Fail-stop tests for the v3 auditor degradation rule (design §4.g).

The auditor prompt switches from full to degraded payload when the event
count crosses ``audit_summary_threshold`` (default 30). In degraded mode,
witness fields (``cited_entities``, ``cited_quote``) and per-side
source-turn tuples are stripped from the embedded edge JSON, and a
note tells the auditor to use ``get_event_detail([ids])`` to recover
them.

Why fail-stop: getting this wrong makes the auditor either (a) overflow
its context budget on large graphs, or (b) silently lose witness
information when degraded — both visible only through prompt inspection.
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
    # Witness payload reaches the rendered prompt in full mode.
    assert '"widget"' in prompt
    assert '"the widget"' in prompt
    # No degradation banner.
    assert "DEGRADED MODE" not in prompt
    assert "get_event_detail" in prompt  # tool referenced in static section


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
    # Witness payload stripped from the degraded edge JSON.
    assert '"widget"' not in prompt
    assert '"the widget"' not in prompt
    # Degradation banner + drill-down instruction present.
    assert "DEGRADED MODE" in prompt
    assert "get_event_detail" in prompt


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
    assert "DEGRADED MODE" not in prompt_5
    assert '"widget"' in prompt_5
    assert "DEGRADED MODE" in prompt_6
    assert '"widget"' not in prompt_6
