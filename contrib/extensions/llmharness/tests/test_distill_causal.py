"""Fail-stop: causal masking of the audit snapshot.

Load-bearing for the distill pipeline (design D2 in
``distill/README.md``). If this masking leaks future information, the
oracle labels become non-causal — the student model, which only sees
past turns at inference, cannot reproduce them, and the entire SFT
dataset is biased toward a function the model cannot learn.
"""

from __future__ import annotations

from llmharness.distill.causal import causal_mask
from llmharness.schema import Edge, EdgeKind, Event, EventKind, Finding


def _ev(eid: int, turns: list[int]) -> Event:
    return Event(id=eid, kind=EventKind.ACT, summary=f"e{eid}", source_turns=turns)


def _edge(src: int, dst: int, src_turns: tuple[int, ...], dst_turns: tuple[int, ...]) -> Edge:
    return Edge(
        src=src,
        dst=dst,
        kind=EdgeKind.DATA,
        reason="test",
        src_turns=src_turns,
        dst_turns=dst_turns,
        cited_entities=("x",),
        cited_quote="",
    )


def test_event_with_future_source_turns_is_dropped() -> None:
    events = [_ev(1, [3, 5]), _ev(2, [3, 7])]
    snap = causal_mask(turn_index=6, events=events, edges=[], findings=[], trajectory=[])
    kept_ids = {e.id for e in snap.events}
    assert kept_ids == {1}, "event referencing turn 7 must be masked at t=6"




def test_edge_dropped_when_endpoint_event_dropped() -> None:
    events = [_ev(1, [2]), _ev(2, [9])]
    edges = [_edge(1, 2, (2,), (9,))]
    snap = causal_mask(turn_index=5, events=events, edges=edges, findings=[], trajectory=[])
    assert {e.id for e in snap.events} == {1}
    assert snap.edges == ()


def test_edge_dropped_when_turn_lists_reach_future() -> None:
    events = [_ev(1, [2]), _ev(2, [3])]
    # both endpoint events are present, but the edge's dst_turns names a
    # future turn — the edge itself is causally invalid.
    edges = [_edge(1, 2, (2,), (8,))]
    snap = causal_mask(turn_index=5, events=events, edges=edges, findings=[], trajectory=[])
    assert {e.id for e in snap.events} == {1, 2}
    assert snap.edges == ()


def test_finding_referencing_dropped_event_is_dropped() -> None:
    events = [_ev(1, [2]), _ev(99, [10])]  # 99 dropped at t=5
    findings = [
        Finding(category="repeated", description="d", related_event_ids=(1, 99)),
        Finding(category="open", description="d", related_event_ids=(1,)),
    ]
    snap = causal_mask(
        turn_index=5, events=events, edges=[], findings=findings, trajectory=[]
    )
    cats = [f.category for f in snap.findings]
    assert cats == ["open"], "finding citing a future event id must be dropped"


