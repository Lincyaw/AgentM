"""Degree validation for the v18 extractor's emitted event graph.

Replaces the v17 plan-structure check (no adjacent linear blocks),
which was a proxy: the LLM mechanically inserted dec/hyp branches
between every tool-call pair to satisfy it, defeating coalescence.

The v18 check operates on the EMITTED graph instead of the plan:
every internal event must be a true branch point (in-degree > 1 OR
out-degree > 1, with in=0 / out=0 reserved for endpoints). A
passthrough event — (in=1, out=1) — is a chain link with no branching
role and should have been merged with its neighbour as basic-block
coalescence. These tests pin the wire-level rejection so future
prompt iterations don't silently weaken the invariant.

See ``_validate_event_degrees`` in
``llmharness.audit.extractor.state``.
"""

from __future__ import annotations

from llmharness.audit.extractor.state import (
    ExtractionState,
    _validate_event_degrees,
)
from llmharness.schema import Edge, EdgeKind, Event, EventKind


def _ev(eid: int, kind: str = "evid", turns: list[int] | None = None) -> Event:
    return Event(
        id=eid,
        kind=EventKind(kind),
        summary=f"e{eid}",
        source_turns=turns or [eid],
    )


def _edge(src: int, dst: int) -> Edge:
    return Edge(
        src=src,
        dst=dst,
        kind=EdgeKind.DATA,
        reason="r",
        src_turns=(src,),
        dst_turns=(dst,),
        cited_entities=("x",),
        cited_quote="",
    )


def test_empty_graph_is_ok() -> None:
    assert _validate_event_degrees([], []) is None


def test_single_event_is_ok() -> None:
    """A single-node firing has in=0, out=0 — that's a legal endpoint."""
    assert _validate_event_degrees([_ev(1)], []) is None


def test_chain_of_two_is_ok() -> None:
    """task -> evid: in=(0,1), out=(1,0). Both legal endpoints."""
    err = _validate_event_degrees(
        [_ev(1, "task"), _ev(2, "evid")],
        [_edge(1, 2)],
    )
    assert err is None


def test_chain_of_three_rejects_middle_passthrough() -> None:
    """1 -> 2 -> 3: event 2 has in=1, out=1 — that's a passthrough."""
    err = _validate_event_degrees(
        [_ev(1), _ev(2), _ev(3)],
        [_edge(1, 2), _edge(2, 3)],
    )
    assert err is not None
    assert "passthrough" in err
    assert "event[2]" in err
    # endpoints 1 and 3 are NOT reported
    assert "event[1] kind=" not in err
    assert "event[3] kind=" not in err


def test_branch_point_is_ok() -> None:
    """1 -> 2, 1 -> 3: event 1 has out=2 (branch point), 2 and 3 are leaves."""
    err = _validate_event_degrees(
        [_ev(1), _ev(2), _ev(3)],
        [_edge(1, 2), _edge(1, 3)],
    )
    assert err is None


def test_merge_point_is_ok() -> None:
    """1 -> 3, 2 -> 3: event 3 has in=2 (merge point)."""
    err = _validate_event_degrees(
        [_ev(1), _ev(2), _ev(3)],
        [_edge(1, 3), _edge(2, 3)],
    )
    assert err is None


def test_all_passthroughs_reported_at_once() -> None:
    """1 -> 2 -> 3 -> 4 -> 5: events 2, 3, 4 are all passthrough."""
    err = _validate_event_degrees(
        [_ev(eid) for eid in range(1, 6)],
        [_edge(1, 2), _edge(2, 3), _edge(3, 4), _edge(4, 5)],
    )
    assert err is not None
    assert "event[2]" in err
    assert "event[3]" in err
    assert "event[4]" in err


def test_finalize_rejects_passthrough_chain() -> None:
    """End-to-end: a chain submitted via commit_batch + finalize fails."""
    state = ExtractionState(
        turn_texts={
            1: "alpha bravo",
            2: "alpha charlie",
            3: "charlie delta",
        }
    )
    err = state.commit_batch(
        [
            {"id": 1, "kind": "task", "summary": "start", "source_turns": [1]},
            {
                "id": 2,
                "kind": "act",
                "summary": "middle",
                "source_turns": [2],
                "refs": [
                    {
                        "to": 1,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["alpha"],
                    }
                ],
            },
            {
                "id": 3,
                "kind": "evid",
                "summary": "end",
                "source_turns": [3],
                "refs": [
                    {
                        "to": 2,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["charlie"],
                    }
                ],
            },
        ]
    )
    assert err is None  # batch shape is fine
    final_err = state.finalize()
    assert final_err is not None
    assert "passthrough" in final_err
    assert "event[2]" in final_err
    assert state.committed is False  # finalize failure keeps state recoverable


def test_finalize_accepts_branching_graph() -> None:
    """Four-event graph with no passthrough: event 1 forks (out=2), event 2
    forks (out=2), event 3 merges (in=2), event 4 is a leaf (in=2, out=0)."""
    state = ExtractionState(
        turn_texts={
            1: "alpha bravo charlie",
            2: "alpha charlie",
            3: "charlie delta alpha",
            4: "charlie alpha",
        }
    )
    err = state.commit_batch(
        [
            {"id": 1, "kind": "task", "summary": "start", "source_turns": [1]},
            {
                "id": 2,
                "kind": "act",
                "summary": "fork",
                "source_turns": [2],
                "refs": [
                    {
                        "to": 1,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["alpha"],
                    }
                ],
            },
            {
                "id": 3,
                "kind": "evid",
                "summary": "merge of 1 and 2",
                "source_turns": [3],
                "refs": [
                    {
                        "to": 1,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["alpha"],
                    },
                    {
                        "to": 2,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["charlie"],
                    },
                ],
            },
            {
                "id": 4,
                "kind": "concl",
                "summary": "leaf citing 2 and 3",
                "source_turns": [4],
                "refs": [
                    {
                        "to": 2,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["alpha"],
                    },
                    {
                        "to": 3,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["charlie"],
                    },
                ],
            },
        ]
    )
    assert err is None
    final_err = state.finalize()
    assert final_err is None, final_err
    assert state.committed is True
    assert len(state.events) == 4


def test_finalize_failure_is_recoverable_via_extra_batch() -> None:
    """After a finalize rejection, the LLM can append a batch that adds
    an extra ref to the passthrough event, promoting it to a branch.
    """
    state = ExtractionState(
        turn_texts={
            1: "alpha bravo",
            2: "alpha charlie",
            3: "charlie delta",
            4: "charlie echo",
        }
    )
    batch1_err = state.commit_batch(
        [
            {"id": 1, "kind": "task", "summary": "start", "source_turns": [1]},
            {
                "id": 2,
                "kind": "act",
                "summary": "middle",
                "source_turns": [2],
                "refs": [
                    {
                        "to": 1,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["alpha"],
                    }
                ],
            },
            {
                "id": 3,
                "kind": "evid",
                "summary": "end",
                "source_turns": [3],
                "refs": [
                    {
                        "to": 2,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["charlie"],
                    }
                ],
            },
        ]
    )
    assert batch1_err is None
    first_finalize = state.finalize()
    assert first_finalize is not None  # event 2 is passthrough
    assert state.committed is False

    # Append a batch that gives event 2 a second out-edge → promotes
    # it from (in=1, out=1) to (in=1, out=2): a true branch point.
    batch2_err = state.commit_batch(
        [
            {
                "id": 4,
                "kind": "evid",
                "summary": "second ref to event 2",
                "source_turns": [4],
                "refs": [
                    {
                        "to": 2,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["charlie"],
                    }
                ],
            },
        ]
    )
    assert batch2_err is None
    second_finalize = state.finalize()
    assert second_finalize is None, second_finalize
    assert state.committed is True
    assert len(state.events) == 4
