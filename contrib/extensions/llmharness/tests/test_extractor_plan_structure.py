"""Soft-warning degree check for the v4 extractor's emitted graph.

V4 (2026-05-24): the previous hard "no passthrough" reject is gone.
The emitted graph is committed unconditionally on a witness-valid
input; the degree heuristic returns an advisory string (or ``None``)
that the caller surfaces alongside the success result. These tests
pin the new contract: finalize never blocks on a chain shape, the
advisory text names the offending event ids, and a true branch /
merge produces no advisory.

See ``_compute_degree_warning`` and
``ExtractionState.compute_degree_warning`` in
``llmharness.audit.extractor.state``.
"""

from __future__ import annotations

from llmharness.audit.extractor.state import (
    ExtractionState,
    _compute_degree_warning,
)
from llmharness.schema import Edge, EdgeKind, Event, EventKind


def _ev(eid: int, kind: str = "act", turns: list[int] | None = None) -> Event:
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


def test_empty_graph_emits_no_warning() -> None:
    assert _compute_degree_warning([], []) is None


def test_single_event_emits_no_warning() -> None:
    """A single-node firing has in=0, out=0 — no chain link."""
    assert _compute_degree_warning([_ev(1)], []) is None


def test_chain_of_two_emits_no_warning() -> None:
    """task -> act: in=(0,1), out=(1,0). Both endpoints, no chain link."""
    warn = _compute_degree_warning(
        [_ev(1, "task"), _ev(2, "act")],
        [_edge(1, 2)],
    )
    assert warn is None


def test_chain_of_three_emits_warning() -> None:
    """1 -> 2 -> 3: event 2 has in=1, out=1 — soft warning expected."""
    warn = _compute_degree_warning(
        [_ev(1), _ev(2), _ev(3)],
        [_edge(1, 2), _edge(2, 3)],
    )
    assert warn is not None
    assert "chain-link" in warn or "Chain-link" in warn
    assert "event[2]" in warn
    # endpoints 1 and 3 are NOT named as offenders
    assert "event[1] kind=" not in warn
    assert "event[3] kind=" not in warn


def test_branch_point_emits_no_warning() -> None:
    """1 -> 2, 1 -> 3: event 1 has out=2; 2 and 3 are leaves. No chain link."""
    warn = _compute_degree_warning(
        [_ev(1), _ev(2), _ev(3)],
        [_edge(1, 2), _edge(1, 3)],
    )
    assert warn is None


def test_merge_point_emits_no_warning() -> None:
    """1 -> 3, 2 -> 3: event 3 has in=2; no chain links."""
    warn = _compute_degree_warning(
        [_ev(1), _ev(2), _ev(3)],
        [_edge(1, 3), _edge(2, 3)],
    )
    assert warn is None


def test_real_fork_and_merge_emits_no_warning() -> None:
    """A graph with a real fork AND a merge has no chain-link nodes.

    Shape:  1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
    Degrees: 1:(0,2), 2:(1,1)? actually 2 has in=1, out=1 — that IS a
    chain link, so let's pick a wider shape that genuinely has none.

    Use:  1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4, 1 -> 4
    Degrees: 1:(0,3), 2:(1,1), 3:(1,1), 4:(3,0)
    Still chain links at 2,3. The shape that has NO chain links is
    the diamond with extra incoming edges:
    1 -> 2, 1 -> 3, 2 -> 3, 2 -> 4, 3 -> 4
    Degrees: 1:(0,2), 2:(1,2), 3:(2,1), 4:(2,0)  ← all true branch / merge.
    """
    warn = _compute_degree_warning(
        [_ev(1), _ev(2), _ev(3), _ev(4)],
        [_edge(1, 2), _edge(1, 3), _edge(2, 3), _edge(2, 4), _edge(3, 4)],
    )
    assert warn is None, warn


def test_all_chain_links_reported_at_once() -> None:
    """1 -> 2 -> 3 -> 4 -> 5: events 2, 3, 4 are all chain links."""
    warn = _compute_degree_warning(
        [_ev(eid) for eid in range(1, 6)],
        [_edge(1, 2), _edge(2, 3), _edge(3, 4), _edge(4, 5)],
    )
    assert warn is not None
    assert "event[2]" in warn
    assert "event[3]" in warn
    assert "event[4]" in warn


def test_finalize_succeeds_with_chain_warning() -> None:
    """End-to-end: a chain submitted via commit_batch + finalize COMMITS.

    V4 contract: chain shape is acceptable. finalize returns None,
    state.committed flips to True, and the soft warning is surfaced
    via ExtractionState.compute_degree_warning().
    """
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
                "kind": "concl",
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
    assert final_err is None
    assert state.committed is True
    # Soft warning surfaces the chain-link middle event.
    warning = state.compute_degree_warning()
    assert warning is not None
    assert "event[2]" in warning


def test_finalize_succeeds_no_warning_on_branching_graph() -> None:
    """Four-event graph with no chain link: event 1 forks (out=2), event 2
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
                "kind": "act",
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
    # No chain link anywhere → no warning.
    assert state.compute_degree_warning() is None
