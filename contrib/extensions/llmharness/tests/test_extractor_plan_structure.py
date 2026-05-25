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


