"""ExtractionState fail-stop tests: validation order + retry budget.

A bug in any of these positions corrupts every extracted graph
downstream. Tests follow the §4.f algorithm exactly: existence ->
src!=dst -> turns subset -> cycle -> per-kind witness, with the retry
budget enforced inside ``add_edge``.
"""

from __future__ import annotations

from llmharness.audit.extractor.state import ExtractionState
from llmharness.schema import EdgeKind, EventKind


def _state_with_two_events() -> tuple[ExtractionState, int, int]:
    state = ExtractionState(
        turn_texts={
            10: "the abnormal_traces table contains four rows",
            11: "we should query abnormal_traces next",
        }
    )
    src = state.register_event(kind=EventKind.EVID, summary="src event", source_turns=[10])
    dst = state.register_event(kind=EventKind.HYP, summary="dst event", source_turns=[11])
    return state, src, dst


def test_register_event_assigns_monotonic_ids_starting_at_one() -> None:
    state = ExtractionState()
    a = state.register_event(kind=EventKind.TASK, summary="a", source_turns=[0])
    b = state.register_event(kind=EventKind.HYP, summary="b", source_turns=[1])
    c = state.register_event(kind=EventKind.ACT, summary="c", source_turns=[2])
    assert (a, b, c) == (1, 2, 3)


def test_add_edge_happy_path_appends_exactly_one_edge() -> None:
    state, src, dst = _state_with_two_events()
    err = state.add_edge(
        src_event_id=src,
        dst_event_id=dst,
        kind=EdgeKind.DATA,
        reason="evidence supports hypothesis",
        src_turns=[10],
        dst_turns=[11],
        cited_entities=["abnormal_traces"],
    )
    assert err is None
    assert len(state.edges) == 1
    assert state.edges[0].src == src
    assert state.edges[0].dst == dst


def test_add_edge_unknown_endpoint_is_rejected_and_counted() -> None:
    state, src, _dst = _state_with_two_events()
    err = state.add_edge(
        src_event_id=src,
        dst_event_id=999,
        kind=EdgeKind.DATA,
        reason="bad",
        src_turns=[10],
        dst_turns=[11],
        cited_entities=["abnormal_traces"],
    )
    assert err is not None
    assert "999" in err
    assert state.edges == []
    assert state.failure_counts[(src, 999, "data")] == 1


def test_add_edge_cycle_is_rejected() -> None:
    state, a, b = _state_with_two_events()
    # First edge a -> b: valid.
    assert (
        state.add_edge(
            src_event_id=a,
            dst_event_id=b,
            kind=EdgeKind.DATA,
            reason="ok",
            src_turns=[10],
            dst_turns=[11],
            cited_entities=["abnormal_traces"],
        )
        is None
    )
    # Second edge b -> a would close a cycle.
    err = state.add_edge(
        src_event_id=b,
        dst_event_id=a,
        kind=EdgeKind.DATA,
        reason="cycle",
        src_turns=[11],
        dst_turns=[10],
        cited_entities=["abnormal_traces"],
    )
    assert err is not None
    assert "cycle" in err.lower()
    assert state.failure_counts[(b, a, "data")] == 1


def test_add_edge_src_turns_must_be_subset_of_event_source_turns() -> None:
    state, src, dst = _state_with_two_events()
    err = state.add_edge(
        src_event_id=src,
        dst_event_id=dst,
        kind=EdgeKind.DATA,
        reason="bad",
        src_turns=[42],  # not in events[src].source_turns = [10]
        dst_turns=[11],
        cited_entities=["abnormal_traces"],
    )
    assert err is not None
    assert "subset" in err


def test_retry_budget_exhaustion_emits_giving_up_sentinel() -> None:
    """Three failures on the same (src, dst, kind) tuple -> terminal sentinel.

    The first two failures return ordinary error strings; the third
    returns a sentinel containing the literal ``giving up on this edge``
    phrase. Subsequent attempts on the same tuple short-circuit with
    the same terminal error WITHOUT re-validating, and ``dropped_edges``
    contains exactly one record.
    """

    state, src, dst = _state_with_two_events()

    def _try() -> str | None:
        return state.add_edge(
            src_event_id=src,
            dst_event_id=dst,
            kind=EdgeKind.DATA,
            reason="bad",
            src_turns=[10],
            dst_turns=[11],
            cited_entities=["nonexistent_token"],  # fails witness check
        )

    err1 = _try()
    err2 = _try()
    err3 = _try()
    err4 = _try()

    assert err1 is not None and "giving up on this edge" not in err1
    assert err2 is not None and "giving up on this edge" not in err2
    assert err3 is not None and "giving up on this edge" in err3
    assert err4 is not None and "giving up on this edge" in err4
    assert len(state.dropped_edges) == 1
    assert state.dropped_edges[0]["src"] == src
    assert state.dropped_edges[0]["dst"] == dst
    assert state.dropped_edges[0]["kind"] == "data"


def test_add_edge_data_requires_non_empty_cited_entities() -> None:
    state, src, dst = _state_with_two_events()
    err = state.add_edge(
        src_event_id=src,
        dst_event_id=dst,
        kind=EdgeKind.DATA,
        reason="missing entities",
        src_turns=[10],
        dst_turns=[11],
        cited_entities=[],
    )
    assert err is not None
    assert "cited_entities" in err


def test_add_edge_ref_requires_non_empty_cited_quote() -> None:
    state, src, dst = _state_with_two_events()
    err = state.add_edge(
        src_event_id=src,
        dst_event_id=dst,
        kind=EdgeKind.REF,
        reason="missing quote",
        src_turns=[10],
        dst_turns=[11],
        cited_quote="",
    )
    assert err is not None
    assert "cited_quote" in err
