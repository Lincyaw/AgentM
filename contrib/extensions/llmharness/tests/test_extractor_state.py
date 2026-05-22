"""ExtractionState fail-stop tests for v3.1 single-tool flow.

The extractor LLM submits the entire graph in ONE ``submit_events``
call; ``ExtractionState.commit`` is the lone validation gate. Bugs
here corrupt every extracted graph downstream, so the tests pin the
load-bearing positions:

- event-shape errors hard-reject the whole submission (state
  unchanged → LLM may retry with the error message)
- ids are GLOBAL — must be >= ``next_event_id``, strictly increasing
  in submission order, and disjoint from ``recent_graph`` ids
- refs may only point to EARLIER events (id < self.id) — this is the
  cycle prevention by construction
- per-kind witness fields are required (data → cited_entities,
  ref → cited_quote)
- witness substring failures DROP the failing ref into
  ``dropped_edges`` while keeping the events accepted (partial-success
  path, design §6)
- commit is one-shot: a second commit on the same state returns an
  error
"""

from __future__ import annotations

from typing import Any

from llmharness.audit.extractor.state import ExtractionState


def _state() -> ExtractionState:
    return ExtractionState(
        turn_texts={
            10: "the abnormal_traces table contains four rows",
            11: "we should query abnormal_traces next",
        }
    )


def _evid(eid: int, *, turns: list[int], summary: str = "evt") -> dict[str, Any]:
    return {"id": eid, "kind": "evid", "summary": summary, "source_turns": turns}


# --- happy path -------------------------------------------------------------


def test_commit_happy_path_accepts_events_and_witnessed_ref() -> None:
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10], summary="src"),
            {
                **_evid(2, turns=[11], summary="dst"),
                "refs": [
                    {
                        "to": 1,
                        "kind": "data",
                        "reason": "evidence supports next step",
                        "cited_entities": ["abnormal_traces"],
                    }
                ],
            },
        ]
    )
    assert err is None
    assert len(state.events) == 2
    assert len(state.edges) == 1
    assert state.edges[0].src == 1 and state.edges[0].dst == 2
    assert state.dropped_edges == ()
    assert state.committed is True


def test_atomic_edit_tools_upsert_and_delete_nodes_and_edges() -> None:
    state = _state()
    assert state.upsert_node(_evid(1, turns=[10], summary="src"))["pending_nodes"] == 1
    assert state.upsert_node(_evid(2, turns=[11], summary="dst"))["pending_nodes"] == 2
    updated = state.upsert_node(
        _evid(2, turns=[11], summary="updated dst"),
    )
    assert not isinstance(updated, str)
    assert state._events_pending[1].summary == "updated dst"

    added_edge = state.upsert_edge(
        {
            "src": 1,
            "dst": 2,
            "kind": "data",
            "reason": "connects",
            "cited_entities": ["abnormal_traces"],
        }
    )
    assert not isinstance(added_edge, str)
    assert len(state._edges_pending) == 1

    updated_edge = state.upsert_edge(
        {
            "src": 1,
            "dst": 2,
            "kind": "data",
            "reason": "updated reason",
            "cited_entities": ["abnormal_traces"],
        }
    )
    assert not isinstance(updated_edge, str)
    assert state._edges_pending[0].reason == "updated reason"

    deleted_edge = state.delete_edge({"src": 1, "dst": 2})
    assert not isinstance(deleted_edge, str)
    assert state._edges_pending == []

    deleted_node = state.delete_node(2)
    assert not isinstance(deleted_node, str)
    assert [ev.id for ev in state._events_pending] == [1]


def test_delete_node_rejects_unknown_node() -> None:
    state = _state()
    err = state.delete_node(999)
    assert isinstance(err, str)
    assert "unknown node_id" in err


def test_commit_with_empty_events_is_accepted() -> None:
    state = _state()
    err = state.commit([])
    assert err is None
    assert state.events == ()
    assert state.edges == ()
    assert state.committed is True


# --- event-shape hard rejects (state unchanged) ----------------------------


def test_commit_rejects_id_below_next_event_id() -> None:
    state = _state()
    state.next_event_id = 41
    err = state.commit([_evid(1, turns=[10])])  # below cursor
    assert err is not None and "below next_event_id" in err
    assert state.committed is False
    assert state.events == ()


def test_commit_rejects_non_increasing_ids() -> None:
    state = _state()
    state.next_event_id = 41
    err = state.commit(
        [
            _evid(41, turns=[10]),
            _evid(41, turns=[11]),  # duplicate — must strictly increase
        ]
    )
    assert err is not None and "strictly greater" in err
    assert state.committed is False


def test_commit_allows_gaps_in_id_sequence() -> None:
    """Strictly increasing ≠ contiguous. Gaps are fine."""
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10], summary="src"),
            {
                **_evid(5, turns=[11], summary="dst"),
                "refs": [
                    {
                        "to": 1,
                        "kind": "data",
                        "reason": "ok",
                        "cited_entities": ["abnormal_traces"],
                    }
                ],
            },
        ]
    )
    assert err is None
    assert len(state.events) == 2
    assert state.events[0].id == 1 and state.events[1].id == 5


def test_commit_rejects_unknown_event_kind() -> None:
    state = _state()
    err = state.commit(
        [
            {
                "id": 1,
                "kind": "not_a_real_kind",
                "summary": "x",
                "source_turns": [10],
            }
        ]
    )
    assert err is not None and "kind" in err
    assert state.committed is False


def test_commit_rejects_empty_summary() -> None:
    state = _state()
    err = state.commit(
        [
            {"id": 1, "kind": "evid", "summary": "   ", "source_turns": [10]},
        ]
    )
    assert err is not None and "summary" in err
    assert state.committed is False


def test_commit_rejects_empty_source_turns() -> None:
    state = _state()
    err = state.commit([{"id": 1, "kind": "evid", "summary": "x", "source_turns": []}])
    assert err is not None and "source_turns" in err
    assert state.committed is False


# --- ref-shape hard rejects ------------------------------------------------


def test_commit_rejects_ref_to_self_or_later_event() -> None:
    """``refs[].to`` must be < self.id — guarantees no cycles by construction."""
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10]),
            {
                **_evid(2, turns=[11]),
                "refs": [
                    {
                        "to": 2,  # self-ref
                        "kind": "ref",
                        "reason": "x",
                        "cited_quote": "abnormal_traces",
                    }
                ],
            },
        ]
    )
    assert err is not None and "EARLIER" in err
    assert state.committed is False


def test_commit_rejects_ref_to_unknown_event_id() -> None:
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10]),
            {
                **_evid(2, turns=[11]),
                "refs": [
                    {
                        "to": 99,
                        "kind": "ref",
                        "reason": "x",
                        "cited_quote": "abnormal_traces",
                    }
                ],
            },
        ]
    )
    assert err is not None and "99" in err
    assert state.committed is False


def test_commit_rejects_data_ref_with_empty_cited_entities() -> None:
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10]),
            {
                **_evid(2, turns=[11]),
                "refs": [
                    {
                        "to": 1,
                        "kind": "data",
                        "reason": "x",
                        "cited_entities": [],
                    }
                ],
            },
        ]
    )
    assert err is not None and "cited_entities" in err
    assert state.committed is False


def test_commit_rejects_ref_kind_with_empty_cited_quote() -> None:
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10]),
            {
                **_evid(2, turns=[11]),
                "refs": [
                    {
                        "to": 1,
                        "kind": "ref",
                        "reason": "x",
                        "cited_quote": "",
                    }
                ],
            },
        ]
    )
    assert err is not None and "cited_quote" in err
    assert state.committed is False


# --- witness drops (partial success) ---------------------------------------


def test_commit_drops_ref_when_witness_substring_missing() -> None:
    """Witness failure → ref dropped, events accepted, partial entry recorded."""
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10]),
            {
                **_evid(2, turns=[11]),
                "refs": [
                    {
                        "to": 1,
                        "kind": "ref",
                        "reason": "x",
                        "cited_quote": "this phrase is absent xyzzy",
                    }
                ],
            },
        ]
    )
    assert err is None  # partial success is not a hard reject
    assert len(state.events) == 2
    assert state.edges == ()
    assert len(state.dropped_edges) == 1
    dropped = state.dropped_edges[0]
    assert dropped["src"] == 1 and dropped["dst"] == 2 and dropped["kind"] == "ref"
    assert "last_error" in dropped


def test_commit_partial_keeps_witnessed_refs_and_drops_failing_ones() -> None:
    state = ExtractionState(
        turn_texts={
            10: "good token here: alpha bravo charlie",
            11: "alpha bravo charlie reappears here",
        }
    )
    err = state.commit(
        [
            _evid(1, turns=[10]),
            {
                **_evid(2, turns=[11]),
                "refs": [
                    {
                        "to": 1,
                        "kind": "ref",
                        "reason": "good ref",
                        "cited_quote": "alpha bravo charlie",
                    },
                    {
                        "to": 1,
                        "kind": "data",
                        "reason": "bad data ref",
                        "cited_entities": ["nonexistent"],
                    },
                ],
            },
        ]
    )
    assert err is None
    assert len(state.edges) == 1
    assert state.edges[0].kind.value == "ref"
    assert len(state.dropped_edges) == 1
    assert state.dropped_edges[0]["kind"] == "data"


# --- one-shot commit invariant ---------------------------------------------


# --- refs presence rule (genesis exception) --------------------------------


def test_commit_rejects_non_first_event_with_empty_refs() -> None:
    """Every event after the first MUST cite at least one earlier event
    (in this firing or via external_refs to recent_graph). Empty refs
    on a later event leave the auditor without a causal trace.
    """
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10]),
            _evid(2, turns=[11]),  # no refs — rejected
        ]
    )
    assert err is not None
    assert "no refs and no external_refs" in err
    assert "genesis exemption" in err
    assert "Earlier-event candidates" in err
    assert "id:1" in err  # the available earlier event is enumerated
    assert state.committed is False


def test_commit_rejects_first_event_when_recent_graph_non_empty() -> None:
    """Genesis exemption only applies when there are NO priors at all.
    A new firing with recent_graph entries must connect its first
    event to the cumulative graph via external_refs, otherwise the
    cumulative graph grows a stray root.
    """
    from llmharness.schema import Event, EventKind

    prior = Event(id=1, kind=EventKind("evid"), summary="prior", source_turns=[5])
    state = ExtractionState(
        turn_texts={5: "irrelevant prior text", 10: "irrelevant new text"},
        recent_graph=(prior,),
        next_event_id=2,
    )
    err = state.commit([_evid(2, turns=[10])])  # first-of-firing, NO refs
    assert err is not None
    assert "genesis exemption" in err
    assert state.committed is False


def test_commit_accepts_genesis_event_with_empty_refs() -> None:
    """id=1 has no in-window predecessor; empty refs is allowed."""
    state = _state()
    err = state.commit([_evid(1, turns=[10])])
    assert err is None
    assert state.events[0].id == 1


def test_commit_is_one_shot() -> None:
    state = _state()
    first = state.commit([_evid(1, turns=[10])])
    assert first is None
    second = state.commit([_evid(1, turns=[10])])
    assert second is not None and "already committed" in second


# --- external_refs (cross-firing) --------------------------------------------


def test_external_ref_accepted_and_attached_to_event() -> None:
    """A witnessed external_ref must land on the event's ``external_refs``
    tuple — that's the only carrier of cross-firing connectivity until
    the aggregator resolves it to a real edge."""
    from llmharness.schema import EdgeKind, Event, EventKind

    prior = Event(
        id=1,
        kind=EventKind("evid"),
        summary="prior",
        source_turns=[5],
    )
    state = ExtractionState(
        turn_texts={
            5: "the abnormal_traces table appears in the prior turn",
            10: "we now query abnormal_traces again",
        },
        recent_graph=(prior,),
    )
    err = state.commit(
        [
            {
                "id": 1,
                "kind": "evid",
                "summary": "new",
                "source_turns": [10],
                "refs": [],
                "external_refs": [
                    {
                        "to_recent_event_id": 1,
                        "kind": "data",
                        "reason": "same table",
                        "cited_entities": ["abnormal_traces"],
                    }
                ],
            }
        ]
    )
    assert err is None, err
    assert len(state.events) == 1
    ev = state.events[0]
    assert len(ev.external_refs) == 1
    er = ev.external_refs[0]
    assert er.to_recent_event_id == 1
    assert er.kind is EdgeKind.DATA


def test_external_ref_alone_satisfies_connection_requirement() -> None:
    """An event can satisfy the connection requirement via external_ref
    alone — refs (in-firing) and external_refs are OR'd. Important for
    the first event of any firing N>=2 where the only available parents
    live in recent_graph."""
    from llmharness.schema import Event, EventKind

    prior = Event(
        id=1,
        kind=EventKind("evid"),
        summary="prior",
        source_turns=[5],
    )
    state = ExtractionState(
        turn_texts={
            5: "abnormal_traces was discussed earlier",
            11: "now also mentions abnormal_traces",
        },
        recent_graph=(prior,),
        next_event_id=2,
    )
    err = state.commit(
        [
            {
                "id": 2,
                "kind": "evid",
                "summary": "first event of this firing, external_ref only",
                "source_turns": [11],
                "refs": [],
                "external_refs": [
                    {
                        "to_recent_event_id": 1,
                        "kind": "data",
                        "reason": "uses prior table",
                        "cited_entities": ["abnormal_traces"],
                    }
                ],
            },
        ]
    )
    assert err is None, err
    assert state.events[0].external_refs[0].to_recent_event_id == 1


def test_external_ref_unknown_id_rejected() -> None:
    state = ExtractionState(turn_texts={10: "x"}, recent_graph=())
    err = state.commit(
        [
            {
                "id": 1,
                "kind": "evid",
                "summary": "g",
                "source_turns": [10],
                "refs": [],
                "external_refs": [
                    {
                        "to_recent_event_id": 99,
                        "kind": "data",
                        "reason": "r",
                        "cited_entities": ["x"],
                    }
                ],
            }
        ]
    )
    assert err is not None
    assert "not found in recent_graph" in err


# --- event-sourcing apply_* surface (cross-firing edits) ----------------------


def test_apply_node_delete_targets_prior_firing_node() -> None:
    """Cross-firing edit: the apply_* surface accepts deleting a node
    that came from a PRIOR firing (i.e. lives in ``recent_graph_dict``
    only, not in this firing's pending ops). The legacy ``delete_node``
    rejects this because it only sees the in-firing pending set.
    """
    from llmharness.audit.graph_ops import NodeDelete
    from llmharness.schema import Event, EventKind

    prior_a = Event(id=1, kind=EventKind("task"), summary="prior a", source_turns=[1])
    prior_b = Event(id=2, kind=EventKind("evid"), summary="prior b", source_turns=[2])
    state = ExtractionState(
        turn_texts={1: "alpha", 2: "bravo"},
        recent_graph_dict={1: prior_a, 2: prior_b},
        next_event_id=3,
    )

    result = state.apply_node_delete(1)
    assert not isinstance(result, str), result
    assert len(state.pending_ops) == 1
    op = state.pending_ops[0]
    assert isinstance(op, NodeDelete) and op.id == 1
    # Folded view drops node 1; node 2 remains visible to subsequent
    # apply_* calls in this firing.
    assert set(state.pending_graph.nodes.keys()) == {2}


def test_apply_edge_upsert_requires_existing_endpoints_in_folded_view() -> None:
    """The apply_* surface validates endpoint nodes against the folded
    graph (recent union pending), not just pending. Adding a node first
    and then an edge to it must work — this proves the test described
    in the brief: build on a prior node, then attach a new edge.
    """
    from llmharness.schema import Event, EventKind

    prior = Event(id=2, kind=EventKind("hyp"), summary="prior hyp", source_turns=[2])
    state = ExtractionState(
        turn_texts={2: "bravo alpha", 3: "charlie alpha"},
        recent_graph_dict={2: prior},
        next_event_id=3,
    )

    # Initial: try to point at a yet-unborn dst — must reject.
    err = state.apply_edge_upsert(
        {
            "src": 2,
            "dst": 99,
            "kind": "data",
            "reason": "r",
            "cited_entities": ["alpha"],
        }
    )
    assert isinstance(err, str) and "99" in err

    # Now create the dst, then attach the edge. Both succeed.
    res = state.apply_node_upsert(
        {"id": 3, "kind": "evid", "summary": "new", "source_turns": [3]}
    )
    assert not isinstance(res, str), res

    res = state.apply_edge_upsert(
        {
            "src": 2,
            "dst": 3,
            "kind": "data",
            "reason": "supports",
            "cited_entities": ["alpha"],
        }
    )
    assert not isinstance(res, str), res
    assert (2, 3, "data") in state.pending_graph.edges


def test_apply_edge_upsert_rejects_fabricated_ref_quote() -> None:
    """Witness validation for kind='ref' edges: the cited_quote must
    literally appear (after case+ws normalization) in the src node's
    source_turns text. This gate was missing from the original atomic
    upsert_edge — fabricated quotes slipped through.
    """
    state = ExtractionState(
        turn_texts={
            10: "the abnormal_traces table has rows",
            11: "we follow up on abnormal_traces",
        }
    )
    # Build two nodes in this firing first.
    state.apply_node_upsert(
        {"id": 1, "kind": "evid", "summary": "src", "source_turns": [10]}
    )
    state.apply_node_upsert(
        {"id": 2, "kind": "act", "summary": "dst", "source_turns": [11]}
    )

    # cited_quote not present in src node's turn text -> rejected.
    err = state.apply_edge_upsert(
        {
            "src": 1,
            "dst": 2,
            "kind": "ref",
            "reason": "r",
            "cited_quote": "never said xyzzy",
        }
    )
    assert isinstance(err, str)
    assert "cited_quote" in err

    # A real quote from turn 10 -> accepted.
    ok = state.apply_edge_upsert(
        {
            "src": 1,
            "dst": 2,
            "kind": "ref",
            "reason": "r",
            "cited_quote": "abnormal_traces",
        }
    )
    assert not isinstance(ok, str), ok
    assert (1, 2, "ref") in state.pending_graph.edges


def test_apply_node_delete_unknown_id_is_rejected() -> None:
    state = ExtractionState(turn_texts={1: "alpha"})
    err = state.apply_node_delete(42)
    assert isinstance(err, str) and "42" in err
    assert state.pending_ops == []


def test_apply_edge_delete_requires_kind() -> None:
    state = ExtractionState(
        turn_texts={1: "alpha", 2: "bravo alpha"}
    )
    state.apply_node_upsert({"id": 1, "kind": "evid", "summary": "s", "source_turns": [1]})
    state.apply_node_upsert({"id": 2, "kind": "act", "summary": "d", "source_turns": [2]})
    state.apply_edge_upsert(
        {"src": 1, "dst": 2, "kind": "data", "reason": "r", "cited_entities": ["alpha"]}
    )
    # No kind in selector -> rejected; the op-log selector contract
    # requires the full triple.
    err = state.apply_edge_delete({"src": 1, "dst": 2})
    assert isinstance(err, str) and "kind" in err
    # With kind -> accepted; folded graph drops the edge.
    ok = state.apply_edge_delete({"src": 1, "dst": 2, "kind": "data"})
    assert not isinstance(ok, str), ok
    assert (1, 2, "data") not in state.pending_graph.edges


def test_external_ref_witness_failure_drops_into_dropped_edges() -> None:
    """Witness failures on external_refs DROP the ref (recorded for
    debugging) but keep the event — partial-success path symmetric with
    in-firing refs."""
    from llmharness.schema import Event, EventKind

    prior = Event(
        id=1,
        kind=EventKind("evid"),
        summary="prior",
        source_turns=[5],
    )
    state = ExtractionState(
        turn_texts={5: "irrelevant prior text", 10: "this turn says nothing"},
        recent_graph=(prior,),
    )
    err = state.commit(
        [
            {
                "id": 1,
                "kind": "evid",
                "summary": "g",
                "source_turns": [10],
                "refs": [],
                "external_refs": [
                    {
                        "to_recent_event_id": 1,
                        "kind": "data",
                        "reason": "fabricated",
                        "cited_entities": ["abnormal_traces"],
                    }
                ],
            }
        ]
    )
    assert err is None
    assert len(state.events[0].external_refs) == 0
    assert len(state.dropped_edges) == 1
    assert state.dropped_edges[0]["src"] == "recent_graph_event#1"
