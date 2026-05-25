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
    return {"id": eid, "kind": "act", "summary": summary, "source_turns": turns}


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
        kind=EventKind("act"),
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
                "kind": "act",
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






# --- event-sourcing apply_* surface (cross-firing edits) ----------------------




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
        {"id": 3, "kind": "act", "summary": "new", "source_turns": [3]}
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
        {"id": 1, "kind": "act", "summary": "src", "source_turns": [10]}
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






