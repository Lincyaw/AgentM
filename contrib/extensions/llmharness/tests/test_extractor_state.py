"""ExtractionState fail-stop tests for v3.1 single-tool flow.

The extractor LLM submits the entire graph in ONE ``submit_events``
call; ``ExtractionState.commit`` is the lone validation gate. Bugs
here corrupt every extracted graph downstream, so the tests pin the
load-bearing positions:

- event-shape errors hard-reject the whole submission (state
  unchanged → LLM may retry with the error message)
- id ordering must be 1..N strictly increasing
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


def test_commit_with_empty_events_is_accepted() -> None:
    state = _state()
    err = state.commit([])
    assert err is None
    assert state.events == ()
    assert state.edges == ()
    assert state.committed is True


# --- event-shape hard rejects (state unchanged) ----------------------------


def test_commit_rejects_non_sequential_ids() -> None:
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10]),
            _evid(3, turns=[11]),  # gap: should have been 2
        ]
    )
    assert err is not None and "1, 2, 3" in err
    assert state.committed is False
    assert state.events == ()


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


def test_commit_rejects_non_genesis_event_with_empty_refs() -> None:
    """id>=2 events MUST cite at least one earlier event in this firing.

    Empty / missing refs on non-genesis events leave the auditor without
    a causal trace across the window — see schema description on
    ``_EVENT_SCHEMA.refs``.
    """
    state = _state()
    err = state.commit(
        [
            _evid(1, turns=[10]),
            _evid(2, turns=[11]),  # no refs — rejected
        ]
    )
    assert err is not None
    assert "events[1].refs must be non-empty" in err
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
