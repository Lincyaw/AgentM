"""Fail-stop tests for the case-aggregation pipeline.

Why fail-stop: the aggregator is the source-of-truth view for human
case review and any downstream training-data export. If sequencing,
graph accumulation, or per-firing input/output capture drifts, the
review will not match the run and the SFT data derived from these
files will be wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmharness.aggregate import collect_case


def _replay_record(
    *,
    phase: str,
    turn_index: int,
    ts_ns: int,
    session_id: str,
    trace_id: str | None = None,
    compose_kwargs: dict[str, Any],
    payload: dict[str, Any],
    output: dict[str, Any] | None,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "phase": phase,
        "turn_index": turn_index,
        "session_id": session_id,
        "trace_id": trace_id if trace_id is not None else f"trace-{session_id}",
        "ts_ns": ts_ns,
        "compose_kwargs": compose_kwargs,
        "payload": payload,
        "provider": None,
        "output": output,
        "status": status,
        "error": None,
        "latency_ms": 100,
        "extras": {},
    }


@pytest.fixture
def sample_run(tmp_path: Path) -> tuple[Path, Path]:
    """One sample's worth of replay log + meta sidecar."""
    sid = "sess-abc123"
    replay_path = tmp_path / "audit_replay" / f"{sid}.jsonl"
    meta_path = tmp_path / "audit_replay" / f"{sid}.meta.json"
    replay_path.parent.mkdir(parents=True)

    traj_snapshot = [
        {"index": 0, "role": "user", "content": [{"type": "text", "text": "go"}]},
        {"index": 1, "role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]

    records = [
        _replay_record(
            phase="extractor",
            turn_index=1,
            ts_ns=1_000_000_000,
            session_id=sid,
            compose_kwargs={},
            payload={"new_turns": traj_snapshot[:1], "recent_graph": []},
            output={
                "events": [{"id": 1, "kind": "task", "summary": "go", "source_turns": [0]}],
                "edges": [],
                "dropped_edges": [],
            },
        ),
        _replay_record(
            phase="extractor",
            turn_index=2,
            ts_ns=2_000_000_000,
            session_id=sid,
            compose_kwargs={},
            payload={"new_turns": traj_snapshot[1:], "recent_graph": []},
            output={
                "events": [{"id": 2, "kind": "hyp", "summary": "h", "source_turns": [1]}],
                "edges": [],
                "dropped_edges": [],
            },
        ),
        _replay_record(
            phase="auditor",
            turn_index=2,
            ts_ns=3_000_000_000,
            session_id=sid,
            compose_kwargs={
                "trajectory_snapshot": traj_snapshot,
                "events": [],
                "edges": [],
                "findings": [],
                "check_errors": {},
                "continuation_notes": [],
                "summary_threshold": 30,
                "tools": ["submit_verdict"],
            },
            payload={"graph": [], "recent_verdicts": []},
            output={
                "surface_reminder": True,
                "reminder_text": "verify",
                "matched_event_ids": [1],
                "continuation_notes": ["watch hyp 1"],
            },
        ),
    ]

    with replay_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    meta_path.write_text(
        json.dumps(
            {
                "sample_id": "rca-mysql-001",
                "dataset_name": "rca-toy",
                "dataset_path": "/data/rca.jsonl",
                "session_id": sid,
                "trace_id": f"trace-{sid}",
            }
        ),
        encoding="utf-8",
    )

    return replay_path, meta_path


# --- collector --------------------------------------------------------------


def test_sidecar_and_meta_share_session_stem_and_join(
    sample_run: tuple[Path, Path],
) -> None:
    """The core fix: a sidecar and its meta written for the same session
    share the ``session_id`` stem, so the stem-paired collector joins the
    meta onto the case.

    Before the rename, the sidecar was keyed by ``session_id`` while the
    meta was keyed by the OTel ``trace_id`` — the two stems diverged on
    live runs and ``collect_case`` silently failed to attach sample_id /
    dataset metadata. Here both files share ``sess-abc123`` and the join
    succeeds.
    """
    replay_path, meta_path = sample_run
    # Both artefacts are keyed by the SAME stem (the session_id).
    assert replay_path.stem == meta_path.name.removesuffix(".meta.json")

    case = collect_case(replay_path=replay_path, meta_path=meta_path)

    # Meta joined: sample/dataset metadata propagated, and case_id derives
    # from the sample id (not the bare session id).
    assert case.meta.sample_id == "rca-mysql-001"
    assert case.meta.dataset_name == "rca-toy"
    assert case.meta.case_id == "rca-mysql-001"
    # Both identity fields surface in the case meta with core's vocabulary.
    assert case.meta.session_id == "sess-abc123"
    assert case.meta.trace_id == "trace-sess-abc123"


def test_case_meta_has_no_legacy_root_session_id(
    sample_run: tuple[Path, Path],
) -> None:
    """The aggregate ``meta.json`` carries ``session_id`` + ``trace_id``,
    never the removed ``root_session_id`` key."""
    from dataclasses import asdict

    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    serialized = asdict(case.meta)
    assert "root_session_id" not in serialized
    assert serialized["session_id"] == "sess-abc123"
    assert serialized["trace_id"] == "trace-sess-abc123"


# --- writer ----------------------------------------------------------------


def test_graph_snapshot_concatenates_globally_unique_ids(
    tmp_path: Path,
) -> None:
    """Two firings already emit globally-unique ids (LIVE adapter
    contract): firing 1 = ids 1..3, firing 2 = ids 4..6. The collector
    concatenates without renumbering; the cumulative snapshot after
    firing 2 holds 6 distinct ids and 4 edges, src/dst preserved."""
    sid = "sess-concat"
    replay_path = tmp_path / "audit_replay" / f"{sid}.jsonl"
    replay_path.parent.mkdir(parents=True)
    fr1 = {
        "events": [
            {"id": 1, "kind": "task", "summary": "t", "source_turns": [0]},
            {"id": 2, "kind": "hyp", "summary": "h", "source_turns": [1]},
            {"id": 3, "kind": "act", "summary": "a", "source_turns": [1]},
        ],
        "edges": [
            {"src": 1, "dst": 2, "kind": "data"},
            {"src": 2, "dst": 3, "kind": "ref"},
        ],
        "dropped_edges": [],
    }
    fr2 = {
        "events": [
            {"id": 4, "kind": "task", "summary": "t", "source_turns": [2]},
            {"id": 5, "kind": "hyp", "summary": "h", "source_turns": [3]},
            {"id": 6, "kind": "act", "summary": "a", "source_turns": [3]},
        ],
        "edges": [
            {"src": 4, "dst": 5, "kind": "data"},
            {"src": 5, "dst": 6, "kind": "ref"},
        ],
        "dropped_edges": [],
    }
    records = [
        _replay_record(
            phase="extractor",
            turn_index=1,
            ts_ns=1,
            session_id=sid,
            compose_kwargs={},
            payload={"new_turns": [], "recent_graph": []},
            output=fr1,
        ),
        _replay_record(
            phase="extractor",
            turn_index=3,
            ts_ns=2,
            session_id=sid,
            compose_kwargs={},
            payload={"new_turns": [], "recent_graph": []},
            output=fr2,
        ),
    ]
    with replay_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    case = collect_case(replay_path=replay_path, meta_path=None)
    snap2 = case.graph_snapshots[1]
    ids = [ev["id"] for ev in snap2.events]
    assert ids == [1, 2, 3, 4, 5, 6], ids
    pairs = [(e["src"], e["dst"]) for e in snap2.edges]
    assert pairs == [(1, 2), (2, 3), (4, 5), (5, 6)], pairs


def test_external_ref_resolves_to_cross_firing_edge_in_snapshot(
    tmp_path: Path,
) -> None:
    """Two firings: firing 1 emits one event; firing 2 emits an event
    whose external_refs[0] targets recent_graph[1] (= firing 1's event).
    The cumulative snapshot after firing 2 must carry a cross-firing
    edge from firing-1's event to firing-2's event, with src/dst in the
    renumbered global id space."""
    sid = "sess-ext"
    replay_path = tmp_path / "audit_replay" / f"{sid}.jsonl"
    replay_path.parent.mkdir(parents=True)
    records = [
        _replay_record(
            phase="extractor",
            turn_index=1,
            ts_ns=1,
            session_id=sid,
            compose_kwargs={},
            payload={"new_turns": [], "recent_graph": []},
            output={
                "events": [
                    {
                        "id": 1,
                        "kind": "task",
                        "summary": "first",
                        "source_turns": [0],
                        "external_refs": [],
                    }
                ],
                "edges": [],
                "dropped_edges": [],
            },
        ),
        _replay_record(
            phase="extractor",
            turn_index=3,
            ts_ns=2,
            session_id=sid,
            compose_kwargs={},
            payload={"new_turns": [], "recent_graph": []},
            output={
                "events": [
                    {
                        "id": 2,
                        "kind": "act",
                        "summary": "follow-up",
                        "source_turns": [3],
                        "external_refs": [
                            {
                                "to_recent_event_id": 1,
                                "kind": "data",
                                "reason": "answers task",
                                "cited_entities": ["foo"],
                                "cited_quote": "",
                            }
                        ],
                    }
                ],
                "edges": [],
                "dropped_edges": [],
            },
        ),
    ]
    with replay_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    case = collect_case(replay_path=replay_path, meta_path=None)
    snap2 = case.graph_snapshots[1]
    assert [ev["id"] for ev in snap2.events] == [1, 2]
    assert len(snap2.edges) == 1
    edge = snap2.edges[0]
    assert edge["src"] == 1
    assert edge["dst"] == 2
    assert edge["kind"] == "data"
    assert edge["src_turns"] == [0]
    assert edge["dst_turns"] == [3]
    # external_refs metadata should not leak onto the snapshot event;
    # it has been consumed into an edge.
    assert "external_refs" not in snap2.events[1] or snap2.events[1]["external_refs"] == []
