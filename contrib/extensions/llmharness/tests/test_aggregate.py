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

from llmharness.aggregate import collect_case, write_case
from llmharness.aggregate.case import CaseLayout


def _replay_record(
    *,
    phase: str,
    turn_index: int,
    ts_ns: int,
    root_session_id: str,
    compose_kwargs: dict[str, Any],
    payload: dict[str, Any],
    output: dict[str, Any] | None,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "phase": phase,
        "turn_index": turn_index,
        "root_session_id": root_session_id,
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
            root_session_id=sid,
            compose_kwargs={},
            payload={"new_turns": traj_snapshot[:1], "recent_graph": []},
            output={
                "events": [
                    {"id": 1, "kind": "task", "summary": "go", "source_turns": [0]}
                ],
                "edges": [],
                "dropped_edges": [],
            },
        ),
        _replay_record(
            phase="extractor",
            turn_index=2,
            ts_ns=2_000_000_000,
            root_session_id=sid,
            compose_kwargs={},
            payload={"new_turns": traj_snapshot[1:], "recent_graph": []},
            output={
                "events": [
                    {"id": 2, "kind": "hyp", "summary": "h", "source_turns": [1]}
                ],
                "edges": [],
                "dropped_edges": [],
            },
        ),
        _replay_record(
            phase="auditor",
            turn_index=2,
            ts_ns=3_000_000_000,
            root_session_id=sid,
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
                "cited_cards": [],
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
                "root_session_id": sid,
            }
        ),
        encoding="utf-8",
    )

    return replay_path, meta_path


# --- collector --------------------------------------------------------------


def test_case_id_prefers_sample_id_over_session_id(sample_run: tuple[Path, Path]) -> None:
    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    assert case.meta.case_id == "rca-mysql-001"
    assert case.meta.root_session_id == "sess-abc123"
    assert case.meta.sample_id == "rca-mysql-001"


def test_case_id_falls_back_to_session_id_when_meta_missing(
    sample_run: tuple[Path, Path],
) -> None:
    replay_path, _meta = sample_run
    case = collect_case(replay_path=replay_path, meta_path=None)
    assert case.meta.case_id == "sess-abc123"
    assert case.meta.sample_id is None


def test_extractor_firings_are_sequenced_in_arrival_order(
    sample_run: tuple[Path, Path],
) -> None:
    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    assert [fr.sequence for fr in case.extractor_firings] == [1, 2]
    assert [fr.turn_index for fr in case.extractor_firings] == [1, 2]


def test_graph_snapshots_accumulate_across_firings(
    sample_run: tuple[Path, Path],
) -> None:
    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    assert len(case.graph_snapshots) == 2
    assert len(case.graph_snapshots[0].events) == 1
    # Second snapshot must be cumulative — event from firing 1 still present.
    assert len(case.graph_snapshots[1].events) == 2
    # Globally-unique ids come straight from the live adapter — the
    # collector concatenates without renumbering. Two distinct ids in,
    # two distinct ids out.
    ids = [ev["id"] for ev in case.graph_snapshots[1].events]
    assert ids == sorted(set(ids)) and len(set(ids)) == 2, (
        f"snapshot ids must be globally unique, got {ids}"
    )


def test_main_agent_messages_come_from_last_auditor_trajectory_snapshot(
    sample_run: tuple[Path, Path],
) -> None:
    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    assert len(case.main_agent_messages) == 2
    assert case.main_agent_messages[0]["role"] == "user"
    assert case.main_agent_messages[1]["role"] == "assistant"


def test_verdict_counts_match_auditor_output(sample_run: tuple[Path, Path]) -> None:
    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    assert case.meta.surfaced_reminders == 1
    assert case.meta.silent_verdicts == 0
    assert case.verdicts[0]["reminder_text"] == "verify"


# --- writer ----------------------------------------------------------------


def test_write_case_produces_canonical_layout(
    sample_run: tuple[Path, Path], tmp_path: Path
) -> None:
    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    out = tmp_path / "cases"
    case_dir = write_case(case, out)

    layout = CaseLayout(root=case_dir)
    assert layout.meta_path.is_file()
    assert layout.main_agent_path.is_file()
    assert layout.verdicts_path.is_file()
    assert layout.trajectory_path.is_file()
    assert layout.readme_path.is_file()
    assert layout.firing_path("extractor", 1, 1).is_file()
    assert layout.firing_path("extractor", 2, 2).is_file()
    assert layout.firing_path("auditor", 1, 2).is_file()
    assert layout.snapshot_path(1).is_file()
    assert layout.snapshot_path(2).is_file()


def test_main_agent_jsonl_is_lossless_relative_to_collector(
    sample_run: tuple[Path, Path], tmp_path: Path
) -> None:
    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    case_dir = write_case(case, tmp_path / "cases")

    with (case_dir / "main_agent.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert rows == case.main_agent_messages


def test_trajectory_jsonl_lists_all_firings_with_refs(
    sample_run: tuple[Path, Path], tmp_path: Path
) -> None:
    replay_path, meta_path = sample_run
    case = collect_case(replay_path=replay_path, meta_path=meta_path)
    case_dir = write_case(case, tmp_path / "cases")

    with (case_dir / "trajectory.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 3  # 2 extractor + 1 auditor
    sources = [r["source"] for r in rows]
    assert sources.count("extractor") == 2
    assert sources.count("auditor") == 1
    # Every ref must point at an existing file in the case dir.
    for row in rows:
        assert (case_dir / row["ref"]).is_file()


def test_sample_id_override_takes_precedence_over_meta_sidecar(
    sample_run: tuple[Path, Path],
) -> None:
    """The CLI overrides must win over a meta sidecar so an aggregator
    invocation can re-tag a session that was collected without
    distill_binding mounted."""
    replay_path, meta_path = sample_run
    case = collect_case(
        replay_path=replay_path,
        meta_path=meta_path,
        sample_id_override="manual-tag",
        dataset_name_override="my-dataset",
        dataset_path_override="/abs/data.jsonl",
    )
    assert case.meta.case_id == "manual-tag"
    assert case.meta.sample_id == "manual-tag"
    assert case.meta.dataset_name == "my-dataset"
    assert case.meta.dataset_path == "/abs/data.jsonl"


def test_main_agent_trajectory_stitches_post_auditor_extractor_turns(
    tmp_path: Path,
) -> None:
    """If extractor firings occur after the last auditor firing, the
    main-agent trajectory must extend past the last auditor snapshot
    using extractor payload.new_turns. Without this stitch the case
    directory silently loses the trailing turns of every run that
    ends mid-auditor-interval."""
    sid = "sess-stitch"
    replay_path = tmp_path / "audit_replay" / f"{sid}.jsonl"
    replay_path.parent.mkdir(parents=True)

    snap_up_to_5 = [
        {"index": i, "role": "user" if i % 2 == 0 else "assistant", "content": [{"type": "text", "text": f"t{i}"}]}
        for i in range(6)
    ]
    new_turns_8 = [
        {"index": 6, "role": "user", "content": [{"type": "text", "text": "t6"}]},
        {"index": 7, "role": "assistant", "content": [{"type": "text", "text": "t7"}]},
        {"index": 8, "role": "user", "content": [{"type": "text", "text": "t8"}]},
    ]
    new_turns_10 = [
        {"index": 9, "role": "assistant", "content": [{"type": "text", "text": "t9"}]},
        {"index": 10, "role": "user", "content": [{"type": "text", "text": "t10"}]},
    ]
    records = [
        _replay_record(
            phase="extractor", turn_index=5, ts_ns=1, root_session_id=sid,
            compose_kwargs={}, payload={"new_turns": snap_up_to_5, "recent_graph": []},
            output={"events": [], "edges": [], "dropped_edges": []},
        ),
        _replay_record(
            phase="auditor", turn_index=5, ts_ns=2, root_session_id=sid,
            compose_kwargs={"trajectory_snapshot": snap_up_to_5},
            payload={}, output={"surface_reminder": False, "matched_event_ids": []},
        ),
        # Two extractor firings AFTER the last auditor firing — their
        # new_turns must be picked up.
        _replay_record(
            phase="extractor", turn_index=8, ts_ns=3, root_session_id=sid,
            compose_kwargs={}, payload={"new_turns": new_turns_8, "recent_graph": []},
            output={"events": [], "edges": [], "dropped_edges": []},
        ),
        _replay_record(
            phase="extractor", turn_index=10, ts_ns=4, root_session_id=sid,
            compose_kwargs={}, payload={"new_turns": new_turns_10, "recent_graph": []},
            output={"events": [], "edges": [], "dropped_edges": []},
        ),
    ]
    with replay_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    case = collect_case(replay_path=replay_path, meta_path=None)
    indices = [m["index"] for m in case.main_agent_messages]
    assert indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (
        f"trajectory must extend past last auditor (turn 5) using extractor "
        f"new_turns; got indices {indices}"
    )


def test_main_agent_stitch_does_not_duplicate_messages_already_in_base(
    tmp_path: Path,
) -> None:
    """Extractor windows may overlap with the auditor's trajectory_snapshot
    (the snapshot already covers every turn up to the firing). The merge
    must dedupe by message ``index`` so messages appear once."""
    sid = "sess-dedupe"
    replay_path = tmp_path / "audit_replay" / f"{sid}.jsonl"
    replay_path.parent.mkdir(parents=True)

    snap_up_to_3 = [
        {"index": i, "role": "user", "content": [{"type": "text", "text": f"t{i}"}]}
        for i in range(4)
    ]
    # Extractor record AFTER the auditor that contains overlapping
    # indices 2-3 plus a new index 4.
    new_turns_overlap = [
        {"index": 2, "role": "user", "content": [{"type": "text", "text": "t2"}]},
        {"index": 3, "role": "user", "content": [{"type": "text", "text": "t3"}]},
        {"index": 4, "role": "user", "content": [{"type": "text", "text": "t4"}]},
    ]
    records = [
        _replay_record(
            phase="auditor", turn_index=3, ts_ns=1, root_session_id=sid,
            compose_kwargs={"trajectory_snapshot": snap_up_to_3},
            payload={}, output={"surface_reminder": False, "matched_event_ids": []},
        ),
        _replay_record(
            phase="extractor", turn_index=4, ts_ns=2, root_session_id=sid,
            compose_kwargs={}, payload={"new_turns": new_turns_overlap, "recent_graph": []},
            output={"events": [], "edges": [], "dropped_edges": []},
        ),
    ]
    with replay_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    case = collect_case(replay_path=replay_path, meta_path=None)
    indices = [m["index"] for m in case.main_agent_messages]
    assert indices == [0, 1, 2, 3, 4], (
        f"overlapping extractor turns must dedupe; got {indices}"
    )


def test_failed_extractor_firing_does_not_advance_graph_snapshot(
    tmp_path: Path,
) -> None:
    """If an extractor firing errors, the cursor doesn't move and no
    snapshot is produced — mirrors the live adapter's behavior."""
    sid = "sess-fail"
    replay_path = tmp_path / "audit_replay" / f"{sid}.jsonl"
    replay_path.parent.mkdir(parents=True)
    records = [
        _replay_record(
            phase="extractor",
            turn_index=1,
            ts_ns=1,
            root_session_id=sid,
            compose_kwargs={},
            payload={},
            output=None,
            status="spawn_error",
        ),
    ]
    with replay_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    case = collect_case(replay_path=replay_path, meta_path=None)
    assert case.meta.extractor_firings == 1
    assert case.graph_snapshots == []


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
            root_session_id=sid,
            compose_kwargs={},
            payload={"new_turns": [], "recent_graph": []},
            output=fr1,
        ),
        _replay_record(
            phase="extractor",
            turn_index=3,
            ts_ns=2,
            root_session_id=sid,
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
            root_session_id=sid,
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
            root_session_id=sid,
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
