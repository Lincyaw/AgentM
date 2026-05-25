"""Fail-stop: the session/trace identity vocabulary on the replay-record
and meta sidecars must round-trip under core's two-field scheme.

Why fail-stop: ``session_id`` is the sidecar/meta filename stem (= the
``.agentm/observability/<session_id>.jsonl`` log name) and ``trace_id``
is the whole-tree group id (= ``ExtensionAPI.root_session_id``). The
aggregate / distill stem-pairing joins a sidecar to its meta by the
shared ``session_id`` stem; if either record drops a field or re-emits
the removed ``root_session_id`` key the join corrupts silently. There is
no backward-compat reader, so the serialized shape is the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmharness.distill.binding import (
    SampleMeta,
    meta_path_for,
    read_sample_meta,
)
from llmharness.replay.record import (
    ReplayRecord,
    iter_records,
    replay_log_path,
    write_record,
)


def _record(session_id: str, trace_id: str) -> ReplayRecord:
    return ReplayRecord(
        phase="extractor",
        turn_index=3,
        session_id=session_id,
        trace_id=trace_id,
        ts_ns=1_700_000_000_000,
        compose_kwargs={"base_prompt": "x"},
        payload={"new_turns": []},
        provider=None,
        output={"events": []},
        status="ok",
    )


def test_replay_record_round_trips_session_and_trace_ids() -> None:
    rec = _record("sess-9", "trace-9")

    via_dict = ReplayRecord.from_dict(rec.to_dict())
    assert via_dict.session_id == "sess-9"
    assert via_dict.trace_id == "trace-9"

    via_jsonl = ReplayRecord.from_dict(json.loads(rec.to_jsonl()))
    assert via_jsonl.session_id == "sess-9"
    assert via_jsonl.trace_id == "trace-9"


def test_replay_record_serialization_has_no_root_session_id() -> None:
    rec = _record("sess-9", "trace-9")

    d = rec.to_dict()
    assert "root_session_id" not in d
    assert d["session_id"] == "sess-9"
    assert d["trace_id"] == "trace-9"

    jsonl = json.loads(rec.to_jsonl())
    assert "root_session_id" not in jsonl
    assert jsonl["session_id"] == "sess-9"
    assert jsonl["trace_id"] == "trace-9"


def test_sample_meta_round_trips_and_drops_root_session_id() -> None:
    meta = SampleMeta(
        sample_id="rca-001",
        dataset_name="ds",
        dataset_path="/data.jsonl",
        session_id="sess-9",
        trace_id="trace-9",
    )
    d = meta.to_dict()
    assert "root_session_id" not in d
    assert d["session_id"] == "sess-9"
    assert d["trace_id"] == "trace-9"


def test_sidecar_and_meta_paths_share_session_stem(tmp_path: Path) -> None:
    """The sidecar (``replay_log_path``) and meta (``meta_path_for``) must
    be keyed by the SAME ``session_id`` so the stem-pairing join works."""
    session_id = "sess-stem"
    sidecar = replay_log_path(tmp_path, session_id)
    meta = meta_path_for(tmp_path, session_id)
    assert sidecar.stem == session_id
    assert meta.name == f"{session_id}.meta.json"
    assert sidecar.parent == meta.parent


def test_meta_sidecar_reader_only_reads_new_keys(tmp_path: Path) -> None:
    """No backward-compat: an old-shape meta carrying only the removed
    ``root_session_id`` key reads back with empty session_id / trace_id."""
    path = tmp_path / "old.meta.json"
    path.write_text(
        json.dumps(
            {
                "sample_id": "rca-001",
                "dataset_name": "ds",
                "dataset_path": "/d.jsonl",
                "root_session_id": "sess-legacy",
            }
        ),
        encoding="utf-8",
    )
    meta = read_sample_meta(path)
    assert meta is not None
    assert meta.session_id == ""
    assert meta.trace_id == ""


def test_written_sidecar_round_trips_via_iter_records(tmp_path: Path) -> None:
    path = tmp_path / "sess-9.jsonl"
    write_record(path, _record("sess-9", "trace-9"))
    (rec,) = list(iter_records(path))
    assert rec.session_id == "sess-9"
    assert rec.trace_id == "trace-9"
