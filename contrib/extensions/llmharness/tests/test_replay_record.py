"""Replay-record I/O smoke tests.

Why this exists: the replay sidecar is the *contract* between the live
audit pipeline and the offline replay CLI. A schema drift here silently
makes every captured trace unreplayable, which is exactly the failure
mode the broader fail-stop testing philosophy in CLAUDE.md asks us to
guard. Two cheap roundtrip tests are enough — the LLM-driving paths are
exercised by the integration suite.
"""

from __future__ import annotations

from pathlib import Path

from llmharness.replay.record import (
    ReplayRecord,
    iter_records,
    read_records,
    replay_log_path,
    write_record,
)


def _make_record(phase: str, turn: int, status: str = "ok") -> ReplayRecord:
    return ReplayRecord(
        phase=phase,  # type: ignore[arg-type]
        turn_index=turn,
        root_session_id="abc123" * 4,
        ts_ns=1_000_000_000,
        compose_kwargs={"prompt_override": None, "summary_threshold": 30},
        payload={"graph": [], "recent_verdicts": []},
        provider=["llmharness.providers.test", {}],
        output={"verdict": {"surface_reminder": False}},
        status=status,  # type: ignore[arg-type]
        latency_ms=42,
    )


def test_record_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    original = _make_record("auditor", turn=14)
    write_record(path, original)

    records = list(iter_records(path))
    assert len(records) == 1
    rec = records[0]
    assert rec.phase == original.phase
    assert rec.turn_index == original.turn_index
    assert rec.compose_kwargs == original.compose_kwargs
    assert rec.payload == original.payload
    assert rec.output == original.output
    assert rec.status == original.status
    assert rec.latency_ms == original.latency_ms


def test_read_records_filters(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    write_record(path, _make_record("extractor", turn=0))
    write_record(path, _make_record("extractor", turn=1, status="no_call"))
    write_record(path, _make_record("auditor", turn=14))

    assert len(read_records(path, phase="extractor")) == 2
    assert len(read_records(path, phase="auditor")) == 1
    assert [r.turn_index for r in read_records(path, phase="extractor")] == [0, 1]
    assert read_records(path, turn_index=14)[0].phase == "auditor"


def test_replay_log_path_layout(tmp_path: Path) -> None:
    # Contract: sidecar lives under <cwd>/.agentm/audit_replay/<trace>.jsonl
    # — view_traj.py and downstream tooling assume this layout.
    p = replay_log_path(tmp_path, "trace-xyz")
    assert p.parent == tmp_path / ".agentm" / "audit_replay"
    assert p.name == "trace-xyz.jsonl"


def test_malformed_lines_ignored(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    write_record(path, _make_record("extractor", turn=0))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")
        fh.write('{"phase":"extractor"}\n')  # missing required fields
    write_record(path, _make_record("auditor", turn=10))

    records = list(iter_records(path))
    # Two good + one missing-fields KeyError (skipped) + one not-JSON (skipped).
    assert len(records) == 2
    assert {r.phase for r in records} == {"extractor", "auditor"}
