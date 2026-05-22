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


def test_replay_record_thinking_roundtrip(tmp_path: Path) -> None:
    """raw_assistant_messages survives JSONL roundtrip with thinking + tool_call.

    Disaster guarded: if the field is dropped on either side of the
    serialization, downstream SFT exporters (Qwen/GLM-style ``<think>``
    targets) silently lose every reasoning trace and the trained model
    can't reproduce the teacher's thought process.
    """
    path = tmp_path / "replay.jsonl"
    blocks = [
        {"type": "thinking", "text": "let me check the second turn"},
        {
            "type": "tool_call",
            "id": "call-1",
            "name": "finalize_extraction",
            "arguments": {},
        },
    ]
    rec = ReplayRecord(
        phase="extractor",
        turn_index=2,
        root_session_id="abc",
        ts_ns=42,
        compose_kwargs={},
        payload={"new_turns": []},
        provider=None,
        output={"events": []},
        status="ok",
        raw_assistant_messages=blocks,
    )
    write_record(path, rec)
    [back] = list(iter_records(path))
    assert back.raw_assistant_messages == blocks


def test_replay_record_omits_raw_assistant_messages_when_empty(tmp_path: Path) -> None:
    """Empty list ⇒ key absent on disk so old sidecars stay byte-identical."""
    path = tmp_path / "replay.jsonl"
    write_record(path, _make_record("auditor", turn=1))
    raw = path.read_text(encoding="utf-8")
    assert "raw_assistant_messages" not in raw


def test_replay_record_back_compat_with_old_sidecar(tmp_path: Path) -> None:
    """A pre-existing sidecar without raw_assistant_messages must load as []."""
    path = tmp_path / "replay.jsonl"
    path.write_text(
        '{"phase":"extractor","turn_index":0,"root_session_id":"old",'
        '"ts_ns":0,"compose_kwargs":{},"payload":{},"provider":null,'
        '"output":null,"status":"ok","error":null,"latency_ms":0}\n',
        encoding="utf-8",
    )
    [rec] = list(iter_records(path))
    assert rec.raw_assistant_messages == []


def test_to_dict_from_dict_round_trip() -> None:
    """``to_dict`` and ``from_dict`` are an exact pair.

    The rl-prompts CLI emits a stripped ``to_dict`` view and the trainer
    re-hydrates it via ``from_dict``; if these two ever drift, the public
    contract with rca-autorl breaks silently.
    """
    original = ReplayRecord(
        phase="extractor",
        turn_index=3,
        root_session_id="sess-roundtrip",
        ts_ns=987,
        compose_kwargs={"base_prompt": "x", "summary_threshold": 30},
        payload={"new_turns": [{"id": "t1"}], "next_event_id": 11},
        provider=["mod.fake", {"k": "v"}],
        output={"events": []},
        status="ok",
        error=None,
        latency_ms=15,
        extras={"turn_texts": {"1": "hi"}},
        raw_assistant_messages=[
            {"type": "tool_call", "id": "c-1", "name": "finalize_extraction", "arguments": {}}
        ],
    )
    back = ReplayRecord.from_dict(original.to_dict())
    assert back == original


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
