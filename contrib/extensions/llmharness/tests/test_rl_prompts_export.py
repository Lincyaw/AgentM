"""Lock the rl-prompts JSONL schema for the rca-autorl online-RL trainer.

The CLI emits one stripped-:class:`ReplayRecord` per row: every field
preserved except the teacher-output fields (output / status / error /
latency_ms / raw_assistant_messages). The row round-trips through
``ReplayRecord.from_dict`` so the trainer can hand it directly to
``replay_extractor_record`` / ``replay_auditor_record`` for a fresh
rollout. Drift here breaks the public contract — pin it.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmharness.distill.cli import main as cli_main
from llmharness.distill.rl_prompts import STRIPPED_FIELDS, hydrate_row
from llmharness.replay.record import ReplayRecord, write_record


def _make_extractor_record(
    *, tmp_path: Path, session_id: str, turn_index: int
) -> Path:
    log = tmp_path / f"{session_id}.jsonl"
    rec = ReplayRecord(
        phase="extractor",
        turn_index=turn_index,
        session_id=session_id,
        trace_id=f"trace-{session_id}",
        ts_ns=123_000_000_000 + turn_index,
        compose_kwargs={"base_prompt": "BE THE EXTRACTOR"},
        payload={"trigger": "tick", "new_turns": [{"id": "t1"}], "next_event_id": 7},
        provider=["mod.fake", {"api_key": "x"}],
        output={"events": [{"id": 7}]},
        status="ok",
        latency_ms=42,
        extras={"turn_texts": {"1": "hello"}},
        raw_assistant_messages=[
            {"type": "tool_call", "name": "finalize_extraction", "arguments": {}},
        ],
    )
    write_record(log, rec)
    return log


def _make_auditor_record(
    *, tmp_path: Path, session_id: str, turn_index: int
) -> None:
    log = tmp_path / f"{session_id}.jsonl"
    rec = ReplayRecord(
        phase="auditor",
        turn_index=turn_index,
        session_id=session_id,
        trace_id=f"trace-{session_id}",
        ts_ns=456_000_000_000 + turn_index,
        compose_kwargs={"summary_threshold": 30},
        payload={"graph": {"nodes": []}, "recent_verdicts": []},
        provider=None,
        output={"verdict": {"surface_reminder": False}},
        status="ok",
    )
    write_record(log, rec)


def test_rl_prompts_cli_emits_stripped_replay_records(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    out = tmp_path / "out" / "rl_prompts.jsonl"

    _make_extractor_record(tmp_path=replay_dir, session_id="sess-a", turn_index=2)
    _make_extractor_record(tmp_path=replay_dir, session_id="sess-b", turn_index=5)
    _make_auditor_record(tmp_path=replay_dir, session_id="sess-b", turn_index=6)

    rc = cli_main(
        [
            "rl-prompts",
            "--replay-dir",
            str(replay_dir),
            "--out",
            str(out),
            "--phase",
            "both",
        ]
    )
    assert rc == 0
    assert out.is_file()

    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 3
    phases = sorted(r["phase"] for r in rows)
    assert phases == ["auditor", "extractor", "extractor"]

    for row in rows:
        # Stripped fields must be absent.
        for forbidden in STRIPPED_FIELDS:
            assert forbidden not in row, forbidden

        # Preserved fields must be present (the substrate the runner needs).
        assert "phase" in row
        assert "turn_index" in row
        assert "session_id" in row
        assert "trace_id" in row
        assert "root_session_id" not in row
        assert "ts_ns" in row
        assert "compose_kwargs" in row
        assert "payload" in row
        assert "provider" in row
        assert "extras" in row

        # Round-trip: the row must hydrate into a usable ReplayRecord with
        # the teacher-output fields cleared to neutral values.
        rec = hydrate_row(row)
        assert isinstance(rec, ReplayRecord)
        assert rec.output is None
        assert rec.status == "ok"
        assert rec.error is None
        assert rec.latency_ms == 0
        assert rec.raw_assistant_messages == []
        # The compose_kwargs / payload survive verbatim — the trainer needs
        # both to feed replay_*_record.
        assert rec.payload == row["payload"]
        assert rec.compose_kwargs == row["compose_kwargs"]


