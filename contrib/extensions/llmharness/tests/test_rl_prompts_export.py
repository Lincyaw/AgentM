"""Lock the rl_prompts.jsonl schema for the rca-autorl RL trainer.

Drift here means the downstream sampler reads a stale shape — locks the
row contract at the CLI boundary so refactors of the prompt-extraction
helper can't silently change the on-disk format.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmharness.distill.cli import main as cli_main
from llmharness.replay.record import ReplayRecord, write_record


def _make_extractor_record(
    *, tmp_path: Path, root_session_id: str, turn_index: int
) -> Path:
    log = tmp_path / f"{root_session_id}.jsonl"
    rec = ReplayRecord(
        phase="extractor",
        turn_index=turn_index,
        root_session_id=root_session_id,
        ts_ns=123_000_000_000 + turn_index,
        compose_kwargs={"window": 3},
        payload={"trigger": "tick", "new_turns": [{"id": "t1"}]},
        provider=["mod.fake", {}],
        output={"events": []},
        status="ok",
        raw_assistant_messages=[
            {"type": "tool_call", "name": "finalize_extraction", "arguments": {}},
        ],
    )
    write_record(log, rec)
    return log


def _make_auditor_record(
    *, tmp_path: Path, root_session_id: str, turn_index: int
) -> None:
    log = tmp_path / f"{root_session_id}.jsonl"
    rec = ReplayRecord(
        phase="auditor",
        turn_index=turn_index,
        root_session_id=root_session_id,
        ts_ns=456_000_000_000 + turn_index,
        compose_kwargs={},
        payload={"graph": {"nodes": []}, "recent_verdicts": []},
        provider=None,
        output={"verdict": {"surface_reminder": False}},
        status="ok",
    )
    write_record(log, rec)


def test_rl_prompts_cli_emits_schema_and_no_target_leak(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    out = tmp_path / "out" / "rl_prompts.jsonl"

    # Two sessions, both phases — should yield 2 extractor + 1 auditor.
    _make_extractor_record(tmp_path=replay_dir, root_session_id="sess-a", turn_index=2)
    _make_extractor_record(tmp_path=replay_dir, root_session_id="sess-b", turn_index=5)
    _make_auditor_record(tmp_path=replay_dir, root_session_id="sess-b", turn_index=6)

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
        assert set(row.keys()) == {
            "phase",
            "sample_id",
            "source_case_id",
            "firing_index",
            "input",
            "meta",
        }
        # No target / output / raw_assistant_messages leak into prompt pool.
        assert "target" not in row
        assert "output" not in row
        assert "raw_assistant_messages" not in row
        # Input shape.
        assert set(row["input"].keys()) == {"system", "user"}
        assert isinstance(row["input"]["system"], str) and row["input"]["system"]
        assert isinstance(row["input"]["user"], str) and row["input"]["user"]
        # Meta carries provenance.
        assert "root_session_id" in row["meta"]
        assert "turn_index" in row["meta"]
        assert "ts_ns" in row["meta"]
        # sample_id encodes case + firing + phase.
        assert row["sample_id"].endswith(f":firing-{row['firing_index']}:{row['phase']}")


def test_rl_prompts_cli_phase_filter(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    out = tmp_path / "rl_prompts.jsonl"

    _make_extractor_record(tmp_path=replay_dir, root_session_id="sess-a", turn_index=1)
    _make_auditor_record(tmp_path=replay_dir, root_session_id="sess-a", turn_index=2)

    rc = cli_main(
        [
            "rl-prompts",
            "--replay-dir",
            str(replay_dir),
            "--out",
            str(out),
            "--phase",
            "auditor",
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["phase"] == "auditor"
