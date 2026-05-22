"""Lock the ``llmharness dataset`` reconstruction contract.

Disaster guarded: a regression that mis-pairs extractor inputs with their
outputs would silently produce wrong training data. We synthesize a tiny
session JSONL by hand (no real run required) so the test pins down the
windowing rules:

  * extractor input new_turns = messages[prev_cursor.last_turn_index+1 ..
    this_cursor.last_turn_index]; recent_graph = audit_events appended
    BEFORE this firing's events.
  * auditor input graph = audit_events accumulator at firing time;
    recent_verdicts = trailing window of prior verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmharness.cli import main as cli_main
from llmharness.distill.export import (
    extractor_records_from_replay,
    legacy_batch_replay_count,
)


def _write_session(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in records:
            h.write(json.dumps(r))
            h.write("\n")


def _msg(role: str, text: str) -> dict:
    return {
        "type": "message",
        "id": f"m-{role}-{text}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "role": role,
            "content": [{"type": "text", "text": text}],
            "timestamp": 0.0,
        },
    }


def _audit_event(eid: int, kind: str, summary: str) -> dict:
    return {
        "type": "llmharness.audit_event",
        "id": f"e-{eid}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "id": eid,
            "kind": kind,
            "summary": summary,
            "refs": [],
            "source_turns": [],
        },
    }


def _cursor(last_turn_index: int) -> dict:
    return {
        "type": "llmharness.extractor_cursor",
        "id": f"c-{last_turn_index}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "last_turn_index": last_turn_index,
            "extraction_run_id": f"run-{last_turn_index}",
        },
    }


def _verdict(surface_reminder: bool, reminder_text: str = "") -> dict:
    return {
        "type": "llmharness.verdict",
        "id": f"v-{surface_reminder}-{reminder_text}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "surface_reminder": surface_reminder,
            "reminder_text": reminder_text,
            "continuation_notes": [],
            "matched_event_ids": [1] if surface_reminder else [],
            "cited_cards": [],
        },
    }


def test_dataset_export_handles_messages_appended_at_end(tmp_path: Path) -> None:
    """Real ``AgentSession.prompt`` flushes messages to SessionManager AFTER
    the loop returns, while audit entries are appended inline. The JSONL
    file therefore lists every audit entry before any of the run's
    messages. ``_walk_dataset`` must reconstruct windows correctly under
    that order, not only under the synthetic interleaved order.

    Disaster guarded: a regression that re-routes the export to a single
    forward pass would produce empty ``new_turns`` for every case except
    the first (silent dataset corruption — output looks structurally fine
    but the inputs are missing).
    """
    session_path = tmp_path / "session.jsonl"
    _write_session(
        session_path,
        [
            {"type": "session", "id": "h", "parent_id": None, "timestamp": 0, "payload": {}},
            _msg("user", "do X"),  # only one message persisted before the run
            _audit_event(0, "task", "User: do X"),
            _cursor(1),
            _audit_event(1, "act", "Assistant did X"),
            _audit_event(2, "evid", "tool: result"),
            _cursor(3),
            _verdict(surface_reminder=True, reminder_text="loop detected"),
            # Messages 1-3 land here, after every audit entry — this mirrors
            # AgentSession.prompt's batch flush at end-of-loop.
            _msg("assistant", "ok"),
            _msg("user", "follow up"),
            _msg("assistant", "done"),
        ],
    )
    out_dir = tmp_path / "out"
    rc = cli_main(["dataset", str(session_path), "--out", str(out_dir)])
    assert rc == 0

    extractor_lines = (out_dir / "extractor.jsonl").read_text().splitlines()
    e0, e1 = (json.loads(line) for line in extractor_lines)
    assert [m["index"] for m in e0["input"]["new_turns"]] == [0, 1]
    assert [m["index"] for m in e1["input"]["new_turns"]] == [2, 3]


def test_dataset_export_pairs_inputs_with_outputs(tmp_path: Path) -> None:
    """Two extractor firings + one auditor firing; check windowing.

    Trajectory:
      msg 0 user, msg 1 assistant   -> extractor fires (cursor at idx 1),
                                       emits 1 event (id 0)
      msg 2 user, msg 3 assistant   -> extractor fires (cursor at idx 3),
                                       emits 2 events (ids 1, 2);
                                       auditor fires after, drift=True
    """
    session_path = tmp_path / "session.jsonl"
    _write_session(
        session_path,
        [
            {"type": "session", "id": "h", "parent_id": None, "timestamp": 0, "payload": {}},
            _msg("user", "do X"),
            _msg("assistant", "ok"),
            _audit_event(0, "task", "User: do X"),
            _cursor(1),
            _msg("user", "follow up"),
            _msg("assistant", "done"),
            _audit_event(1, "act", "Assistant did X"),
            _audit_event(2, "evid", "tool: result"),
            _cursor(3),
            _verdict(surface_reminder=True, reminder_text="loop detected"),
        ],
    )

    out_dir = tmp_path / "out"
    rc = cli_main(["dataset", str(session_path), "--out", str(out_dir)])
    assert rc == 0

    extractor_lines = (out_dir / "extractor.jsonl").read_text().splitlines()
    auditor_lines = (out_dir / "auditor.jsonl").read_text().splitlines()
    assert len(extractor_lines) == 2
    assert len(auditor_lines) == 1

    e0, e1 = (json.loads(line) for line in extractor_lines)

    # Firing 0: window [0, 1], no prior graph, output event id 0.
    assert e0["meta"]["turn_window"] == [0, 1]
    assert [m["index"] for m in e0["input"]["new_turns"]] == [0, 1]
    assert e0["input"]["recent_graph"] == []
    assert [ev["id"] for ev in e0["output"]["events"]] == [0]

    # Firing 1: window [2, 3], prior graph carries event 0, outputs 1 & 2.
    assert e1["meta"]["turn_window"] == [2, 3]
    assert [m["index"] for m in e1["input"]["new_turns"]] == [2, 3]
    assert [ev["id"] for ev in e1["input"]["recent_graph"]] == [0]
    assert [ev["id"] for ev in e1["output"]["events"]] == [1, 2]

    # Auditor fires after both extractors → graph has all 3 events.
    a0 = json.loads(auditor_lines[0])
    assert [ev["id"] for ev in a0["input"]["graph"]] == [0, 1, 2]
    assert a0["input"]["recent_verdicts"] == []
    assert a0["output"]["verdict"]["surface_reminder"] is True
    assert a0["output"]["verdict"]["reminder_text"] == "loop detected"


# --- SFT distill export (extractor_records_from_replay) ----------------------


def _ok_extractor_replay_record(
    *,
    raw_assistant_messages: list[dict] | None = None,
) -> dict:
    """One ok extractor replay-record dict; thinking is opt-in per test."""
    rec: dict = {
        "phase": "extractor",
        "status": "ok",
        "root_session_id": "sess-x",
        "turn_index": 4,
        "ts_ns": 1,
        "payload": {"new_turns": [], "recent_graph": []},
        "output": {
            "events": [
                {"id": 1, "kind": "task", "summary": "go", "source_turns": [0]}
            ],
            "edges": [],
            "dropped_edges": [],
        },
    }
    if raw_assistant_messages is not None:
        rec["raw_assistant_messages"] = raw_assistant_messages
    return rec


def test_dataset_export_emits_multi_turn_extractor_trajectory() -> None:
    """Under v19 the SFT target is a multi-turn assistant trajectory.

    Each recorded tool_call becomes its own assistant message and the
    preceding thinking block(s) get attached as ``<think>`` content. A
    student trained on this learns to drive an incremental graph build,
    not to dump one batch.

    Disaster guarded: a regression that collapses back to one terminal
    tool_call (the v18 shape) would silently retrain the student on a
    deleted tool surface.
    """
    rec = _ok_extractor_replay_record(
        raw_assistant_messages=[
            {"type": "thinking", "text": "scan window"},
            {
                "type": "tool_call",
                "name": "upsert_node",
                "arguments": {"id": 1, "kind": "act"},
            },
            {"type": "thinking", "text": "emit the evid"},
            {
                "type": "tool_call",
                "name": "upsert_node",
                "arguments": {"id": 2, "kind": "evid"},
            },
            {
                "type": "tool_call",
                "name": "finalize_extraction",
                "arguments": {},
            },
        ]
    )
    [sft] = list(extractor_records_from_replay([rec], sample_id="s-1"))
    out = json.loads(sft.to_jsonl())
    messages = out["target"]["messages"]
    assert [m["tool_calls"][0]["function"]["name"] for m in messages] == [
        "upsert_node",
        "upsert_node",
        "finalize_extraction",
    ]
    assert messages[0]["content"] == "<think>scan window</think>\n\n"
    assert messages[1]["content"] == "<think>emit the evid</think>\n\n"
    # finalize had no preceding thinking — empty content (not stray tags).
    assert messages[2]["content"] == ""


def test_distill_cli_export_preserves_thinking_through_sidecar(tmp_path: Path) -> None:
    """End-to-end: replay sidecar with raw_assistant_messages → distill CLI
    → SFT row content carries the ``<think>`` wrapping.

    Disaster guarded: the unit tests above feed dicts directly into
    ``extractor_records_from_replay``, so a regression in
    ``distill.cli._replay_record_dicts`` (e.g. forgetting to forward
    ``raw_assistant_messages`` when converting ReplayRecord → dict) goes
    unnoticed. This test runs the actual CLI ``export`` subcommand
    against a synthesized sidecar so the conversion layer is exercised.
    """
    from llmharness.distill.cli import main as distill_main

    sid = "sess-thinking-regression"
    replay_dir = tmp_path / ".agentm" / "audit_replay"
    replay_dir.mkdir(parents=True)
    sidecar = replay_dir / f"{sid}.jsonl"
    record = {
        "phase": "extractor",
        "turn_index": 4,
        "root_session_id": sid,
        "ts_ns": 1,
        "compose_kwargs": {},
        "payload": {"new_turns": [], "recent_graph": []},
        "provider": None,
        "output": {
            "events": [
                {"id": 1, "kind": "task", "summary": "go", "source_turns": [0]}
            ],
            "edges": [],
            "dropped_edges": [],
        },
        "status": "ok",
        "error": None,
        "latency_ms": 0,
        "raw_assistant_messages": [
            {"type": "thinking", "text": "reason about the turn"},
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "finalize_extraction",
                "arguments": {},
            },
        ],
    }
    sidecar.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (replay_dir / f"{sid}.meta.json").write_text(
        json.dumps(
            {
                "sample_id": "s-regression",
                "dataset_name": "test",
                "dataset_path": "",
                "root_session_id": sid,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "sft"
    rc = distill_main(
        [
            "export",
            "--labels",
            str(tmp_path / "labels-empty"),
            "--replay-dir",
            str(replay_dir),
            "--out",
            str(out_dir),
            "--phase",
            "extractor",
        ]
    )
    assert rc == 0
    extractor_jsonl = out_dir / "extractor.jsonl"
    rows = [json.loads(line) for line in extractor_jsonl.read_text().splitlines() if line]
    assert len(rows) == 1
    content = rows[0]["target"]["messages"][0]["content"]
    assert content == "<think>reason about the turn</think>\n\n"


def test_dataset_export_skips_legacy_batch_replays() -> None:
    """Legacy v18 replays use ``submit_events_batch``; v19 cannot train on them.

    Disaster guarded: training on the old terminal-batch shape under
    the new tool surface would teach the student a deleted contract.
    The export must skip the record AND a separate counter must report
    how many such records were skipped so the operator notices.
    """
    legacy_rec = _ok_extractor_replay_record(
        raw_assistant_messages=[
            {"type": "thinking", "text": "v18 single shot"},
            {
                "type": "tool_call",
                "name": "submit_events_batch",
                "arguments": {"events": [], "done": True},
            },
        ]
    )
    assert list(extractor_records_from_replay([legacy_rec], sample_id="s-1")) == []
    assert legacy_batch_replay_count([legacy_rec]) == 1


def test_dataset_export_skips_record_with_no_recorded_tool_calls() -> None:
    """Empty raw_assistant_messages (spawn_error / older sidecars).

    Disaster guarded: there is nothing for the student to learn from
    a target trajectory with zero tool calls; emitting a row with an
    empty messages list would silently degrade training.
    """
    rec_empty = _ok_extractor_replay_record(raw_assistant_messages=[])
    assert list(extractor_records_from_replay([rec_empty], sample_id="s-1")) == []
    rec_missing = _ok_extractor_replay_record()
    assert list(extractor_records_from_replay([rec_missing], sample_id="s-1")) == []
