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


def _verdict(drift: bool, reminder: str = "") -> dict:
    return {
        "type": "llmharness.verdict",
        "id": f"v-{drift}-{reminder}",
        "parent_id": None,
        "timestamp": 0.0,
        "payload": {
            "drift": drift,
            "type": "stuck_loop" if drift else None,
            "reminder": reminder,
            "matched_event_ids": [],
            "cited_cards": [],
            "downstream_reaction": None,
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
            _audit_event(1, "action", "Assistant did X"),
            _audit_event(2, "evidence", "tool: result"),
            _cursor(3),
            _verdict(drift=True, reminder="loop detected"),
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
            _audit_event(1, "action", "Assistant did X"),
            _audit_event(2, "evidence", "tool: result"),
            _cursor(3),
            _verdict(drift=True, reminder="loop detected"),
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
    assert a0["output"]["verdict"]["drift"] is True
    assert a0["output"]["verdict"]["reminder"] == "loop detected"
