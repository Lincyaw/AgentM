"""Fail-stop tests for the single-event-log merge.

Three positions documented in ``.claude/designs/single-event-log.md``:

1. ``continue_recent`` must rebuild the same in-memory trajectory from a
   merged JSONL containing interleaved ``message.appended`` rows and other
   OTel-shaped event spans. Wrong → resumes silently truncate or drop
   conversation history.
2. The observability sink's bounded queue must **block** (not drop) on
   overflow. Wrong → silent loss of ``message.appended`` rows = lost
   trajectory entries (the spec explicitly forbids the old "drop on full"
   policy now that this log is authoritative).
3. ``atexit`` shutdown must drain the queue before process exit. Wrong →
   tail loss on every CLI invocation.

These tests don't go through the full ``AgentSession`` — they exercise the
two seams (SessionManager._load, _Sink) directly because that's where the
contract lives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentm.core.abi import text_message
from agentm.core.abi.session import (
    CURRENT_SESSION_VERSION,
    SessionHeader,
    message_entry,
)
from agentm.core.runtime.session_manager import SessionManager


def _wrap_header(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "session/header/v0",
        "kind": "session.header",
        "name": "session.header",
        "record": record,
    }


def _wrap_message(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "session/message/v0",
        "kind": "message.appended",
        "name": "message.appended",
        "record": record,
    }


def _wrap_other_span() -> dict[str, Any]:
    """A plausible OTel-shaped event row that ``_load`` must ignore."""

    return {
        "schema": "otel/span/v0",
        "kind": "turn.summary",
        "trace_id": "abc",
        "span_id": "1234",
        "name": "turn:0",
        "attributes": {"turn_index": 0, "tool_calls": []},
        "status": {"code": "OK"},
    }


def test_continue_recent_reads_interleaved_merged_log(tmp_path: Path) -> None:
    """The trajectory rebuilt from a merged log must equal the trajectory
    that would have been built from the legacy split files.

    We hand-craft a file containing the two SessionManager-relevant rows
    interleaved with unrelated OTel spans; ``SessionManager.open`` must
    pick out the header + messages in order and ignore the rest.
    """

    cwd = tmp_path
    obs_dir = cwd / ".agentm" / "observability"
    obs_dir.mkdir(parents=True)
    sid = "abcd1234"
    log = obs_dir / f"{sid}.jsonl"

    header = SessionHeader(
        type="session",
        version=CURRENT_SESSION_VERSION,
        id=sid,
        timestamp=1.0,
        cwd=str(cwd),
        parent_session=None,
    )
    e1 = message_entry(text_message("hello"), parent_id=None)
    e2 = message_entry(text_message("world"), parent_id=e1.id)

    # Build the entry dict via the same serializer the runtime uses so
    # the round-trip really matches production shape.
    from agentm.core.runtime.session_manager import _entry_to_record, _header_to_record

    rows = [
        _wrap_other_span(),
        _wrap_header(_header_to_record(header)),
        _wrap_other_span(),
        _wrap_message(_entry_to_record(e1)),
        {"schema": "otel/span/v0", "kind": "event.dispatch", "name": "x"},
        _wrap_message(_entry_to_record(e2)),
        _wrap_other_span(),
    ]
    with log.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    mgr = SessionManager.continue_recent(str(cwd))
    assert mgr.get_session_id() == sid
    assert mgr.get_cwd() == str(cwd)

    entries = mgr.get_entries()
    assert [e.id for e in entries] == [e1.id, e2.id]
    assert mgr.get_leaf_id() == e2.id

    # Compare against an equivalent in-memory build (the trajectory the
    # legacy split-file path would have produced from the same emit
    # sequence).
    expected = SessionManager.in_memory(cwd=str(cwd))
    expected.new_session(id=sid)
    expected.append(e1)
    expected.append(e2)
    assert [e.id for e in mgr.get_entries()] == [e.id for e in expected.get_entries()]
    assert mgr.get_session_id() == expected.get_session_id()


def test_observability_sink_blocks_on_overflow(tmp_path: Path) -> None:
    """A burst larger than the queue must NOT drop rows — the sink blocks
    until the writer thread drains, and every record lands on disk.

    Spec: ``Replace the current "bounded queue, drop on full" with
    "bounded queue ... block on full"``. We pick a tiny queue (16 slots)
    and a tiny batch so the test takes milliseconds, then push 200 records
    — far more than the queue can hold — and assert all 200 are on disk
    after close.
    """

    from agentm.extensions.builtin.observability import _Sink

    log = tmp_path / "trace.jsonl"
    sink = _Sink(log, max_queue=16, batch_size=4)
    try:
        N = 200
        for i in range(N):
            sink.write({"i": i})
    finally:
        sink.close()

    lines = [
        line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == N
    seen = sorted(json.loads(line)["i"] for line in lines)
    assert seen == list(range(N))


def test_observability_sink_drains_on_close(tmp_path: Path) -> None:
    """``close()`` is the ``atexit`` drain path — anything still in the
    queue must reach disk before the writer thread exits.

    We push one record, immediately ``close``, and assert the row is on
    disk. The single-record case is the load-bearing one: it's what the
    atexit hook protects against (tail loss of the final emit).
    """

    from agentm.extensions.builtin.observability import _Sink

    log = tmp_path / "trace.jsonl"
    sink = _Sink(log, max_queue=100, batch_size=8)
    sink.write({"tail": True})
    sink.close()

    lines = [
        line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"tail": True}
