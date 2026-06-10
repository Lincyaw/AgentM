"""Integration test for ``llmharness-replay agent-from-reminder``.

Fail-stop position: the BRANCH LENGTH (count of entries ≤ turn t) on
the new session's active branch MUST equal what was on the source
branch up to that point. If this drifts, the student model sees a
different prefix at inference (the branched-session replay) than during
training (the original live session) — train/inference parity is the
whole reason prefix-replay exists.

The test synthesises a small persisted session + matching audit_replay
sidecar, invokes the CLI command in ``--print-only`` mode, and asserts
the new session file was written, the branched branch length matches,
and the printed command carries the right ``--resume`` id + the right
reminder text.
"""

from __future__ import annotations

import time
from pathlib import Path

from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    text_message,
)
from agentm.core.abi.session import ENTRY_TYPE_MESSAGE, message_entry
from agentm.core.runtime.session_manager import SessionManager
from typer.testing import CliRunner

from llmharness import entry_types as et
from llmharness.replay.cli import app
from llmharness.replay.record import ReplayRecord, write_record


def _make_assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=time.time(),
        stop_reason=None,
        usage=None,
    )


def _make_source_session(
    *, session_dir: Path, cwd: str, n_turns: int
) -> tuple[SessionManager, Path]:
    """Build a persisted session with 2*n_turns + 1 message entries.

    Layout: user, assistant, user, assistant, ..., assistant.
    The k-th assistant message corresponds to turn_index = 2k+1 in the
    ``len(messages) - 1`` convention.
    """
    mgr = SessionManager.create(cwd, session_dir)
    mgr.append(message_entry(text_message("start"), parent_id=None))
    for i in range(n_turns):
        mgr.append(message_entry(_make_assistant_message(f"reply {i}"), mgr.get_leaf_id()))
        # Sprinkle an audit_graph_op entry between turns so we exercise
        # the "non-message entries on the branch get copied through"
        # path — the brief is explicit that filtering them out by hand
        # is wrong.
        mgr.append_custom_entry(
            et.AUDIT_GRAPH_OP,
            {
                "op": "node_upsert",
                "id": i + 1,
                "kind": "act",
                "summary": f"step {i + 1}",
                "source_turns": [i],
            },
        )
        if i < n_turns - 1:
            mgr.append(message_entry(text_message(f"user {i + 1}"), mgr.get_leaf_id()))
    session_file = mgr.session_file
    assert session_file is not None
    return mgr, session_file


def _write_audit_record(*, sidecar_path: Path, session_id: str, turn: int, reminder: str) -> None:
    rec = ReplayRecord(
        phase="auditor",
        turn_index=turn,
        session_id=session_id,
        trace_id=f"trace-{session_id}",
        ts_ns=int(time.time_ns()),
        compose_kwargs={},
        payload={},
        provider=None,
        output={
            "surface_reminder": True,
            "reminder_text": reminder,
            "continuation_notes": [],
            "matched_event_ids": [],
        },
        status="ok",
    )
    write_record(sidecar_path, rec)


def test_agent_from_reminder_branches_at_turn_and_prints_command(
    tmp_path: Path,
) -> None:
    cwd = str(tmp_path)
    session_dir = tmp_path / ".agentm" / "sessions"
    audit_dir = tmp_path / ".agentm" / "audit_replay"
    audit_dir.mkdir(parents=True)

    source_mgr, source_file = _make_source_session(session_dir=session_dir, cwd=cwd, n_turns=5)
    root_id = source_mgr.get_session_id()

    # turn=3 means the auditor record was emitted at trajectory index 3
    # (the second assistant message in the session — message indices 0,
    # 1, 2, 3 = user, assistant, user, assistant).
    target_turn = 3
    reminder_text = "please re-check assumption A in light of evidence E"
    sidecar = audit_dir / f"{root_id}.jsonl"
    _write_audit_record(
        sidecar_path=sidecar,
        session_id=root_id,
        turn=target_turn,
        reminder=reminder_text,
    )

    # Expected branch length on the new session: every entry up to and
    # including the (target_turn)-th message entry on the source branch.
    branch = source_mgr.get_active_branch()
    message_positions: list[int] = []
    for i, entry in enumerate(branch):
        if entry.type == ENTRY_TYPE_MESSAGE:
            message_positions.append(i)
    expected_prefix_len = message_positions[target_turn] + 1
    # Capture the *content* of the expected leaf entry; ``create_branched_session``
    # rewrites entry ids on the way out, so a leaf-id equality assert would
    # always fail. The leaf must instead carry the same (type, payload) as
    # the source-branch entry that ends turn t.
    expected_leaf_entry = branch[message_positions[target_turn]]

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "agent-from-reminder",
            "--audit-replay",
            str(sidecar),
            "--turn",
            str(target_turn),
            "--session-dir",
            str(session_dir),
            "--print-only",
        ],
    )
    assert result.exit_code == 0, result.stdout

    # New session file must exist under the session dir.
    new_files = sorted(p for p in session_dir.glob("*.jsonl") if p != source_file)
    assert len(new_files) == 1, f"expected one branched session file; got {new_files}"
    branched_file = new_files[0]

    branched_mgr = SessionManager.open(branched_file)
    new_branch = branched_mgr.get_active_branch()
    assert len(new_branch) == expected_prefix_len, (
        f"branched branch length {len(new_branch)} != expected "
        f"{expected_prefix_len} — prefix drift would break train/inference parity"
    )
    # Leaf identity — by content, not by id (create_branched_session
    # rewrites ids). A count-only drift (off-by-one on a non-message
    # entry, e.g. the audit_event between turns) would slip past the
    # length check but be caught here: the leaf payload must match the
    # source-side entry that ends turn t.
    branched_leaf = branched_mgr.get_leaf_entry()
    assert branched_leaf is not None
    assert branched_mgr.get_leaf_id() == new_branch[-1].id
    assert branched_leaf.type == expected_leaf_entry.type
    assert branched_leaf.payload == expected_leaf_entry.payload

    # Header must record parent_session linkage.
    header = branched_mgr.get_header()
    assert header is not None
    assert header.parent_session == str(source_file)

    # Printed command must wire the new session id + the reminder text.
    new_sid = branched_mgr.get_session_id()
    assert f"--resume {new_sid}" in result.stdout or new_sid in result.stdout
    assert reminder_text in result.stdout
    assert "llmharness.replay.reminder_seed" in result.stdout
    assert "llmharness.atom" in result.stdout
    assert "enable_reminders" in result.stdout
