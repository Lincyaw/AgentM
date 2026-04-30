"""Tests for ``agentm.harness.session_manager``."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.core.kernel import text_message

from agentm.harness.session_manager import (
    InMemorySessionManager,
    JsonlSessionManager,
    message_entry,
)


def test_inmemory_append_and_get_messages_in_order() -> None:
    """Appended messages appear in the active branch in insertion order."""

    sm = InMemorySessionManager()
    msg_a = text_message("a", timestamp=0.0)
    msg_b = text_message("b", timestamp=0.0)
    e_a = message_entry(msg_a, parent_id=None)
    sm.append(e_a)
    e_b = message_entry(msg_b, parent_id=e_a.id)
    sm.append(e_b)

    msgs = sm.get_messages()
    assert msgs == [msg_a, msg_b]
    branch = sm.get_active_branch()
    assert [e.id for e in branch] == [e_a.id, e_b.id]


def test_inmemory_fork_diverges_from_parent() -> None:
    """Forking at an entry produces a manager whose appends create a sibling
    branch; the parent manager's view is unaffected."""

    sm = InMemorySessionManager()
    root = message_entry(text_message("root"), parent_id=None)
    sm.append(root)
    child_a = message_entry(text_message("a"), parent_id=root.id)
    sm.append(child_a)

    fork = sm.fork_at(root.id)
    child_b = message_entry(text_message("b"), parent_id=root.id)
    fork.append(child_b)

    # Parent still on its branch (root → a).
    assert [e.id for e in sm.get_active_branch()] == [root.id, child_a.id]
    # Fork is on the new branch (root → b).
    assert [e.id for e in fork.get_active_branch()] == [root.id, child_b.id]


def test_inmemory_navigate_to_changes_active_branch() -> None:
    """Navigating to a different leaf updates ``get_active_branch`` output."""

    sm = InMemorySessionManager()
    root = message_entry(text_message("root"), parent_id=None)
    sm.append(root)
    a = message_entry(text_message("a"), parent_id=root.id)
    sm.append(a)
    # Manually create a sibling entry by setting parent_id to root.
    b = message_entry(text_message("b"), parent_id=root.id)
    sm.append(b)

    # After appending b, b is the active leaf.
    assert sm.get_active_branch()[-1].id == b.id

    sm.navigate_to(a.id)
    assert [e.id for e in sm.get_active_branch()] == [root.id, a.id]


def test_inmemory_find_returns_entry_or_none() -> None:
    sm = InMemorySessionManager()
    root = message_entry(text_message("root"), parent_id=None)
    sm.append(root)

    assert sm.find(root.id) is root
    assert sm.find("does-not-exist") is None


def test_jsonl_durable_round_trip(tmp_path: Path) -> None:
    """Entries written by one manager are readable by a fresh instance.

    Payload reconstruction is best-effort (becomes a dict) — assert metadata
    equality only, per the module-docstring limitation note.
    """

    path = tmp_path / "session.jsonl"

    sm1 = JsonlSessionManager(path)
    e1 = message_entry(text_message("hello"), parent_id=None)
    sm1.append(e1)
    e2 = message_entry(text_message("world"), parent_id=e1.id)
    sm1.append(e2)

    # File is on disk and contains both entries.
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    # Reopen.
    sm2 = JsonlSessionManager(path)
    branch = sm2.get_active_branch()
    assert [e.id for e in branch] == [e1.id, e2.id]
    assert [e.type for e in branch] == ["message", "message"]
    assert [e.parent_id for e in branch] == [None, e1.id]
    # Payload is dict-shaped after reload (Phase 1 limitation).
    for e in branch:
        assert isinstance(e.payload, dict)


def test_jsonl_appends_are_durable_after_reopen(tmp_path: Path) -> None:
    """Appending after reopening preserves earlier entries."""

    path = tmp_path / "s.jsonl"
    sm1 = JsonlSessionManager(path)
    e1 = message_entry(text_message("first"), parent_id=None)
    sm1.append(e1)

    sm2 = JsonlSessionManager(path)
    e2 = message_entry(text_message("second"), parent_id=e1.id)
    sm2.append(e2)

    # Third instance sees both.
    sm3 = JsonlSessionManager(path)
    assert [e.id for e in sm3.get_active_branch()] == [e1.id, e2.id]


def test_inmemory_rejects_dangling_parent_id() -> None:
    """Appending an entry whose parent_id doesn't exist surfaces as an
    explicit ValueError rather than silently producing an orphan."""

    sm = InMemorySessionManager()
    orphan = message_entry(text_message("orphan"), parent_id="nonexistent")
    with pytest.raises(ValueError):
        sm.append(orphan)
