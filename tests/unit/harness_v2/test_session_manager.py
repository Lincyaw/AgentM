"""Tests for ``agentm.harness.session_manager``."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.core.kernel import TextContent, text_message

from agentm.harness.session_manager import (
    InMemorySessionManager,
    JsonlSessionManager,
    SessionManager,
    message_entry,
)


def test_inmemory_append_and_get_messages_in_order() -> None:
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
    sm = InMemorySessionManager()
    root = message_entry(text_message("root"), parent_id=None)
    sm.append(root)
    child_a = message_entry(text_message("a"), parent_id=root.id)
    sm.append(child_a)

    fork = sm.fork_at(root.id)
    child_b = message_entry(text_message("b"), parent_id=root.id)
    fork.append(child_b)

    assert [e.id for e in sm.get_active_branch()] == [root.id, child_a.id]
    assert [e.id for e in fork.get_active_branch()] == [root.id, child_b.id]


def test_inmemory_navigate_to_changes_active_branch() -> None:
    sm = InMemorySessionManager()
    root = message_entry(text_message("root"), parent_id=None)
    sm.append(root)
    a = message_entry(text_message("a"), parent_id=root.id)
    sm.append(a)
    b = message_entry(text_message("b"), parent_id=root.id)
    sm.append(b)

    assert sm.get_active_branch()[-1].id == b.id

    sm.navigate_to(a.id)
    assert [e.id for e in sm.get_active_branch()] == [root.id, a.id]


def test_inmemory_find_returns_entry_or_none() -> None:
    sm = InMemorySessionManager()
    root = message_entry(text_message("root"), parent_id=None)
    sm.append(root)

    assert sm.find(root.id) is root
    assert sm.find("does-not-exist") is None


def test_build_session_context_uses_latest_compaction_summary() -> None:
    sm = InMemorySessionManager()
    sm.append_message(text_message("first", timestamp=1.0))
    sm.append_message(text_message("second", timestamp=2.0))
    third = sm.append_message(text_message("third", timestamp=3.0))
    sm.append_custom_entry(
        "compaction",
        {
            "summary": "Compaction summary of 2 earlier messages",
            "first_kept_entry_id": third.id,
        },
    )
    sm.append_message(text_message("after", timestamp=5.0))

    texts = [
        block.text
        for msg in sm.get_messages()
        for block in getattr(msg, "content", [])
        if isinstance(block, TextContent)
    ]
    assert texts[0] == "Compaction summary of 2 earlier messages"
    assert "first" not in texts
    assert "second" not in texts
    assert "third" in texts
    assert texts[-1] == "after"


def test_get_tree_marks_nodes_with_compacted_ancestor_state() -> None:
    sm = InMemorySessionManager()
    root = sm.append_message(text_message("root", timestamp=1.0))
    kept = sm.append_message(text_message("kept", timestamp=2.0))
    compact = sm.append_custom_entry(
        "compaction",
        {
            "summary": "Compaction summary",
            "first_kept_entry_id": kept.id,
        },
    )
    tail = sm.append_message(text_message("tail", timestamp=3.0))

    sm.branch(root.id)
    sibling = sm.append_message(text_message("sibling", timestamp=4.0))

    roots = sm.get_tree()
    assert len(roots) == 1
    root_node = roots[0]
    assert root_node.entry.id == root.id
    assert root_node.has_compacted_ancestor is False

    child_map = {child.entry.id: child for child in root_node.children}
    assert child_map[kept.id].has_compacted_ancestor is False
    assert child_map[sibling.id].has_compacted_ancestor is False

    compact_node = child_map[kept.id].children[0]
    assert compact_node.entry.id == compact.id
    assert compact_node.has_compacted_ancestor is False
    assert compact_node.children[0].entry.id == tail.id
    assert compact_node.children[0].has_compacted_ancestor is True


def test_jsonl_durable_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"

    sm1 = JsonlSessionManager(path)
    e1 = sm1.append_message(text_message("hello"))
    e2 = sm1.append_message(text_message("world"))

    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    sm2 = JsonlSessionManager(path)
    branch = sm2.get_active_branch()
    assert [e.id for e in branch] == [e1.id, e2.id]
    assert [e.type for e in branch] == ["message", "message"]
    assert [e.parent_id for e in branch] == [None, e1.id]
    assert [msg.role for msg in sm2.get_messages()] == ["user", "user"]


def test_jsonl_appends_are_durable_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    sm1 = JsonlSessionManager(path)
    e1 = sm1.append_message(text_message("first"))

    sm2 = JsonlSessionManager(path)
    e2 = sm2.append_message(text_message("second"))

    sm3 = JsonlSessionManager(path)
    assert [e.id for e in sm3.get_active_branch()] == [e1.id, e2.id]


def test_create_branched_session_linearizes_selected_path(tmp_path: Path) -> None:
    manager = SessionManager.create(str(tmp_path), tmp_path / "sessions")
    root = manager.append_message(text_message("root"))
    left = manager.append_message(text_message("left"))
    manager.branch(root.id)
    right = manager.append_message(text_message("right"))

    fork_path = manager.create_branched_session(right.id)
    assert fork_path is not None

    forked = SessionManager.open(fork_path)
    texts = [
        block.text
        for msg in forked.get_messages()
        for block in getattr(msg, "content", [])
        if isinstance(block, TextContent)
    ]
    assert texts == ["root", "right"]
    assert left.id != right.id


def test_inmemory_rejects_dangling_parent_id() -> None:
    sm = InMemorySessionManager()
    orphan = message_entry(text_message("orphan"), parent_id="nonexistent")
    with pytest.raises(ValueError):
        sm.append(orphan)
