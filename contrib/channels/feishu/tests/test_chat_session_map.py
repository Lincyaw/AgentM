from __future__ import annotations

from pathlib import Path

from agentm_feishu.chat_session_map import ChatSessionMap


def test_set_get_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    a = ChatSessionMap(path)
    a.set("oc_chat", "sess-1")
    a.set("oc_chat", "sess-2", thread_id="omt_thread")
    a.set("oc_other", "sess-3")

    b = ChatSessionMap(path)
    assert b.get("oc_chat") == "sess-1"
    assert b.get("oc_chat", "omt_thread") == "sess-2"
    assert b.get("oc_other") == "sess-3"
    assert b.get("oc_chat", "missing-thread") is None


def test_drop_removes_route(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    m = ChatSessionMap(path)
    m.set("oc_chat", "sess-1")
    m.drop("oc_chat")
    assert m.get("oc_chat") is None
    assert ChatSessionMap(path).get("oc_chat") is None


def test_corrupt_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text("not json {{{")
    m = ChatSessionMap(path)
    assert m.snapshot() == {}
    m.set("oc_chat", "sess-1")
    assert m.get("oc_chat") == "sess-1"
