from __future__ import annotations

from pathlib import Path

from agentm_channels.chat_session_map import ChatSessionMap


def test_set_get_persists(tmp_path: Path) -> None:
    p = tmp_path / "map.json"
    a = ChatSessionMap(p)
    a.set("feishu:c1", "sess-1")
    a.set("feishu:c1::omt_thread", "sess-2")
    a.set("slack:c2", "sess-3")
    b = ChatSessionMap(p)
    assert b.get("feishu:c1") == "sess-1"
    assert b.get("feishu:c1::omt_thread") == "sess-2"
    assert b.get("slack:c2") == "sess-3"
    assert b.get("nope") is None


def test_drop_removes(tmp_path: Path) -> None:
    p = tmp_path / "map.json"
    m = ChatSessionMap(p)
    m.set("feishu:c1", "sess-1")
    m.drop("feishu:c1")
    assert m.get("feishu:c1") is None
    assert ChatSessionMap(p).get("feishu:c1") is None
