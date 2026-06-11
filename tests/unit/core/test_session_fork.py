"""Fork semantics on ``JsonlSessionStore``.

Verifies the fail-stop invariant: a forked session is a *new* session
whose trajectory begins with a prefix of the source, and whose header
records the source via ``parent_session``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.core.abi import TextContent, UserMessage, AssistantMessage, text_message
from agentm.core.abi.session import ENTRY_TYPE_MESSAGE
from agentm.core.runtime.session_manager import JsonlSessionStore, SessionManager


def _user(text: str) -> UserMessage:
    return text_message(text, timestamp=0.0)


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
        stop_reason="end_turn",
    )


def _make_source(session_dir: Path) -> SessionManager:
    mgr = SessionManager(cwd=str(session_dir), session_dir=session_dir, persist=True)
    mgr.append_message(_user("hello"))
    mgr.append_message(_assistant("world"))
    mgr.append_message(_user("how are you"))
    mgr.append_message(_assistant("fine"))
    return mgr


class TestFork:
    def test_fork_copies_all_messages(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        store = JsonlSessionStore(session_dir=tmp_path)

        forked = store.fork(source.get_session_id())
        branch = forked.get_branch()
        messages = [e.payload for e in branch if e.type == ENTRY_TYPE_MESSAGE]

        assert len(messages) == 4
        assert messages[0].content[0].text == "hello"  # type: ignore[union-attr]
        assert messages[3].content[0].text == "fine"  # type: ignore[union-attr]

    def test_fork_truncates_with_up_to(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        store = JsonlSessionStore(session_dir=tmp_path)

        forked = store.fork(source.get_session_id(), up_to=2)
        branch = forked.get_branch()
        messages = [e.payload for e in branch if e.type == ENTRY_TYPE_MESSAGE]

        assert len(messages) == 2
        assert messages[0].content[0].text == "hello"  # type: ignore[union-attr]
        assert messages[1].content[0].text == "world"  # type: ignore[union-attr]

    def test_fork_creates_new_session_id(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        store = JsonlSessionStore(session_dir=tmp_path)

        forked = store.fork(source.get_session_id())
        assert forked.get_session_id() != source.get_session_id()

    def test_fork_records_parent_session(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        store = JsonlSessionStore(session_dir=tmp_path)

        forked = store.fork(source.get_session_id())
        header = forked.get_header()
        assert header is not None
        assert header.parent_session == source.get_session_id()

    def test_fork_persists_to_new_jsonl(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        store = JsonlSessionStore(session_dir=tmp_path)

        forked = store.fork(source.get_session_id(), up_to=2)
        forked_file = forked.session_file
        assert forked_file is not None
        assert forked_file.exists()
        assert forked_file != source.session_file

        reloaded = SessionManager.open(forked_file)
        branch = reloaded.get_branch()
        messages = [e.payload for e in branch if e.type == ENTRY_TYPE_MESSAGE]
        assert len(messages) == 2

    def test_fork_does_not_mutate_source(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        store = JsonlSessionStore(session_dir=tmp_path)

        forked = store.fork(source.get_session_id(), up_to=2)
        forked.append_message(_user("extra"))

        source_branch = source.get_branch()
        source_messages = [e.payload for e in source_branch if e.type == ENTRY_TYPE_MESSAGE]
        assert len(source_messages) == 4

    def test_fork_unknown_source_raises(self, tmp_path: Path) -> None:
        store = JsonlSessionStore(session_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.fork("nonexistent")
