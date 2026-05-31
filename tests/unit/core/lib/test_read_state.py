"""Unit tests for the read-file state module used by tool_read → tool_edit."""

from __future__ import annotations

from agentm.core.lib.read_state import FileReadState, clear, get_read_state, record_read


def test_record_and_get() -> None:
    clear()
    record_read("./src/foo.py", total_lines=100, is_partial=False)
    state = get_read_state("./src/foo.py")
    assert state is not None
    assert state.total_lines == 100
    assert state.is_partial is False


def test_not_read_returns_none() -> None:
    clear()
    assert get_read_state("./unknown.py") is None


def test_path_normalization() -> None:
    clear()
    record_read("./src/../src/foo.py", total_lines=50, is_partial=True)
    state = get_read_state("src/foo.py")
    assert state is not None
    assert state.is_partial is True
    assert state.total_lines == 50


def test_full_read_overwrites_partial() -> None:
    clear()
    record_read("foo.py", total_lines=100, is_partial=True)
    assert get_read_state("foo.py") == FileReadState(total_lines=100, is_partial=True)
    record_read("foo.py", total_lines=100, is_partial=False)
    state = get_read_state("foo.py")
    assert state is not None
    assert state.is_partial is False


def test_clear_removes_all_state() -> None:
    record_read("a.py", total_lines=10, is_partial=False)
    record_read("b.py", total_lines=20, is_partial=True)
    clear()
    assert get_read_state("a.py") is None
    assert get_read_state("b.py") is None
