"""Fail-stop: read-before-edit state must be isolated per session.

When many sessions run concurrently in one process (batch evaluation), one
session's read of a path must not clobber another session's read state for the
same path — otherwise the edit-staleness guard cross-contaminates across
sessions. The driver binds the session id on a ContextVar; each asyncio task
copies the context, so the binding is task-local.
"""

from __future__ import annotations

import asyncio

import pytest

from agentm.core.lib import read_state


@pytest.fixture(autouse=True)
def _clean_state():
    read_state.clear()
    yield
    read_state.clear()


def test_same_path_isolated_across_bound_sessions():
    read_state.bind_session("sess-a")
    read_state.record_read("/repo/foo.py", total_lines=10, is_partial=False)

    read_state.bind_session("sess-b")
    # Session B never read foo.py — must not see A's record.
    assert read_state.get_read_state("/repo/foo.py") is None

    read_state.record_read("/repo/foo.py", total_lines=3, is_partial=True)
    b = read_state.get_read_state("/repo/foo.py")
    assert b is not None and b.is_partial is True and b.total_lines == 3

    # A's record is untouched by B writing the same path.
    read_state.bind_session("sess-a")
    a = read_state.get_read_state("/repo/foo.py")
    assert a is not None and a.is_partial is False and a.total_lines == 10


def test_concurrent_tasks_do_not_share_read_state():
    async def reader(session_id: str, lines: int) -> int | None:
        read_state.bind_session(session_id)
        read_state.record_read("/shared/path.py", total_lines=lines, is_partial=False)
        await asyncio.sleep(0)  # yield so tasks interleave
        st = read_state.get_read_state("/shared/path.py")
        return st.total_lines if st else None

    async def main() -> list[int | None]:
        return await asyncio.gather(reader("s1", 1), reader("s2", 2), reader("s3", 3))

    # Each task copies the parent context, so its bind_session is task-local;
    # interleaving must not let one task read another's line count.
    assert asyncio.run(main()) == [1, 2, 3]


def test_unbound_falls_back_to_shared_scope():
    # No session bound: default "" scope preserves single-session behaviour.
    read_state.record_read("/x.py", total_lines=5, is_partial=False)
    st = read_state.get_read_state("/x.py")
    assert st is not None and st.total_lines == 5
