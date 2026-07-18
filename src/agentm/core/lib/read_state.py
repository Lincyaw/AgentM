"""Session-scoped read-file state for read-before-edit coordination.

After a successful read, the ``read`` tool calls :func:`record_read` so
that the ``edit`` tool can later call :func:`get_read_state` to decide
whether the file was fully or only partially read.

State is keyed by ``(session_id, normalized_path)``.  The active session is
bound on the :data:`_CURRENT_SESSION` ContextVar by the session driver
via :func:`bind_session`; asyncio tasks copy
the context at creation, so when many sessions run concurrently in one
process (batch evaluation) each session's read-before-edit state is isolated
and one session's read/edit of a path cannot clobber another's.  Calls made
with no session bound (tests, standalone use) fall back to a shared ``""``
session, preserving the original single-session behaviour.

Aligned with Claude Code's ``readFileState`` — tracks mtime and a content
hash so the ``edit`` tool can detect files modified between read and edit.
"""

from __future__ import annotations

import hashlib
import os
from contextvars import ContextVar

from loguru import logger
from dataclasses import dataclass


@dataclass(slots=True)
class FileReadState:
    """Metadata recorded after a successful file read."""

    total_lines: int
    is_partial: bool
    mtime_ns: int = 0
    content_hash: str = ""


# Active session id for read-state attribution. Bound per session by the
# driver task; a ContextVar (not a module global) so concurrent sessions in
# one process do not share read-before-edit state.
_CURRENT_SESSION: ContextVar[str] = ContextVar(
    "agentm_read_state_session", default=""
)

_state: dict[tuple[str, str], FileReadState] = {}


def bind_session(session_id: str) -> None:
    """Bind *session_id* as the read-state scope for the current task.

    Called once at the top of the session driver task so every read/edit
    tool call within that session keys its state under its own session id.
    """
    _CURRENT_SESSION.set(session_id)


def _key(normalized: str) -> tuple[str, str]:
    return (_CURRENT_SESSION.get(), normalized)


def record_read(
    path: str,
    *,
    total_lines: int,
    is_partial: bool,
    mtime_ns: int = 0,
    content_hash: str = "",
) -> None:
    """Record that *path* was read.  Called by the ``read`` tool."""
    normalized = os.path.normpath(path)
    _state[_key(normalized)] = FileReadState(
        total_lines=total_lines,
        is_partial=is_partial,
        mtime_ns=mtime_ns,
        content_hash=content_hash,
    )


def get_read_state(path: str) -> FileReadState | None:
    """Return the last recorded read state for *path*, or ``None``."""
    normalized = os.path.normpath(path)
    return _state.get(_key(normalized))


def file_modified_since_read(path: str) -> bool:
    """Return True if the file's mtime is newer than the last recorded read.

    Returns False if the file was never read (caller should check
    ``get_read_state`` for that) or if the file no longer exists.
    """
    normalized = os.path.normpath(path)
    state = _state.get(_key(normalized))
    if state is None or state.mtime_ns == 0:
        return False
    try:
        current_mtime_ns = os.stat(normalized).st_mtime_ns
    except OSError as exc:
        logger.debug("read_state: stat({}) failed, reporting not-changed: {}", normalized, exc)
        return False
    return current_mtime_ns > state.mtime_ns


def content_hash_for(data: bytes) -> str:
    """Return the sha256 hex digest for *data*."""
    return hashlib.sha256(data).hexdigest()


def clear() -> None:
    """Clear all recorded state for every session.  Useful for testing."""
    _state.clear()
