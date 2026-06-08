"""Shared read-file state for read-before-edit coordination.

After a successful read, the ``read`` tool calls :func:`record_read` so
that the ``edit`` tool can later call :func:`get_read_state` to decide
whether the file was fully or only partially read.  The module-level
``_state`` dict is per-process, which is fine since each AgentM session
runs in its own process.

Aligned with Claude Code's ``readFileState`` — tracks mtime and a content
hash so the ``edit`` tool can detect files modified between read and edit.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


@dataclass(slots=True)
class FileReadState:
    """Metadata recorded after a successful file read."""

    total_lines: int
    is_partial: bool
    mtime_ns: int = 0
    content_hash: str = ""


_state: dict[str, FileReadState] = {}


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
    _state[normalized] = FileReadState(
        total_lines=total_lines,
        is_partial=is_partial,
        mtime_ns=mtime_ns,
        content_hash=content_hash,
    )


def get_read_state(path: str) -> FileReadState | None:
    """Return the last recorded read state for *path*, or ``None``."""
    normalized = os.path.normpath(path)
    return _state.get(normalized)


def file_modified_since_read(path: str) -> bool:
    """Return True if the file's mtime is newer than the last recorded read.

    Returns False if the file was never read (caller should check
    ``get_read_state`` for that) or if the file no longer exists.
    """
    normalized = os.path.normpath(path)
    state = _state.get(normalized)
    if state is None or state.mtime_ns == 0:
        return False
    try:
        current_mtime_ns = os.stat(normalized).st_mtime_ns
    except OSError:
        return False
    return current_mtime_ns > state.mtime_ns


def content_hash_for(data: bytes) -> str:
    """Return the sha256 hex digest for *data*."""
    return hashlib.sha256(data).hexdigest()


def clear() -> None:
    """Clear all recorded state.  Useful for testing."""
    _state.clear()
