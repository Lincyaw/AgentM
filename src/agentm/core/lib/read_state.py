"""Shared read-file state for read-before-edit coordination.

After a successful read, ``tool_read`` calls :func:`record_read` so that
``tool_edit`` can later call :func:`get_read_state` to decide whether the
file was fully or only partially read.  The module-level ``_state`` dict
is per-process, which is fine since each AgentM session runs in its own
process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class FileReadState:
    """Metadata recorded after a successful file read."""

    total_lines: int
    is_partial: bool


_state: dict[str, FileReadState] = {}


def record_read(path: str, *, total_lines: int, is_partial: bool) -> None:
    """Record that *path* was read.  Called by ``tool_read``."""
    normalized = os.path.normpath(path)
    _state[normalized] = FileReadState(
        total_lines=total_lines,
        is_partial=is_partial,
    )


def get_read_state(path: str) -> FileReadState | None:
    """Return the last recorded read state for *path*, or ``None``."""
    normalized = os.path.normpath(path)
    return _state.get(normalized)


def clear() -> None:
    """Clear all recorded state.  Useful for testing."""
    _state.clear()
