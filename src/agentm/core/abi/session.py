"""Session-tree data shapes shared by the kernel and harness.

These are pure data classes plus a minimal read-only Protocol over the
session tree. They live in ``core/abi`` so the constitution layer
(``core/_internal/compaction``, ``core/_internal/catalog``) can manipulate
session entries without reverse-importing from ``harness``.

The full ``SessionManager`` (with persistence, fork/navigate semantics)
remains in ``harness/session_manager.py`` and implements ``SessionTree``
implicitly. Atoms that need the read-only surface should depend on
``SessionTree`` rather than the concrete manager.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi.messages import AgentMessage

CURRENT_SESSION_VERSION = 1


@dataclass(frozen=True, slots=True)
class SessionHeader:
    type: str
    version: int
    id: str
    timestamp: float
    cwd: str
    parent_session: str | None = None


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """One immutable node in the session tree."""

    type: str
    id: str
    parent_id: str | None
    timestamp: float
    payload: Any


@dataclass(frozen=True, slots=True)
class SessionContext:
    messages: list[AgentMessage]


@dataclass(slots=True)
class SessionTreeNode:
    entry: SessionEntry
    children: list["SessionTreeNode"]
    has_compacted_ancestor: bool = False


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


def message_entry(msg: AgentMessage, parent_id: str | None) -> SessionEntry:
    return SessionEntry(
        type="message",
        id=_new_id(),
        parent_id=parent_id,
        timestamp=_now(),
        payload=msg,
    )


def branch_summary_entry(
    summary: str,
    parent_id: str | None,
    *,
    from_id: str | None = None,
    details: Any = None,
) -> SessionEntry:
    return SessionEntry(
        type="branch_summary",
        id=_new_id(),
        parent_id=parent_id,
        timestamp=_now(),
        payload={
            "summary": summary,
            "from_id": from_id or "root",
            "details": details,
        },
    )


def compaction_entry(payload: Any, parent_id: str | None) -> SessionEntry:
    return SessionEntry(
        type="compaction",
        id=_new_id(),
        parent_id=parent_id,
        timestamp=_now(),
        payload=payload,
    )


@runtime_checkable
class SessionTree(Protocol):
    """Read-only window over the session tree.

    The full ``SessionManager`` implements this implicitly. Constitution-
    layer modules (compaction, branch summarization) depend only on this
    surface so they don't need to import from ``harness``.
    """

    def get_branch(self, leaf_id: str | None = None) -> list[SessionEntry]: ...
    def get_entry(self, entry_id: str) -> SessionEntry | None: ...


__all__ = [
    "CURRENT_SESSION_VERSION",
    "SessionContext",
    "SessionEntry",
    "SessionHeader",
    "SessionTree",
    "SessionTreeNode",
    "branch_summary_entry",
    "compaction_entry",
    "message_entry",
]
