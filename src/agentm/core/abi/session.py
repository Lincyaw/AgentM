"""Session-tree data shapes shared by the kernel and runtime.

These are pure data classes plus a minimal read-only Protocol over the
session tree. They live in ``core/abi`` so the constitution layer
(``core/_internal/catalog`` and the compaction engine inside the
``llm_compaction`` atom) can manipulate session entries without
reverse-importing from ``core.runtime``.

The full ``SessionManager`` (with persistence, fork/navigate semantics)
lives in ``core/runtime/session_manager.py`` and implements
``SessionTree`` implicitly. Atoms that need the read-only surface should
depend on ``SessionTree`` rather than the concrete manager.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi.messages import AgentMessage

CURRENT_SESSION_VERSION = 1


# Canonical kernel-defined entry-type strings. The kernel itself recognises
# only these three structural categories; any additional ``entry.type``
# values come from atoms and are routed exclusively through
# :data:`ENTRY_MATERIALIZERS`. Centralising the literals here removes the
# scattered string-literal entry-type comparisons that issue #76 calls out:
# the kernel still tests by category, but there are no scenario-name string
# literals embedded in branching logic.
ENTRY_TYPE_MESSAGE = "message"
ENTRY_TYPE_BRANCH_SUMMARY = "branch_summary"
ENTRY_TYPE_COMPACTION = "compaction"
# Durable turn-boundary marker. Appended at every clean ``agent_end`` (see
# ``AgentSession._on_agent_end_commit_boundary``). It carries no materializer,
# so ``build_session_context`` never turns it into an LLM message — it is a
# resume-only signal: a process killed mid-turn never reaches the handler, so
# its half-turn is left unmarked and ``_truncate_to_last_boundary`` sheds it on
# the next cold load. Invisible to the model, visible to resume.
ENTRY_TYPE_TURN_COMMITTED = "turn_committed"


@dataclass(frozen=True, slots=True)
class SessionHeader:
    type: str
    version: int
    id: str
    timestamp: float
    cwd: str
    parent_session: str | None = None
    config: dict[str, Any] | None = None


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
        type=ENTRY_TYPE_MESSAGE,
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
        type=ENTRY_TYPE_BRANCH_SUMMARY,
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
        type=ENTRY_TYPE_COMPACTION,
        id=_new_id(),
        parent_id=parent_id,
        timestamp=_now(),
        payload=payload,
    )


def turn_committed_entry(parent_id: str | None, payload: Any = None) -> SessionEntry:
    return SessionEntry(
        type=ENTRY_TYPE_TURN_COMMITTED,
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
    surface so they don't need to import from ``core.runtime``.
    """

    def get_branch(self, leaf_id: str | None = None) -> list[SessionEntry]: ...
    def get_entry(self, entry_id: str) -> SessionEntry | None: ...


@runtime_checkable
class EntryMaterializer(Protocol):
    """Convert a ``SessionEntry`` into an ``AgentMessage`` (or ``None``).

    Atoms register one materializer per ``entry.type`` they own. The
    runtime's ``build_session_context`` consults the global
    ``ENTRY_MATERIALIZERS`` registry so the kernel does not branch on
    string-literal entry types.
    """

    def to_message(self, entry: SessionEntry) -> AgentMessage | None: ...


# Mutable module-level registry — populated at atom install time. Keys are
# ``entry.type`` strings; values are objects satisfying ``EntryMaterializer``.
# Empty by default; atoms (typically ``extensions.builtin.compaction_prompts``)
# register defaults for ``"message"`` / ``"branch_summary"`` / ``"compaction"``.
ENTRY_MATERIALIZERS: dict[str, EntryMaterializer] = {}


__all__ = [
    "CURRENT_SESSION_VERSION",
    "ENTRY_MATERIALIZERS",
    "ENTRY_TYPE_BRANCH_SUMMARY",
    "ENTRY_TYPE_COMPACTION",
    "ENTRY_TYPE_MESSAGE",
    "ENTRY_TYPE_TURN_COMMITTED",
    "EntryMaterializer",
    "SessionContext",
    "SessionEntry",
    "SessionHeader",
    "SessionTree",
    "SessionTreeNode",
    "branch_summary_entry",
    "compaction_entry",
    "message_entry",
    "turn_committed_entry",
]
