"""Session-store port for presenter-owned session lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from agentm.core.abi.session import SessionContext, SessionEntry, SessionHeader
from agentm.core.abi.messages import AgentMessage


@runtime_checkable
class SessionState(Protocol):
    """Mutable session state consumed by the runtime."""

    @property
    def session_file(self) -> Path | None: ...

    def append_message(self, message: AgentMessage) -> SessionEntry: ...
    def build_session_context(self, leaf_id: str | None = None) -> SessionContext: ...
    def get_session_id(self) -> str: ...
    def get_session_file(self) -> str | None: ...
    def get_header(self) -> SessionHeader | None: ...
    def is_persisted(self) -> bool: ...


@runtime_checkable
class SessionStore(Protocol):
    """Session lookup/creation boundary for CLI and other presenters."""

    def open(self, id: str) -> SessionState: ...
    def most_recent(self, cwd: Path) -> SessionState | None: ...
    def create(self, cwd: Path) -> SessionState: ...
    def fork(
        self,
        source_id: str,
        *,
        up_to: int | None = None,
        message_id: str | None = None,
        turn_id: int | None = None,
        turn_index: int | None = None,
    ) -> SessionState:
        """Create a new session seeded with messages from an existing one.

        Reads the source session's message entries and replays them into a
        fresh session. The new session's header records the source via
        ``parent_session`` so the fork relationship is queryable.

        ``message_id`` forks at an exact persisted message entry. ``turn_id``
        and ``turn_index`` fork at a resolved turn boundary. ``up_to`` is the
        legacy message-count selector retained for compatibility.
        """
        ...


__all__ = ["SessionState", "SessionStore"]
