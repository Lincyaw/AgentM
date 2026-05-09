"""Session-store port for presenter-owned session lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from agentm.core.abi.session import SessionContext, SessionEntry, SessionHeader
from agentm.core.abi.messages import AgentMessage


@runtime_checkable
class SessionState(Protocol):
    """Mutable session state consumed by the harness."""

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


__all__ = ["SessionState", "SessionStore"]
