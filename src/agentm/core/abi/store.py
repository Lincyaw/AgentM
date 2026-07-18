"""TrajectoryStore — persistence protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agentm.core.abi.trajectory import Turn, TurnRef


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Metadata for a session record in the store."""

    id: str
    parent_id: str | None = None
    fork_point: TurnRef | None = None
    purpose: str = "root"
    cwd: str = ""
    created_at: float = 0.0
    config: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@runtime_checkable
class TrajectoryStore(Protocol):
    """Persistence boundary for trajectories.

    ``append`` must be atomic — a Turn is either fully written or not.
    """

    def create_session(self, meta: SessionMeta) -> None: ...

    def append(self, session_id: str, turn: Turn) -> None: ...

    def append_round(
        self, session_id: str, turn_id: str, round_data: dict[str, object]
    ) -> None:
        """Durable round checkpoint.  Optional — stores may no-op."""
        ...

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]: ...

    def load_prefix(
        self, session_id: str, up_to: TurnRef
    ) -> tuple[SessionMeta, list[Turn]]: ...

    def session_children(self, session_id: str) -> list[str]: ...

    def session_exists(self, session_id: str) -> bool: ...

    def list_sessions(self) -> list[SessionMeta]: ...

    def load_durable_rounds(
        self, session_id: str, turn_id: str
    ) -> list[dict[str, object]]: ...


__all__ = [
    "SessionMeta",
    "TrajectoryStore",
]
