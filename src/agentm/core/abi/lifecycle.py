"""World-effect lifecycle ports.

Lifecycle is the boundary that keeps a session's committed trajectory aligned
with external effects. It is intentionally narrower than a generic hook API:
implementations manage turn-scoped effect transactions, forked world state, and
resume restoration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agentm.core.abi.trajectory import Turn, TurnRef


LifecycleMeta = Mapping[str, str | int | float | bool | None]


@dataclass(frozen=True, slots=True)
class EffectTxn:
    """Opaque handle for side effects produced while one turn is executing."""

    session_id: str
    turn_id: str
    turn_index: int
    token: str = ""
    metadata: LifecycleMeta = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Durable identity for a world-state snapshot."""

    id: str
    session_id: str = ""
    ref: TurnRef | None = None
    metadata: LifecycleMeta = field(default_factory=dict)


@runtime_checkable
class EffectScope(Protocol):
    """Turn-scoped lifecycle for external effects.

    ``begin_turn`` starts an effect transaction for the active turn,
    ``commit_turn`` finalizes it after durable commit, and ``abandon_turn``
    discards it when execution fails or is cancelled.
    """

    async def begin_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        turn_index: int,
    ) -> EffectTxn:
        ...

    async def commit_turn(self, txn: EffectTxn, turn: Turn) -> None:
        ...

    async def abandon_turn(self, txn: EffectTxn) -> None:
        ...

    async def fork_at(
        self,
        ref: TurnRef,
        *,
        source_session_id: str,
        child_session_id: str,
    ) -> "EffectScope":
        ...

    async def restore(
        self,
        *,
        session_id: str,
        turns: Sequence[Turn],
    ) -> None:
        ...


@runtime_checkable
class EnvironmentSnapshotter(Protocol):
    """Backend-owned snapshot/restore boundary for execution environments."""

    async def snapshot(
        self,
        *,
        session_id: str,
        ref: TurnRef,
    ) -> EnvironmentSnapshot:
        ...

    async def fork_from(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        child_session_id: str,
    ) -> EnvironmentSnapshot | None:
        ...

    async def restore_to(self, snapshot: EnvironmentSnapshot) -> None:
        ...


__all__ = [
    "EffectScope",
    "EffectTxn",
    "EnvironmentSnapshot",
    "EnvironmentSnapshotter",
    "LifecycleMeta",
]
