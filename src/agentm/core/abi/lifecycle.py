"""World-effect lifecycle ports.

Lifecycle is the boundary that keeps a session's committed trajectory aligned
with external effects. It is intentionally narrower than a generic hook API:
implementations manage turn-scoped effect transactions, forked world state, and
resume restoration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.trajectory import Turn, TurnRef


LifecycleMeta = Mapping[str, str | int | float | bool | None]
EnvironmentRestoreState = Literal["restored", "degraded_readonly"]
EnvironmentCheckpoint = Literal["before_turn", "after_turn", "fork"]


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


@dataclass(frozen=True, slots=True)
class EnvironmentRestoreStatus:
    """Recorded result of a resume-time environment restore attempt."""

    session_id: str
    restored: bool
    state: EnvironmentRestoreState
    error: str | None = None
    metadata: LifecycleMeta = field(default_factory=dict)


class EnvironmentRestoreError(RuntimeError):
    """Raised when resume cannot restore the external effect scope."""


@runtime_checkable
class EnvironmentRestoreFailureHandler(Protocol):
    """Host enforcement required before a failed restore may continue."""

    async def activate_degraded_readonly(
        self,
        status: EnvironmentRestoreStatus,
    ) -> None:
        """Make the resumed session unable to mutate its external world."""


@runtime_checkable
class EffectScope(Protocol):
    """Turn-scoped lifecycle for external effects.

    ``begin_turn`` starts an effect transaction for the active turn,
    ``prepare_turn`` durably captures the resulting world before the
    authoritative Turn append, ``commit_turn`` confirms it after append, and
    ``abandon_turn`` restores the pre-turn world when execution fails.
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

    async def prepare_turn(self, txn: EffectTxn, turn: Turn) -> None:
        ...

    async def abandon_turn(self, txn: EffectTxn) -> None:
        ...

    async def fork_at(
        self,
        ref: TurnRef,
        *,
        source_session_id: str,
        child_session_id: str,
    ) -> "EnvironmentFork":
        ...

    async def restore(
        self,
        *,
        session_id: str,
        turns: Sequence[Turn],
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class EnvironmentFork:
    """Backend-produced bindings for an isolated trajectory branch.

    A fork is more than an ``EffectScope`` clone. The child must execute
    against the same world represented by that scope, so the backend also
    returns the child's cwd and, when it owns one, its operations bundle.
    Workspace resource writers are rebound separately through
    ``EnvironmentForkableResourceWriter``.
    """

    effect_scope: EffectScope
    cwd: str
    operations: EnvironmentOperations | None = None


@runtime_checkable
class EnvironmentSnapshotter(Protocol):
    """Backend-owned snapshot/restore boundary for execution environments."""

    async def snapshot(
        self,
        *,
        session_id: str,
        ref: TurnRef,
        metadata: LifecycleMeta | None = None,
    ) -> EnvironmentSnapshot:
        ...

    async def fork_from(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        child_session_id: str,
    ) -> EnvironmentFork | None:
        ...

    async def restore_to(self, snapshot: EnvironmentSnapshot) -> None:
        ...

    async def find_snapshot(
        self,
        *,
        session_id: str,
        ref: TurnRef | None = None,
        checkpoint: EnvironmentCheckpoint,
    ) -> EnvironmentSnapshot | None:
        ...

    async def discard(self, snapshot: EnvironmentSnapshot) -> None:
        ...


__all__ = [
    "EffectScope",
    "EffectTxn",
    "EnvironmentFork",
    "EnvironmentRestoreError",
    "EnvironmentRestoreFailureHandler",
    "EnvironmentRestoreStatus",
    "EnvironmentRestoreState",
    "EnvironmentCheckpoint",
    "EnvironmentSnapshot",
    "EnvironmentSnapshotter",
    "LifecycleMeta",
]
