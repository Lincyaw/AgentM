# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""World-effect lifecycle ports.

Lifecycle is the boundary that keeps a session's committed trajectory aligned
with external effects. It is intentionally narrower than a generic hook API:
implementations manage turn-scoped effect transactions, forked world state, and
resume restoration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.trajectory import Turn, TurnRef


LifecycleMeta = Mapping[str, str | int | float | bool | None]
EnvironmentRestoreState = Literal["restored", "degraded_readonly"]
EnvironmentCheckpoint = Literal["before_turn", "after_turn", "fork"]


def _require_nonempty_string(
    value: object,
    label: str,
    *,
    optional: bool = False,
) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value:
        expected = "a non-empty string or None" if optional else "a non-empty string"
        raise TypeError(f"{label} must be {expected}")


def _require_turn_ref(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{label} must be a non-empty string or non-negative integer")
    if (isinstance(value, str) and not value) or (isinstance(value, int) and value < 0):
        raise ValueError(f"{label} must be a non-empty string or non-negative integer")


def _require_index(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{label} must be a non-negative integer")


def _freeze_metadata(value: LifecycleMeta, label: str) -> LifecycleMeta:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    copied: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise TypeError(f"{label}[{key!r}] must be a JSON scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{label}[{key!r}] must be finite")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class EffectTxn:
    """Opaque handle for side effects produced while one turn is executing."""

    session_id: str
    turn_id: str
    turn_index: int
    token: str
    metadata: LifecycleMeta = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.session_id, "effect transaction session_id")
        _require_nonempty_string(self.turn_id, "effect transaction turn_id")
        _require_index(self.turn_index, "effect transaction turn_index")
        _require_nonempty_string(self.token, "effect transaction token")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "effect transaction metadata"),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Durable identity for a world-state snapshot."""

    id: str
    session_id: str
    ref: TurnRef
    metadata: LifecycleMeta = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.id, "environment snapshot id")
        _require_nonempty_string(self.session_id, "environment snapshot session_id")
        _require_turn_ref(self.ref, "environment snapshot ref")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "environment snapshot metadata"),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentRestoreStatus:
    """Recorded result of a resume-time environment restore attempt."""

    session_id: str
    restored: bool
    state: EnvironmentRestoreState
    error: str | None = None
    metadata: LifecycleMeta = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.session_id, "environment restore session_id")
        if not isinstance(self.restored, bool):
            raise TypeError("environment restore restored must be a bool")
        if self.state not in {"restored", "degraded_readonly"}:
            raise ValueError(f"invalid environment restore state: {self.state!r}")
        _require_nonempty_string(
            self.error,
            "environment restore error",
            optional=True,
        )
        if self.restored != (self.state == "restored"):
            raise ValueError("environment restore state must agree with restored")
        if self.restored and self.error is not None:
            raise ValueError("a restored environment cannot carry an error")
        if not self.restored and self.error is None:
            raise ValueError("a degraded environment restore requires an error")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "environment restore metadata"),
        )


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

    The captured world must exclude control-plane persistence owned by other
    commit participants, including trajectory logs, resource transaction
    journals, catalogs, and snapshot metadata. Those stores recover from their
    own durable records; restoring them as world state would rewind a commit
    that already reached a known outcome.
    """

    async def begin_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        turn_index: int,
    ) -> EffectTxn: ...

    async def commit_turn(self, txn: EffectTxn, turn: Turn) -> None: ...

    async def prepare_turn(self, txn: EffectTxn, turn: Turn) -> None: ...

    async def abandon_turn(self, txn: EffectTxn) -> None: ...

    async def fork_at(
        self,
        ref: TurnRef,
        *,
        source_session_id: str,
        child_session_id: str,
    ) -> "EnvironmentFork": ...

    async def restore(
        self,
        *,
        session_id: str,
        turns: Sequence[Turn],
    ) -> None: ...


@runtime_checkable
class EnvironmentForkLease(Protocol):
    """Own provisional resources created while constructing a child branch.

    ``commit()`` promotes the fork point into restorable child state before
    child-session metadata is published. ``Session.fork()`` calls ``abandon()``
    when construction fails before that durable publication, including caller
    cancellation after the backend operation started. Both methods must be
    idempotent, and ``abandon()`` remains valid after ``commit()`` until the
    child session is durable. Once metadata is durable, normal lifetime
    ownership belongs to the child's registered services.
    """

    async def commit(self) -> None: ...

    async def abandon(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EnvironmentFork:
    """Backend-produced bindings for an isolated trajectory branch.

    A fork is more than an ``EffectScope`` clone. The child must execute
    against the same world represented by that scope, so the backend also
    returns the child's cwd and, when it owns one, its operations bundle.
    Workspace resource writers are rebound separately through
    ``EnvironmentForkableResourceWriter``. ``lease`` is mandatory because an
    async backend may finish allocating the branch after its caller is
    cancelled; the runtime must still be able to clean up that known result.
    """

    effect_scope: EffectScope
    cwd: str
    lease: EnvironmentForkLease
    operations: EnvironmentOperations | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.effect_scope, EffectScope):
            raise TypeError("environment fork effect_scope must implement EffectScope")
        if not isinstance(self.cwd, str):
            raise TypeError("environment fork cwd must be a string")
        if not isinstance(self.lease, EnvironmentForkLease):
            raise TypeError(
                "environment fork lease must implement EnvironmentForkLease"
            )
        if self.operations is not None and not isinstance(
            self.operations,
            EnvironmentOperations,
        ):
            raise TypeError(
                "environment fork operations must implement EnvironmentOperations"
            )


@runtime_checkable
class EnvironmentSnapshotter(Protocol):
    """Backend-owned snapshot/restore boundary for execution environments."""

    async def snapshot(
        self,
        *,
        session_id: str,
        ref: TurnRef,
        metadata: LifecycleMeta | None = None,
    ) -> EnvironmentSnapshot: ...

    async def fork_from(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        child_session_id: str,
    ) -> EnvironmentFork | None: ...

    async def restore_to(self, snapshot: EnvironmentSnapshot) -> None: ...

    async def find_snapshot(
        self,
        *,
        session_id: str,
        ref: TurnRef | None = None,
        checkpoint: EnvironmentCheckpoint,
    ) -> EnvironmentSnapshot | None: ...

    async def discard(self, snapshot: EnvironmentSnapshot) -> None: ...


__all__ = [
    "EffectScope",
    "EffectTxn",
    "EnvironmentFork",
    "EnvironmentForkLease",
    "EnvironmentRestoreError",
    "EnvironmentRestoreFailureHandler",
    "EnvironmentRestoreStatus",
    "EnvironmentRestoreState",
    "EnvironmentCheckpoint",
    "EnvironmentSnapshot",
    "EnvironmentSnapshotter",
    "LifecycleMeta",
]
