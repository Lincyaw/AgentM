"""Lifecycle hooks for fork, replay, resume, and crash recovery.

Tools have side effects (file writes, command execution, sandbox
state).  When a session forks or replays, the environment state may
be inconsistent with the trajectory.  These hooks let atoms and
environments execute recovery strategies at the right moments.

Hook registration is per-session via ``api.register_lifecycle_hook``.
Hooks run in registration order.  Failures are logged but do not
block the operation — partial recovery is better than no session.

Hook taxonomy:

- ``on_fork``: a new session is being created from a prefix of this
  session's trajectory.  The hook receives the fork point and can set
  up the child environment (e.g., snapshot a sandbox).
- ``on_resume``: a session is being loaded from persisted state.  The
  hook receives the committed turns and can restore environment state
  (e.g., replay file writes into a fresh sandbox).
- ``on_replay``: like resume but the trajectory will be re-executed
  from a specific point.  The hook can roll back environment state to
  match the replay point.
- ``on_abandon``: a turn was abandoned (crash / interrupt).  The hook
  can clean up partial side effects from the uncommitted turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentm.core.v2.abi.trajectory import Round, Turn, TurnRef


@dataclass(frozen=True, slots=True)
class ForkEvent:
    """Passed to ``on_fork`` hooks."""

    source_session_id: str
    fork_session_id: str
    fork_point: TurnRef
    source_turns: Sequence[Turn]


@dataclass(frozen=True, slots=True)
class ResumeEvent:
    """Passed to ``on_resume`` hooks."""

    session_id: str
    committed_turns: Sequence[Turn]


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    """Passed to ``on_replay`` hooks."""

    session_id: str
    replay_from: TurnRef
    committed_turns: Sequence[Turn]


@dataclass(frozen=True, slots=True)
class AbandonEvent:
    """Passed to ``on_abandon`` hooks."""

    session_id: str
    turn_index: int
    completed_rounds: Sequence[Round]


@runtime_checkable
class LifecycleHook(Protocol):
    """Atom-registered hook for session lifecycle transitions.

    Each method is optional — implement only the events you care about.
    All methods are async to support I/O (sandbox API calls, file
    operations).
    """

    async def on_fork(self, event: ForkEvent) -> None: ...

    async def on_resume(self, event: ResumeEvent) -> None: ...

    async def on_replay(self, event: ReplayEvent) -> None: ...

    async def on_abandon(self, event: AbandonEvent) -> None: ...


class LifecycleHookRegistry:
    """Collects and dispatches lifecycle hooks."""

    __slots__ = ("_hooks",)

    def __init__(self) -> None:
        self._hooks: list[LifecycleHook] = []

    def register(self, hook: LifecycleHook) -> None:
        self._hooks.append(hook)

    async def fire_fork(self, event: ForkEvent) -> None:
        for hook in self._hooks:
            if hasattr(hook, "on_fork"):
                try:
                    await hook.on_fork(event)
                except Exception:
                    from loguru import logger
                    logger.exception("lifecycle on_fork failed")

    async def fire_resume(self, event: ResumeEvent) -> None:
        for hook in self._hooks:
            if hasattr(hook, "on_resume"):
                try:
                    await hook.on_resume(event)
                except Exception:
                    from loguru import logger
                    logger.exception("lifecycle on_resume failed")

    async def fire_replay(self, event: ReplayEvent) -> None:
        for hook in self._hooks:
            if hasattr(hook, "on_replay"):
                try:
                    await hook.on_replay(event)
                except Exception:
                    from loguru import logger
                    logger.exception("lifecycle on_replay failed")

    async def fire_abandon(self, event: AbandonEvent) -> None:
        for hook in self._hooks:
            if hasattr(hook, "on_abandon"):
                try:
                    await hook.on_abandon(event)
                except Exception:
                    from loguru import logger
                    logger.exception("lifecycle on_abandon failed")


__all__ = [
    "AbandonEvent",
    "ForkEvent",
    "LifecycleHook",
    "LifecycleHookRegistry",
    "ReplayEvent",
    "ResumeEvent",
]
