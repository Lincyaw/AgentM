"""Generic registry for atom-owned asynchronous background work.

This helper is deliberately smaller than any concrete background atom. It owns
only the shared mechanics needed by detached work units: a task handle, a
mutable cancellation source, status-based slot accounting, and a registry lock.
Atoms keep all policy-specific behavior such as progress events, result
rendering, persistence, and cleanup.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

from agentm.core.abi.cancel import CancelSource

RUNNING = "running"


@dataclass(slots=True, kw_only=True)
class BackgroundTask:
    """Handle for one detached asyncio task.

    Owners may subclass this to attach domain-specific state. The registry only
    reads ``task_id``, ``task``, ``abort_signal``, ``status``, and ``read``.
    Any status other than ``RUNNING`` is considered terminal for slot
    accounting.
    """

    task_id: str
    task: asyncio.Task[None] | None = None
    abort_signal: CancelSource
    status: str = RUNNING
    read: bool = False


T = TypeVar("T", bound=BackgroundTask)


class SlotLimitReached(RuntimeError):
    """Raised when a bounded registry has no available worker slot."""


class BackgroundTaskRegistry(Generic[T]):
    """Async-safe registry of detached task handles with optional slot caps."""

    def __init__(self, *, max_workers: int | None) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1 or None")
        self._max_workers = max_workers
        self._tasks: dict[str, T] = {}
        self._lock = asyncio.Lock()
        self._reserved_slots = 0

    @property
    def is_unbounded(self) -> bool:
        return self._max_workers is None

    @property
    def lock(self) -> asyncio.Lock:
        """Registry lock for owners that need a wider critical section."""

        return self._lock

    def get(self, task_id: str) -> T | None:
        """Return a handle by id. Use under ``lock`` when racing mutations."""

        return self._tasks.get(task_id)

    def values(self) -> list[T]:
        """Return a snapshot of registered handles."""

        return list(self._tasks.values())

    async def reserve_slot(self) -> None:
        """Reserve capacity before creating a bounded background unit."""

        if self._max_workers is None:
            return
        async with self._lock:
            running = sum(1 for task in self._tasks.values() if task.status == RUNNING)
            if running + self._reserved_slots >= self._max_workers:
                raise SlotLimitReached(
                    f"max_workers limit reached ({self._max_workers})"
                )
            self._reserved_slots += 1

    async def release_slot(self) -> None:
        """Undo a reservation when task creation fails before registration."""

        if self._max_workers is None:
            return
        async with self._lock:
            if self._reserved_slots <= 0:
                raise RuntimeError("no reserved background task slot to release")
            self._reserved_slots -= 1

    async def register(self, task: T) -> None:
        """Register a task, consuming a prior reservation when bounded."""

        async with self._lock:
            if self._max_workers is not None:
                if self._reserved_slots <= 0:
                    raise RuntimeError(
                        "bounded background task registration requires reserve_slot()"
                    )
                self._reserved_slots -= 1
            self._tasks[task.task_id] = task

    async def cancel(self, task_id: str) -> bool:
        """Set a running task's cancellation source."""

        async with self._lock:
            handle = self._tasks.get(task_id)
            if handle is None or handle.status != RUNNING:
                return False
            handle.abort_signal.set()
            return True


__all__ = [
    "RUNNING",
    "BackgroundTask",
    "BackgroundTaskRegistry",
    "SlotLimitReached",
]
