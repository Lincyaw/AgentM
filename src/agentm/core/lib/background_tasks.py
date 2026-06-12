"""Generic registry for asyncio background units.

This is the substrate shared by every atom that runs work in a detached
``asyncio.Task`` while the agent keeps taking turns. It was extracted out of
``extensions/builtin/sub_agent.py`` (the original prototype, where the unit is
a child :class:`AgentSession`) so a future ``background_exec`` atom — whose
unit is a single tool coroutine — can sit on the same machinery without either
atom importing the other (forbids atom→atom imports; ``core.lib`` is the
non-atom seam they share).

Generalization boundary: this module owns only what is generic to *any*
asyncio background unit — the task handle, a free-text status string, an abort
signal, a read flag, the registry dict + lock, slot accounting against
``max_workers``, and first-completed / wait-one polling. Everything specific
to a unit (how it is spawned, what its result means, how completion is
rendered for the model) stays in the owning atom, which subclasses
:class:`BackgroundTask` to carry its own fields and drives status transitions
itself. The registry never sets ``status`` — it only reads it for slot
accounting and poll filtering — and never spawns work; the owner creates the
:class:`asyncio.Task` and calls :meth:`BackgroundTaskRegistry.register`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

# Free-text status, owner-defined. The registry treats ``"running"`` as the
# one status that occupies a worker slot and is eligible for polling; every
# other value is considered terminal for accounting purposes. Owners pick
# their own richer vocabulary (sub_agent uses completed/aborted/error).
RUNNING = "running"


@dataclass(slots=True, kw_only=True)
class BackgroundTask:
    """Handle for one detached :class:`asyncio.Task`.

    Owners subclass this to attach unit-specific state (sub_agent's
    ``_ChildTask`` adds the child session, pending instructions, artifacts,
    etc.). The registry only ever touches the fields declared here:
    ``task_id`` for keying, ``task`` for awaiting/polling, ``abort_signal``
    for cancellation, ``status`` for slot accounting and poll filtering, and
    ``read`` as a producer-managed delivery flag the registry leaves alone but
    that lives here because it is generic to "has the owner surfaced this unit
    to the agent yet".
    """

    task_id: str
    task: asyncio.Task[Any]
    abort_signal: asyncio.Event
    status: str = RUNNING
    read: bool = False


T = TypeVar("T", bound=BackgroundTask)


class SlotLimitReached(RuntimeError):
    """Raised by :meth:`BackgroundTaskRegistry.reserve_slot` when the registry
    is already at ``max_workers`` running + reserved units."""


class BackgroundTaskRegistry(Generic[T]):
    """Async-safe registry of :class:`BackgroundTask` handles with slot caps.

    The lock guards the registry dict and the reserved-slot counter. The
    owner's flow is: :meth:`reserve_slot` (fails fast at capacity) → create the
    unit + its :class:`asyncio.Task` → :meth:`register` (which releases the
    reservation and inserts the handle). If creation fails between the two,
    the owner calls :meth:`release_slot` to undo the reservation. This mirrors
    sub_agent's original ``_reserved_slots`` bookkeeping exactly, so a child
    that is mid-spawn still counts against ``max_workers``.

    Pass ``max_workers=None`` for the unbounded contract: slot accounting is
    skipped entirely, :meth:`reserve_slot` / :meth:`release_slot` short-circuit,
    and :meth:`register` inserts without touching the (unused) counter. Owners
    that already have the work running before they register — and therefore
    have nothing to fail-fast on — use this mode (``background_exec``).
    """

    def __init__(self, *, max_workers: int | None) -> None:
        self._max_workers = max_workers
        self._tasks: dict[str, T] = {}
        self._lock = asyncio.Lock()
        self._reserved_slots = 0

    @property
    def is_unbounded(self) -> bool:
        """True when constructed with ``max_workers=None`` — slot accounting
        is a no-op. Exposed so owners can branch on it without reaching into
        ``_max_workers``."""

        return self._max_workers is None

    @property
    def lock(self) -> asyncio.Lock:
        """The registry lock, exposed so owners can extend a critical section
        over their own per-handle state (e.g. draining pending instructions)
        under the same mutual exclusion the registry uses."""

        return self._lock

    def get(self, task_id: str) -> T | None:
        """Return the handle for ``task_id`` or ``None``. Call under
        :attr:`lock` when racing against :meth:`register`/:meth:`remove`."""

        return self._tasks.get(task_id)

    def values(self) -> list[T]:
        """Snapshot of all registered handles. Call under :attr:`lock`."""

        return list(self._tasks.values())

    async def reserve_slot(self) -> None:
        """Claim a worker slot, raising :class:`SlotLimitReached` if running +
        reserved units already meet ``max_workers``.

        Counts handles whose status is :data:`RUNNING` plus outstanding
        reservations, so concurrent dispatches cannot oversubscribe between
        reserve and register. In the unbounded contract
        (``max_workers is None``) this is a no-op — there is no capacity to
        check against.
        """

        if self._max_workers is None:
            return
        async with self._lock:
            running = sum(1 for t in self._tasks.values() if t.status == RUNNING)
            if running + self._reserved_slots >= self._max_workers:
                raise SlotLimitReached(
                    f"max_workers limit reached ({self._max_workers})"
                )
            self._reserved_slots += 1

    async def release_slot(self) -> None:
        """Release a reservation taken by :meth:`reserve_slot` without
        registering a handle (the unit failed to spawn). No-op in the unbounded
        contract (``max_workers is None``)."""

        if self._max_workers is None:
            return
        async with self._lock:
            self._reserved_slots -= 1

    async def register(self, task: T) -> None:
        """Insert ``task`` and release the reservation that preceded it in one
        critical section (a reserved slot becomes a live running slot).

        Bounded owners (``max_workers=N``) follow the documented
        ``reserve_slot → create task → register`` flow (sub_agent), so the
        decrement here cancels out the reservation. Unbounded owners
        (``max_workers=None`` — background_exec) never reserve in the first
        place and skip the counter entirely.
        """

        async with self._lock:
            if self._max_workers is not None:
                self._reserved_slots -= 1
            self._tasks[task.task_id] = task

    async def poll_first_completed(self) -> None:
        """Block until at least one running unit reaches a terminal state.

        Returns immediately when nothing is running. The wait is over the
        underlying :class:`asyncio.Task`s; the owner re-reads ``status`` after
        this returns. Equivalent to sub_agent's ``check_tasks`` poll.
        """

        async with self._lock:
            running = [t.task for t in self._tasks.values() if t.status == RUNNING]
        if running:
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

    async def wait_one(self, task_id: str) -> None:
        """Await the single unit ``task_id`` if it is still running.

        Snapshots the handle under the lock, then awaits its task outside the
        lock so other registry operations are not blocked for the duration.
        Silently returns if the id is unknown or already terminal.
        """

        async with self._lock:
            handle = self._tasks.get(task_id)
            if handle is None or handle.status != RUNNING:
                return
            task = handle.task
        await task

    async def cancel(self, task_id: str) -> bool:
        """Signal abort on ``task_id`` and return whether the signal was set.

        Sets the handle's ``abort_signal`` (the cooperative-cancellation
        mechanism every background unit honours). Returns ``False`` if the id
        is unknown or the unit is already terminal — the owner turns that into
        its own error message. The status flip to a terminal value is the
        unit's job once it observes the signal, not the registry's.
        """

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
