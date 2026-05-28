"""Load-bearing invariants of the generic background-task registry.

Fail-stop position: ``BackgroundTaskRegistry`` is the shared substrate every
background-running atom (``sub_agent`` today, ``background_exec`` next) sits on.
If slot accounting stops refusing at capacity, an atom can oversubscribe
``max_workers``; if first-completed polling never returns, a poll-style tool
hangs the turn; if ``cancel`` stops setting the abort signal, a detached unit
becomes uncancellable. These three are the only behaviours the registry is
responsible for — the owning atom drives status and result handling itself.
"""

from __future__ import annotations

import asyncio

import pytest

from agentm.core.lib.background_tasks import (
    RUNNING,
    BackgroundTask,
    BackgroundTaskRegistry,
    SlotLimitReached,
)


def _make_task(task_id: str, coro: asyncio.Future[None] | None = None) -> BackgroundTask:
    async def _idle() -> None:
        await asyncio.Event().wait()  # never completes on its own

    return BackgroundTask(
        task_id=task_id,
        task=asyncio.ensure_future(coro if coro is not None else _idle()),
        abort_signal=asyncio.Event(),
    )


@pytest.mark.asyncio
async def test_slot_limit_refuses_at_capacity() -> None:
    registry: BackgroundTaskRegistry[BackgroundTask] = BackgroundTaskRegistry(
        max_workers=2
    )
    # Two reservations fill the cap (running + reserved == max_workers).
    await registry.reserve_slot()
    await registry.register(_make_task("a"))
    await registry.reserve_slot()
    with pytest.raises(SlotLimitReached):
        await registry.reserve_slot()
    # Registering the second consumes its reservation; still at cap.
    await registry.register(_make_task("b"))
    with pytest.raises(SlotLimitReached):
        await registry.reserve_slot()
    # A terminal unit no longer counts as running, freeing a real slot...
    registry.get("a").status = "done"  # type: ignore[union-attr]
    await registry.reserve_slot()
    # ...and a reservation released without a register (failed spawn) frees it
    # again, so the next dispatch can take it.
    await registry.release_slot()
    await registry.reserve_slot()

    for handle in registry.values():
        handle.task.cancel()


@pytest.mark.asyncio
async def test_poll_first_completed_returns_when_one_finishes() -> None:
    registry: BackgroundTaskRegistry[BackgroundTask] = BackgroundTaskRegistry(
        max_workers=4
    )

    async def _quick() -> None:
        return None

    quick = _make_task("quick", asyncio.ensure_future(_quick()))
    slow = _make_task("slow")
    await registry.reserve_slot()
    await registry.register(quick)
    await registry.reserve_slot()
    await registry.register(slow)

    # Returns as soon as ``quick`` finishes, without waiting on ``slow``.
    await asyncio.wait_for(registry.poll_first_completed(), timeout=1.0)
    assert quick.task.done()
    assert not slow.task.done()

    slow.task.cancel()


@pytest.mark.asyncio
async def test_cancel_sets_abort_signal() -> None:
    registry: BackgroundTaskRegistry[BackgroundTask] = BackgroundTaskRegistry(
        max_workers=4
    )
    handle = _make_task("c")
    await registry.reserve_slot()
    await registry.register(handle)

    assert not handle.abort_signal.is_set()
    assert await registry.cancel("c") is True
    assert handle.abort_signal.is_set()

    # Unknown id and already-terminal units both refuse the cancel.
    assert await registry.cancel("missing") is False
    handle.status = "aborted"
    assert await registry.cancel("c") is False

    handle.task.cancel()


@pytest.mark.asyncio
async def test_wait_one_skips_unknown_and_terminal() -> None:
    registry: BackgroundTaskRegistry[BackgroundTask] = BackgroundTaskRegistry(
        max_workers=4
    )

    async def _quick() -> None:
        return None

    done = _make_task("done", asyncio.ensure_future(_quick()))
    await registry.reserve_slot()
    await registry.register(done)
    await asyncio.wait_for(registry.poll_first_completed(), timeout=1.0)
    done.status = "completed"

    # Already terminal → returns immediately, does not block on a fresh await.
    await asyncio.wait_for(registry.wait_one("done"), timeout=1.0)
    # Unknown id → returns immediately rather than raising or hanging.
    await asyncio.wait_for(registry.wait_one("missing"), timeout=1.0)


def test_registry_only_counts_running_for_slots() -> None:
    # ``RUNNING`` is the one status that occupies a slot; this is the contract
    # owners rely on when they flip a handle to a terminal value.
    assert RUNNING == "running"


@pytest.mark.asyncio
async def test_register_without_reservation_never_drives_counter_negative() -> None:
    """The reservation counter holds ``>= 0`` for both owner flows.

    ``background_exec`` registers handles whose work is ALREADY running, so it
    never reserves first. Without the clamp, each such ``register`` would
    decrement ``_reserved_slots`` below zero; here it must stay pinned at zero.
    The sub_agent ``reserve_slot → register`` net must remain unchanged (a
    reservation taken before a register cancels out exactly).
    """

    registry: BackgroundTaskRegistry[BackgroundTask] = BackgroundTaskRegistry(
        max_workers=1_000_000
    )

    # No-reservation registers (background_exec flow): counter stays at 0.
    for i in range(3):
        await registry.register(_make_task(f"bg{i}"))
        assert registry._reserved_slots >= 0
    assert registry._reserved_slots == 0

    # Mixed in a reserve→register pair (sub_agent flow): still nets to 0.
    await registry.reserve_slot()
    assert registry._reserved_slots == 1
    await registry.register(_make_task("reserved"))
    assert registry._reserved_slots == 0

    for handle in registry.values():
        handle.task.cancel()
