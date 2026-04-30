"""Tests for the kernel event bus.

The bus is small but its contract is load-bearing for every extension:
- serial dispatch order per channel
- exceptions in handlers are isolated
- async + sync handlers interoperate
- emit returns handler return values for cancel/replace decisions
"""

from __future__ import annotations

import pytest

from agentm.core.kernel.events import EventBus


@pytest.mark.asyncio
async def test_handlers_run_in_registration_order() -> None:
    """The loop's mutation contract requires later handlers to observe earlier
    handlers' mutations — that's only safe if dispatch is serial and ordered."""

    bus = EventBus()
    log: list[str] = []

    bus.on("ch", lambda e: log.append("a"))
    bus.on("ch", lambda e: log.append("b"))
    bus.on("ch", lambda e: log.append("c"))

    await bus.emit("ch", object())
    assert log == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_unsubscribe_removes_handler() -> None:
    """Calling the returned unsubscribe must stop further deliveries."""

    bus = EventBus()
    calls: list[int] = []
    unsub = bus.on("ch", lambda e: calls.append(1))
    await bus.emit("ch", None)
    unsub()
    await bus.emit("ch", None)
    assert calls == [1]
    # second unsubscribe must be a no-op
    unsub()


@pytest.mark.asyncio
async def test_async_handlers_are_awaited() -> None:
    """Coroutine handlers must run to completion before emit returns; that's
    the only way mutation order across async handlers can be deterministic."""

    bus = EventBus()
    out: list[str] = []

    async def handler_a(_: object) -> None:
        out.append("a")

    def handler_b(_: object) -> None:
        out.append("b")

    bus.on("ch", handler_a)
    bus.on("ch", handler_b)
    await bus.emit("ch", None)
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_handler_exception_does_not_break_chain() -> None:
    """One buggy extension must not silence its peers. This is the resilience
    invariant cited in the design doc §3.5."""

    bus = EventBus()
    fired: list[str] = []

    bus.on("ch", lambda e: fired.append("a"))

    def boom(_: object) -> None:
        fired.append("b-raise")
        raise ValueError("bang")

    bus.on("ch", boom)
    bus.on("ch", lambda e: fired.append("c"))

    results = await bus.emit("ch", None)
    assert fired == ["a", "b-raise", "c"]
    # The exception slot must be None (so callers collecting returns won't
    # confuse "no opinion" with "I crashed").
    assert results[1] is None


@pytest.mark.asyncio
async def test_emit_returns_handler_return_values() -> None:
    """The cancel/replace contract relies on emit reporting per-handler
    returns in order so a collector can pick the last non-None."""

    bus = EventBus()
    bus.on("ch", lambda e: None)
    bus.on("ch", lambda e: {"block": True})
    bus.on("ch", lambda e: {"block": False})

    results = await bus.emit("ch", None)
    assert results == [None, {"block": True}, {"block": False}]


@pytest.mark.asyncio
async def test_clear_removes_all_subscriptions() -> None:
    """`clear()` is what the harness calls between sessions; if it leaks
    handlers the next session inherits stale state."""

    bus = EventBus()
    fired: list[int] = []
    bus.on("a", lambda e: fired.append(1))
    bus.on("b", lambda e: fired.append(2))
    bus.clear()
    await bus.emit("a", None)
    await bus.emit("b", None)
    assert fired == []
