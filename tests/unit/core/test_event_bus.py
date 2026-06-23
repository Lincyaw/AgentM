"""Event bus: priority dispatch, error isolation, channel separation."""

from __future__ import annotations

from typing import Any

import pytest

from agentm.core.abi.bus import EventBus
from agentm.core.abi.events import BusPriority, Event


@pytest.mark.asyncio
async def test_priority_ordering() -> None:
    bus = EventBus()
    order: list[str] = []

    def pre_handler(event: Any) -> None:
        order.append("PRE")

    def normal_handler(event: Any) -> None:
        order.append("NORMAL")

    def post_handler(event: Any) -> None:
        order.append("POST")

    bus.on("ch", pre_handler, priority=BusPriority.PRE)
    bus.on("ch", normal_handler, priority=BusPriority.NORMAL)
    bus.on("ch", post_handler, priority=BusPriority.POST)

    await bus.emit("ch", Event())
    assert order == ["PRE", "NORMAL", "POST"]


@pytest.mark.asyncio
async def test_fifo_within_same_priority() -> None:
    bus = EventBus()
    order: list[str] = []

    def handler_a(event: Any) -> None:
        order.append("a")

    def handler_b(event: Any) -> None:
        order.append("b")

    bus.on("ch", handler_a, priority=BusPriority.NORMAL)
    bus.on("ch", handler_b, priority=BusPriority.NORMAL)

    await bus.emit("ch", Event())
    assert order == ["a", "b"]


@pytest.mark.asyncio
async def test_handler_exception_does_not_block_others() -> None:
    bus = EventBus()
    call_log: list[str] = []

    def handler_1(event: Any) -> None:
        call_log.append("before")

    def handler_2(event: Any) -> None:
        raise RuntimeError("boom")

    def handler_3(event: Any) -> None:
        call_log.append("after")

    bus.on("ch", handler_1, priority=BusPriority.NORMAL)
    bus.on("ch", handler_2, priority=BusPriority.NORMAL)
    bus.on("ch", handler_3, priority=BusPriority.NORMAL)

    results = await bus.emit("ch", Event())
    assert call_log == ["before", "after"]
    assert results[1] is None


@pytest.mark.asyncio
async def test_channel_isolation() -> None:
    bus = EventBus()
    called_a = False
    called_b = False

    def handler_a(event: Any) -> None:
        nonlocal called_a
        called_a = True

    def handler_b(event: Any) -> None:
        nonlocal called_b
        called_b = True

    bus.on("channel_a", handler_a)
    bus.on("channel_b", handler_b)

    await bus.emit("channel_a", Event())
    assert called_a is True
    assert called_b is False


@pytest.mark.asyncio
async def test_emit_returns_handler_values() -> None:
    bus = EventBus()

    def handler_int(event: Any) -> int:
        return 42

    def handler_str(event: Any) -> str:
        return "hello"

    bus.on("ch", handler_int)
    bus.on("ch", handler_str)

    results = await bus.emit("ch", Event())
    assert results == [42, "hello"]


@pytest.mark.asyncio
async def test_unsubscribe() -> None:
    bus = EventBus()
    called = False

    def handler(event: Any) -> None:
        nonlocal called
        called = True

    unsub = bus.on("ch", handler)
    unsub()

    await bus.emit("ch", Event())
    assert called is False


@pytest.mark.asyncio
async def test_unsubscribe_is_idempotent() -> None:
    bus = EventBus()

    def handler(event: Any) -> None:
        pass

    unsub = bus.on("ch", handler)
    unsub()
    unsub()  # second call must not raise


@pytest.mark.asyncio
async def test_emit_on_empty_channel() -> None:
    bus = EventBus()
    results = await bus.emit("nonexistent", Event())
    assert results == []


@pytest.mark.asyncio
async def test_dispatch_id_assigned_on_emit() -> None:
    bus = EventBus()
    event = Event()
    original_id = event.dispatch_id

    # At least one handler must exist so emit reaches the dispatch_id
    # assignment (the bus short-circuits on empty channels).
    bus.on("ch", lambda e: None)
    await bus.emit("ch", event)

    assert event.dispatch_id
    assert isinstance(event.dispatch_id, str)
    assert len(event.dispatch_id) == 32  # uuid4 hex length
    # emit overwrites the construction-time default
    assert event.dispatch_id != original_id
