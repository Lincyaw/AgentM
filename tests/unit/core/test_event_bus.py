"""Event bus: priority dispatch, error isolation, channel separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from agentm.core.abi.bus import EventBus
from agentm.core.abi.events import BusPriority, Event, HookContract


@dataclass(slots=True)
class _MutableEvent(Event):
    CHANNEL: ClassVar[str] = "mutable"
    HOOK: ClassVar[HookContract] = HookContract(
        mutation_contract="event.items may be mutated",
        mutable_fields=("items",),
    )
    items: list[str]
    label: str


@dataclass(slots=True)
class _ReadonlyEvent(Event):
    CHANNEL: ClassVar[str] = "readonly"
    HOOK: ClassVar[HookContract] = HookContract()
    items: list[str]


@dataclass(slots=True)
class _OpaquePayload:
    value: object


@dataclass(slots=True)
class _OpaqueEvent(Event):
    CHANNEL: ClassVar[str] = "opaque"
    HOOK: ClassVar[HookContract] = HookContract()
    payload: _OpaquePayload


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


def test_unsubscribe_removes_the_exact_duplicate_registration() -> None:
    bus = EventBus()

    def handler(event: Any) -> None:
        pass

    first_unsubscribe = bus.on("ch", handler)
    first = bus.subscriptions_for("ch")[0]
    second_unsubscribe = bus.on("ch", handler)
    second = bus.subscriptions_for("ch")[1]

    second_unsubscribe()

    assert bus.subscriptions_for("ch") == [first]
    assert first is not second
    first_unsubscribe()
    assert bus.subscriptions_for("ch") == []


@pytest.mark.asyncio
async def test_emit_on_empty_channel() -> None:
    bus = EventBus()
    event = Event()
    original_id = event.dispatch_id

    results = await bus.emit("nonexistent", event)

    assert results == []
    assert event.dispatch_id != original_id


@pytest.mark.asyncio
async def test_clear_invalidates_cached_handlers() -> None:
    bus = EventBus()
    calls = 0

    def handler(event: Any) -> None:
        nonlocal calls
        calls += 1

    bus.on("ch", handler)
    await bus.emit("ch", Event())
    bus.clear()
    await bus.emit("ch", Event())

    assert calls == 1


@pytest.mark.asyncio
async def test_dispatch_id_assigned_on_emit() -> None:
    bus = EventBus()
    event = Event()
    original_id = event.dispatch_id

    bus.on("ch", lambda e: None)
    await bus.emit("ch", event)

    assert event.dispatch_id
    assert isinstance(event.dispatch_id, str)
    assert len(event.dispatch_id) == 32  # uuid4 hex length
    # emit overwrites the construction-time default
    assert event.dispatch_id != original_id


@pytest.mark.asyncio
async def test_strict_event_mutations_allow_declared_fields() -> None:
    bus = EventBus()
    event = _MutableEvent(items=[], label="stable")
    bus.on(event.CHANNEL, lambda current: current.items.append("ok"))

    await bus.emit(event.CHANNEL, event)

    assert event.items == ["ok"]


@pytest.mark.asyncio
async def test_strict_event_mutations_reject_undeclared_fields() -> None:
    bus = EventBus()
    event = _MutableEvent(items=[], label="stable")

    def mutate_readonly(current: _MutableEvent) -> None:
        current.label = "changed"

    bus.on(event.CHANNEL, mutate_readonly)

    with pytest.raises(RuntimeError, match="undeclared readonly fields"):
        await bus.emit(event.CHANNEL, event)
    assert event.label == "stable"


@pytest.mark.asyncio
async def test_strict_event_mutations_reject_observation_event_changes() -> None:
    bus = EventBus()
    event = _ReadonlyEvent(items=["stable"])
    bus.on(event.CHANNEL, lambda current: current.items.append("changed"))

    with pytest.raises(RuntimeError, match="undeclared readonly fields"):
        await bus.emit(event.CHANNEL, event)
    assert event.items == ["stable"]


@pytest.mark.asyncio
async def test_readonly_snapshot_does_not_compare_opaque_deepcopy_identity() -> None:
    bus = EventBus()
    event = _OpaqueEvent(payload=_OpaquePayload(value=object()))
    bus.on(event.CHANNEL, lambda _current: None)

    await bus.emit(event.CHANNEL, event)


def test_strict_event_mutations_apply_to_sync_dispatch() -> None:
    bus = EventBus()
    event = _MutableEvent(items=[], label="stable")

    def mutate_readonly(current: _MutableEvent) -> None:
        current.label = "changed"

    bus.on(event.CHANNEL, mutate_readonly)

    with pytest.raises(RuntimeError, match="undeclared readonly fields"):
        bus.emit_sync(event.CHANNEL, event)
    assert event.label == "stable"
