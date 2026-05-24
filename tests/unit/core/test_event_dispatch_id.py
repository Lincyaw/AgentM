"""Fail-stop tests for the ``dispatch_id`` ABI field.

The :class:`Event` ABI carries a ``dispatch_id`` field; the :class:`EventBus`
reassigns it on every ``emit`` / ``emit_sync`` entry. That id is the join
key consumers use to correlate a single ``agentm.event.dispatch`` log
record with every ``agentm.handler.invoke`` record fanned out from it.
Four properties must hold:

1. **Stable within one dispatch.** Every handler invoked on a single
   ``emit`` / ``emit_sync`` call sees the same ``event.dispatch_id``.
   Wrong → the join key is per-handler, not per-dispatch, and consumers
   cannot recover the fanout.
2. **Fresh across dispatches.** Re-emitting the same Event instance
   produces a different id (the bus overwrites the field on entry).
   Wrong → re-dispatches of the same instance look like one combined
   dispatch to downstream readers, double-counting handler runs.
3. **Construction default.** An Event constructed and never emitted still
   has a valid (non-empty) ``dispatch_id`` from the field's default
   factory. Wrong → test fixtures and standalone constructions crash on
   attribute access.
4. **Nested-emit isolation.** A handler that emits a *different* Event
   instance on the same bus stamps the inner event with its own id; the
   outer event's id is unaffected because each event carries its own
   field. Wrong → nested emits collide on the shared join key.

We do not assert on the *exact* format of the id (uuid4 hex is an
implementation detail); we assert on the structural invariants the
observability sink depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from agentm.core.abi.events import Event, EventBus


@dataclass(slots=True)
class _Probe(Event):
    """Minimal Event subclass for these tests."""

    CHANNEL: ClassVar[str] = "probe"
    value: int = 0


def test_dispatch_id_is_stable_within_one_emit_sync() -> None:
    bus = EventBus()
    observed: list[str] = []

    def handler_a(event: Any) -> None:
        observed.append(event.dispatch_id)

    def handler_b(event: Any) -> None:
        observed.append(event.dispatch_id)

    bus.on("ch", handler_a)
    bus.on("ch", handler_b)
    bus.emit_sync("ch", _Probe(value=1))

    assert len(observed) == 2
    assert observed[0]
    assert observed[0] == observed[1], (
        "every handler on a single dispatch must see the same dispatch_id"
    )


@pytest.mark.asyncio
async def test_dispatch_id_is_stable_within_one_async_emit() -> None:
    bus = EventBus()
    observed: list[str] = []

    async def handler_a(event: Any) -> None:
        observed.append(event.dispatch_id)

    def handler_b(event: Any) -> None:
        observed.append(event.dispatch_id)

    bus.on("ch", handler_a)
    bus.on("ch", handler_b)
    await bus.emit("ch", _Probe(value=1))

    assert len(observed) == 2
    assert observed[0]
    assert observed[0] == observed[1]


def test_dispatch_id_is_fresh_across_emits_for_same_instance() -> None:
    """Re-emitting the same Event instance must produce a distinct id.
    The bus overwrites the field on entry; the field default only matters
    when an event is never emitted.
    """
    bus = EventBus()
    seen: list[str] = []

    def handler(event: Any) -> None:
        seen.append(event.dispatch_id)

    bus.on("ch", handler)
    event = _Probe(value=1)
    bus.emit_sync("ch", event)
    bus.emit_sync("ch", event)

    assert len(seen) == 2
    assert seen[0] and seen[1]
    assert seen[0] != seen[1], (
        "re-dispatching the same event instance must produce a distinct id"
    )


def test_event_has_default_dispatch_id_when_never_emitted() -> None:
    """A freshly constructed Event must already have a non-empty
    ``dispatch_id`` from the field default factory, so test fixtures and
    standalone constructions remain usable without going through the bus.
    """
    event = _Probe(value=1)
    assert event.dispatch_id, "default_factory must populate dispatch_id"
    # Two independently constructed events have independent default ids.
    other = _Probe(value=2)
    assert event.dispatch_id != other.dispatch_id


def test_nested_emit_uses_each_events_own_id() -> None:
    """A handler that emits a *different* event on the same bus stamps the
    inner event with its own fresh id; once the inner emit returns, the
    outer event's id is unchanged because each event owns its own field.
    """
    bus = EventBus()
    captured: dict[str, str] = {}

    inner_event = _Probe(value=99)
    outer_event = _Probe(value=1)

    def inner_handler(event: Any) -> None:
        captured["inner"] = event.dispatch_id

    def outer_handler(event: Any) -> None:
        captured["outer_before"] = event.dispatch_id
        bus.emit_sync("inner", inner_event)
        captured["outer_after"] = event.dispatch_id

    bus.on("inner", inner_handler)
    bus.on("outer", outer_handler)
    bus.emit_sync("outer", outer_event)

    assert captured["outer_before"]
    assert captured["inner"]
    assert captured["outer_before"] != captured["inner"], (
        "nested emit must stamp the inner event with its own dispatch_id"
    )
    assert captured["outer_after"] == captured["outer_before"], (
        "outer event's dispatch_id field is untouched by the nested emit"
    )


def test_dispatch_id_survives_handler_exception() -> None:
    """A handler raising ``Exception`` is logged-and-swallowed by the bus.
    The event's ``dispatch_id`` is set before any handler runs, so the
    surviving observer-side records still see the same id.
    """
    bus = EventBus()
    seen: list[str] = []

    def boom(_event: Any) -> None:
        raise RuntimeError("intentional")

    def after(event: Any) -> None:
        seen.append(event.dispatch_id)

    bus.on("ch", boom)
    bus.on("ch", after)
    event = _Probe(value=1)
    bus.emit_sync("ch", event)

    assert seen == [event.dispatch_id]
    assert event.dispatch_id


def test_non_event_payload_is_dispatched_unchanged() -> None:
    """Extensions may emit raw dict payloads on bespoke channels; the bus
    must not try to set ``dispatch_id`` on a non-Event and crash. The dict
    flows through untouched.
    """
    bus = EventBus()
    seen: list[Any] = []

    def handler(payload: Any) -> None:
        seen.append(payload)

    bus.on("ch", handler)
    payload = {"x": 1}
    bus.emit_sync("ch", payload)
    assert seen == [payload]
    assert "dispatch_id" not in payload
