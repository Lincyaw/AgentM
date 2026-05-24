"""Fail-stop tests for ``EventBus`` dispatch_id.

The bus-owned dispatch_id is the join key that lets a single
``agentm.event.dispatch`` log record be correlated with every
``agentm.handler.invoke`` record it fanned out into. Three properties must
hold:

1. **Stable within one dispatch.** Every handler invoked on a single
   ``emit`` / ``emit_sync`` call sees the same id via
   ``bus.current_dispatch_id()``. Wrong → the join key is per-handler, not
   per-dispatch, and consumers cannot recover the fanout.
2. **Fresh across dispatches.** Re-emitting the same channel / event
   produces a different id. Wrong → re-dispatches of the same event
   instance look like one combined dispatch to downstream readers, double-
   counting handler runs.
3. **Nested-dispatch isolation.** A handler that re-emits on the same bus
   sees its own nested id during the inner dispatch; once the inner emit
   returns, the outer id is restored. Wrong → the outer dispatch's
   handler.invoke records get stamped with a leaked inner id and the
   correlation is silently wrong (the worst failure mode — no test fires).

We do not assert on the *exact* format of the id (uuid4 hex is an
implementation detail); we assert on the structural invariants the
observability sink depends on.
"""

from __future__ import annotations

import pytest

from agentm.core.abi.events import EventBus


def test_dispatch_id_is_stable_within_one_emit_sync() -> None:
    bus = EventBus()
    observed: list[str | None] = []

    def handler_a(_event: object) -> None:
        observed.append(bus.current_dispatch_id())

    def handler_b(_event: object) -> None:
        observed.append(bus.current_dispatch_id())

    bus.on("ch", handler_a)
    bus.on("ch", handler_b)
    bus.emit_sync("ch", {"x": 1})

    assert len(observed) == 2
    assert observed[0] is not None
    assert observed[0] == observed[1], (
        "every handler on a single dispatch must see the same dispatch_id"
    )


@pytest.mark.asyncio
async def test_dispatch_id_is_stable_within_one_async_emit() -> None:
    bus = EventBus()
    observed: list[str | None] = []

    async def handler_a(_event: object) -> None:
        observed.append(bus.current_dispatch_id())

    def handler_b(_event: object) -> None:
        observed.append(bus.current_dispatch_id())

    bus.on("ch", handler_a)
    bus.on("ch", handler_b)
    await bus.emit("ch", {"x": 1})

    assert len(observed) == 2
    assert observed[0] is not None
    assert observed[0] == observed[1]


def test_dispatch_id_is_fresh_across_emits() -> None:
    bus = EventBus()
    seen: list[str | None] = []

    def handler(_event: object) -> None:
        seen.append(bus.current_dispatch_id())

    bus.on("ch", handler)
    payload = {"same": "instance"}
    bus.emit_sync("ch", payload)
    bus.emit_sync("ch", payload)

    assert len(seen) == 2
    assert seen[0] is not None and seen[1] is not None
    assert seen[0] != seen[1], (
        "re-dispatching the same event instance must produce a distinct id"
    )


def test_dispatch_id_is_none_outside_any_emit() -> None:
    bus = EventBus()
    assert bus.current_dispatch_id() is None


def test_dispatch_id_stack_unwinds_on_nested_emit() -> None:
    bus = EventBus()
    captured: dict[str, str | None] = {}

    def inner(_event: object) -> None:
        captured["inner"] = bus.current_dispatch_id()

    def outer(_event: object) -> None:
        captured["outer_before"] = bus.current_dispatch_id()
        bus.emit_sync("inner", {})
        captured["outer_after"] = bus.current_dispatch_id()

    bus.on("inner", inner)
    bus.on("outer", outer)
    bus.emit_sync("outer", {})

    assert captured["outer_before"] is not None
    assert captured["inner"] is not None
    assert captured["outer_before"] != captured["inner"], (
        "nested emit must produce its own dispatch_id, not reuse the outer's"
    )
    assert captured["outer_after"] == captured["outer_before"], (
        "outer dispatch_id must be restored after the nested emit returns"
    )
    # And once the outer emit has returned, the stack is empty again.
    assert bus.current_dispatch_id() is None


def test_dispatch_id_stack_unwinds_when_handler_raises() -> None:
    """A handler raising ``Exception`` is logged-and-swallowed by the bus,
    but the stack must still pop. Otherwise a buggy handler permanently
    pollutes the bus's dispatch state for every subsequent emit.
    """
    bus = EventBus()

    def boom(_event: object) -> None:
        raise RuntimeError("intentional")

    bus.on("ch", boom)
    bus.emit_sync("ch", {})

    assert bus.current_dispatch_id() is None, (
        "handler exceptions must not leave a dangling dispatch_id on the stack"
    )
