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


from agentm.core.abi import Event, EventBus


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












