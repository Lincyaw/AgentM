"""Per-peer ordered send queue (§2.6 unified delivery channel).

All outbound to a peer — durable *and* ephemeral — flows through one
in-memory FIFO consumed by a single sender task that writes the socket in
order. The queue *is* the ordering guarantee: enqueue order == event order
== delivery order, so the receiver never has to reorder (no wire sequence
number needed in a single-process gateway).

Reliability is orthogonal and lives elsewhere: a durable item carries its
``outbox_id`` (the row already persisted by ``OutboxStore.enqueue``) and is
acked only after a successful write; an ephemeral item carries
``outbox_id is None`` and is never persisted.

Bounding: when the queue grows past ``high_water`` the oldest *ephemeral*
item is dropped (ephemeral is best-effort live decoration). Durable items
are never dropped — they are already on disk and replay on reconnect.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from agentm.gateway.wire import Envelope


@dataclass(frozen=True, slots=True)
class SendItem:
    """One queued outbound frame.

    ``outbox_id`` is the durable row id to ack after a successful write, or
    ``None`` for an ephemeral frame (best-effort, not persisted, droppable).
    """

    envelope: Envelope
    outbox_id: int | None


class SendQueue:
    """Single-consumer FIFO with selective drop of oldest ephemeral.

    Not thread-safe by design — every caller runs on the gateway event
    loop. ``put`` is sync (producers never block); ``get`` is the single
    sender task awaiting the next frame.
    """

    def __init__(self, high_water: int = 1000) -> None:
        self._items: deque[SendItem] = deque()
        self._event = asyncio.Event()
        self._high_water = high_water
        # Observability: total ephemeral frames shed under backpressure.
        self.dropped_ephemeral = 0

    def put(self, envelope: Envelope, outbox_id: int | None) -> None:
        self._items.append(SendItem(envelope, outbox_id))
        if len(self._items) > self._high_water:
            self._drop_oldest_ephemeral()
        self._event.set()

    def _drop_oldest_ephemeral(self) -> None:
        """Drop the oldest ephemeral item; durable items are never dropped.

        O(n) scan, but only on the over-high-water path. If every queued
        item is durable there is nothing to shed and the queue grows — that
        is the correct outcome (durable must not be lost).
        """
        for i, item in enumerate(self._items):
            if item.outbox_id is None:
                del self._items[i]
                self.dropped_ephemeral += 1
                return

    async def get(self) -> SendItem:
        # Single-consumer wait: clear-then-check guards the put-between-
        # clear-and-wait race (a put that set() the event makes wait()
        # return immediately, then the non-empty check pops it).
        while not self._items:
            self._event.clear()
            await self._event.wait()
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)


__all__ = ["SendItem", "SendQueue"]
