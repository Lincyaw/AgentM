"""Fail-stop: the per-peer ordered send queue (§2.6).

This queue *is* the gateway's delivery-ordering guarantee — durable and
ephemeral frames share one FIFO so nothing overtakes anything else — and
its bounding rule (shed oldest ephemeral, never a durable) is what keeps a
slow peer from either losing the reliability floor or exhausting memory.
If either invariant breaks, the ordering fix and the at-least-once floor
silently regress.
"""

from __future__ import annotations

import asyncio

import pytest

from agentm.gateway.send_queue import SendQueue
from agentm.gateway.wire import WIRE_VERSION, Envelope


def _env(env_id: str) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=env_id,
        kind="outbound",
        ts=1.0,
        session_key="terminal:t1",
        body={"channel": "terminal", "chat_id": "t1", "content": env_id},
    )


@pytest.mark.asyncio
async def test_fifo_order_preserved_across_classes() -> None:
    """Durable and ephemeral interleave in one FIFO — delivery order is
    exactly enqueue order, regardless of class (the ordering guarantee)."""
    q = SendQueue(high_water=100)
    q.put(_env("a"), 1)  # durable
    q.put(_env("b"), None)  # ephemeral
    q.put(_env("c"), 2)  # durable
    out = [(await q.get()).envelope.id for _ in range(3)]
    assert out == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_get_blocks_until_put() -> None:
    """Single-consumer get() waits on an empty queue and wakes on put."""
    q = SendQueue()
    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)  # let it reach the wait
    assert not getter.done()
    q.put(_env("x"), None)
    item = await asyncio.wait_for(getter, timeout=1.0)
    assert item.envelope.id == "x"


@pytest.mark.asyncio
async def test_backpressure_sheds_oldest_ephemeral_keeps_durable_order() -> None:
    """Over high-water, the OLDEST ephemeral is dropped; durable frames are
    never dropped and surviving order is preserved."""
    q = SendQueue(high_water=3)
    q.put(_env("e1"), None)  # ephemeral (oldest)
    q.put(_env("d1"), 1)  # durable
    q.put(_env("e2"), None)  # ephemeral
    # 4th put crosses high_water(3) -> shed oldest ephemeral == e1.
    q.put(_env("d2"), 2)  # durable
    assert q.dropped_ephemeral == 1
    ids = [(await q.get()).envelope.id for _ in range(len(q))]
    assert ids == ["d1", "e2", "d2"]  # e1 gone, rest in order


@pytest.mark.asyncio
async def test_durable_never_dropped_even_when_all_durable() -> None:
    """An all-durable queue past high-water has nothing to shed — it grows
    rather than lose a durable frame (the reliability floor is absolute)."""
    q = SendQueue(high_water=2)
    for i in range(5):
        q.put(_env(f"d{i}"), i + 1)
    assert q.dropped_ephemeral == 0
    assert len(q) == 5
    ids = [(await q.get()).envelope.id for _ in range(5)]
    assert ids == ["d0", "d1", "d2", "d3", "d4"]
