"""What a bus dispatches once segments are linked into it.

A segment is one context's own subscriptions, dispatched as part of a bus
because it is linked.  Every bus-wide operation therefore has to mean the same
thing for a linked segment's handlers as for the bus's own, or the operation's
name is a promise it does not keep.
"""

from __future__ import annotations

from agentm.core.abi.bus import EventBus


def test_clearing_a_bus_takes_its_linked_segments_with_it() -> None:
    """``clear()`` clears what the bus dispatches, not one table of it.

    A linked segment's handlers dispatch on this bus, so leaving them linked
    would leave behind exactly what clearing removes.  ``_force_clear`` at
    shutdown already unlinks them, which is what makes the difference an
    oversight rather than a distinction.
    """

    bus = EventBus()
    segment = bus.segment("agentm.probe")
    segment.on("probe.channel", lambda event: None)
    bus.link(segment)
    assert len(bus.subscriptions("probe.channel")) == 1

    bus.clear()

    assert bus.subscriptions("probe.channel") == []
    assert bus.channels() == []


def test_a_restore_does_not_reissue_a_sequence_number_still_in_use() -> None:
    """``replace_from`` restores tables, never the allocator.

    An install snapshots the bus, and its rollback restores the snapshot's
    tables.  Since the rollback stopped putting the link list back, a context
    that finished installing while the failing one was in flight is still
    linked afterwards — and its subscriptions still hold the sequence numbers
    they were issued.  Rewinding the allocator beneath them re-issues those
    numbers, two live subscriptions tie on ``(priority, seq)``, and the one
    subscribed later dispatches first.

    A sequence number nobody holds costs nothing, so the allocator only ever
    moves forward.
    """

    bus = EventBus()
    snapshot = bus.copy()

    segment = bus.segment("agentm.late")
    order: list[str] = []
    segment.on("probe.channel", lambda event: order.append("segment"))
    bus.link(segment)

    bus.replace_from(snapshot)

    bus.on("probe.channel", lambda event: order.append("host"))
    bus.emit_sync("probe.channel", object())

    assert order == ["segment", "host"]
    seqs = [sub.seq for sub in bus.subscriptions("probe.channel")]
    assert len(set(seqs)) == len(seqs)
