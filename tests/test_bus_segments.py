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


def test_dispatch_order_follows_rank_before_subscription_order() -> None:
    """Reinstalling one atom must not move its handlers to the end.

    Cross-segment order used to be ``(priority, seq)``, so a handler's place
    was decided by when its atom happened to subscribe. Reinstalling an atom
    minted fresh sequence numbers and sent its handlers to the back of their
    band -- and anything that cared had to reinstall every atom after it to put
    the order back, which is exactly what ``atom_watch`` did on every scenario
    change.

    ``(priority, rank, seq)`` makes the place a property of the declarations.
    Seq still decides within one segment, which is what it is good for: two
    handlers from one atom go in the order that atom subscribed them.
    """

    bus = EventBus()
    order: list[str] = []
    shallow = bus.segment("shallow")
    deep = bus.segment("deep")
    deep.rank = 1
    shallow.on("probe.channel", lambda event: order.append("shallow"))
    deep.on("probe.channel", lambda event: order.append("deep"))
    bus.link(shallow)
    bus.link(deep)

    bus.emit_sync("probe.channel", object())
    assert order == ["shallow", "deep"]

    # The shallow atom is reinstalled: a fresh segment, the newest sequence
    # numbers on the bus, and the same rank.
    order.clear()
    bus.unlink(shallow)
    replacement = bus.segment("shallow")
    replacement.on("probe.channel", lambda event: order.append("shallow"))
    bus.link(replacement)

    bus.emit_sync("probe.channel", object())
    assert order == ["shallow", "deep"]
