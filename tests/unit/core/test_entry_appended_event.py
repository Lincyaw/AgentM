"""Fail-stop for ``EntryAppendedEvent`` — must fire exactly once per
``SessionView.append_entry`` call and carry the persisted entry's id.

Live-inspector / aegis-ui depend on this event firing on every entry write
(assistant message, ``llmharness.audit_event``, plan submission, …). If the
emit drifts off (e.g. a future refactor moves ``append_entry`` through a
different code path or forgets to fire post-persist), the UI silently goes
blind on entry-tree writes — exactly the failure mode this event was
introduced to prevent. Lock it down at the contract level.
"""

from __future__ import annotations

from agentm.core.abi import EventBus
from agentm.core.abi.events import EntryAppendedEvent
from agentm.core.runtime.session_helpers import SessionView
from agentm.core.runtime.session_manager import SessionManager


def _make_view(bus: EventBus) -> SessionView:
    sm = SessionManager(cwd="/tmp", persist=False)
    sm.new_session(id="sess-test")
    return SessionView(
        sm,
        loop_config_getter=lambda: None,  # type: ignore[arg-type, return-value]
        bus=bus,
    )


def test_entry_appended_event_fires_once_per_append() -> None:
    bus = EventBus()
    received: list[EntryAppendedEvent] = []

    def handler(event: EntryAppendedEvent) -> None:
        received.append(event)

    bus.on(EntryAppendedEvent.CHANNEL, handler)

    view = _make_view(bus)
    entry_id = view.append_entry("custom_kind", {"hello": "world"})

    assert len(received) == 1
    event = received[0]
    assert event.entry_id == entry_id
    assert event.entry_type == "custom_kind"
    assert event.parent_id is None  # first entry, no parent
    assert event.payload == {"hello": "world"}
    assert event.session_id == "sess-test"

    # Second append: parent_id should chain to the first.
    second_id = view.append_entry("custom_kind", {"k": 2})
    assert len(received) == 2
    assert received[1].entry_id == second_id
    assert received[1].parent_id == entry_id


def test_entry_appended_event_skipped_when_bus_absent() -> None:
    """``SessionView`` constructed without a bus must not crash on append."""

    sm = SessionManager(cwd="/tmp", persist=False)
    sm.new_session(id="sess-nobus")
    view = SessionView(
        sm,
        loop_config_getter=lambda: None,  # type: ignore[arg-type, return-value]
        bus=None,
    )
    # No assertion needed beyond "doesn't raise".
    view.append_entry("x", {})
