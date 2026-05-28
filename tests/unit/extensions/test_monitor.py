"""Fail-stop coverage for the ``monitor`` atom (Session Inbox step 4).

Design: ``.claude/designs/session-inbox.md`` (``monitor`` section) +
``.claude/plans/2026-05-28-session-inbox.md`` (step 4). Same fail-stop
discipline as ``test_background_exec``: every test pins a load-bearing
position (the wakeup fire path, the channel dedup-replace, the cancel
isolation that does NOT touch the shared signal, the shutdown drain, the
stale-api guard, the rendering shape). No happy-path padding.

The atom is driven through the same lightweight fake ``ExtensionAPI`` stub
style as ``test_background_exec``, with a real :class:`SessionInbox` so
``post_inbox`` (dedup_key replace) semantics are exercised end-to-end.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from agentm.core.abi import TextContent, ToolResult
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.abi.extension import ExtensionAPI, ExtensionStaleError
from agentm.core.runtime.session_inbox import InboxItem, SessionInbox, render_item
from agentm.extensions.builtin import monitor
from agentm.extensions.builtin.monitor import _MonitorManager


class _FakeApi:
    """Minimal ExtensionAPI surface ``monitor`` touches.

    ``post_inbox`` delegates to a real :class:`SessionInbox`; ``on`` records
    handlers per channel and hands back a real ``Unsubscribe`` that drops the
    matching entry so the test can fire a channel by calling its handlers and
    verify subscribe/unsubscribe semantics.
    """

    def __init__(self) -> None:
        self.tools: list[Any] = []
        self.inbox = SessionInbox()
        self._handlers: dict[str, list[Any]] = {}

    def post_inbox(
        self, *, source: str, payload: Any, dedup_key: str | None = None
    ) -> None:
        self.inbox.push(
            InboxItem(source=source, payload=payload, dedup_key=dedup_key)
        )

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def on(self, channel: str, handler: Any, *, priority: int = 500) -> Any:
        bucket = self._handlers.setdefault(channel, [])
        bucket.append(handler)

        def _unsub() -> None:
            if handler in bucket:
                bucket.remove(handler)

        return _unsub

    def fire(self, channel: str, event: Any) -> None:
        """Test helper — invoke every subscribed handler for ``channel``."""

        for h in list(self._handlers.get(channel, [])):
            h(event)


def _manager(api: _FakeApi) -> _MonitorManager:
    return _MonitorManager(api=cast(ExtensionAPI, api))


def _payload(result: Any) -> dict[str, Any]:
    assert isinstance(result, ToolResult)
    text = "".join(b.text for b in result.content if isinstance(b, TextContent))
    return json.loads(text)


@pytest.mark.asyncio
async def test_schedule_wakeup_posts_inbox_item_after_delay() -> None:
    """After ``delay`` elapses the wakeup posts ONE ``source="monitor"`` inbox
    item with the expected payload kind/monitor_id/note and a
    ``monitor-wake-{id}`` dedup_key."""

    api = _FakeApi()
    mgr = _manager(api)
    res = await mgr.schedule_wakeup({"delay": 0.01, "note": "ping"})
    payload = _payload(res)
    monitor_id = payload["monitor_id"]
    assert payload["kind"] == "wakeup"
    assert payload["status"] == "pending"
    assert payload["delay"] == 0.01

    # Let the sleep elapse + give the task a chance to post.
    state = mgr._monitors[monitor_id]
    assert state.task is not None
    await state.task

    drained = api.inbox.drain()
    assert len(drained) == 1
    item = drained[0]
    assert item.source == "monitor"
    assert item.dedup_key == f"monitor-wake-{monitor_id}"
    assert item.payload == {
        "kind": "wakeup",
        "monitor_id": monitor_id,
        "note": "ping",
        "delay": 0.01,
    }
    assert state.status == "fired"


@pytest.mark.asyncio
async def test_create_monitor_channel_fire_pushes_inbox_item() -> None:
    """A channel fire posts a ``source="monitor"`` item with ``kind="channel"``,
    the channel name, the dedup_key, and a short event_summary."""

    api = _FakeApi()
    mgr = _manager(api)
    res = await mgr.create_monitor({"watch": "tool_call", "note": "n"})
    monitor_id = _payload(res)["monitor_id"]

    api.fire("tool_call", {"event": "first"})
    drained = api.inbox.drain()
    assert len(drained) == 1
    item = drained[0]
    assert item.source == "monitor"
    assert item.dedup_key == f"monitor-chan-{monitor_id}"
    assert isinstance(item.payload, dict)
    assert item.payload["kind"] == "channel"
    assert item.payload["channel"] == "tool_call"
    assert item.payload["monitor_id"] == monitor_id
    assert item.payload["note"] == "n"
    assert "first" in item.payload["event_summary"]


@pytest.mark.asyncio
async def test_channel_fires_dedup_replace_no_stacking() -> None:
    """Two channel fires in a row collapse to ONE undrained inbox item under
    the monitor's stable dedup_key — same discipline as the step-3 ticker so
    a stuck agent never finds a pile of stale fires."""

    api = _FakeApi()
    mgr = _manager(api)
    res = await mgr.create_monitor({"watch": "agent_end"})
    monitor_id = _payload(res)["monitor_id"]

    api.fire("agent_end", {"event": 1})
    api.fire("agent_end", {"event": 2})

    drained = api.inbox.drain()
    assert len(drained) == 1
    assert drained[0].dedup_key == f"monitor-chan-{monitor_id}"
    # Latest fire wins (dedup replaces in place).
    assert "2" in drained[0].payload["event_summary"]


@pytest.mark.asyncio
async def test_cancel_wakeup_before_fire_posts_nothing() -> None:
    """Cancelling a wakeup before its delay elapses cancels the asyncio.Task
    AND keeps the inbox empty — no fire is posted."""

    api = _FakeApi()
    mgr = _manager(api)
    res = await mgr.schedule_wakeup({"delay": 60.0})
    monitor_id = _payload(res)["monitor_id"]

    state = mgr._monitors[monitor_id]
    assert state.task is not None
    assert not state.task.done()

    cancel = await mgr.cancel_monitor({"monitor_id": monitor_id})
    assert _payload(cancel)["status"] == "cancelled"

    # Yield once so the cancellation actually propagates and the task settles.
    await asyncio.sleep(0)
    assert state.task.done()
    assert api.inbox.is_empty()
    assert state.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_channel_unsubscribes_so_later_fires_post_nothing() -> None:
    """Cancelling a channel monitor removes its subscription — subsequent fires
    are no-ops as far as the inbox is concerned."""

    api = _FakeApi()
    mgr = _manager(api)
    res = await mgr.create_monitor({"watch": "tool_call"})
    monitor_id = _payload(res)["monitor_id"]

    cancel = await mgr.cancel_monitor({"monitor_id": monitor_id})
    assert _payload(cancel)["status"] == "cancelled"

    # The channel bucket no longer contains a handler.
    assert api._handlers.get("tool_call", []) == []

    api.fire("tool_call", {"event": "after-cancel"})
    assert api.inbox.is_empty()


@pytest.mark.asyncio
async def test_cancel_after_fired_keeps_fired_status() -> None:
    """MAJOR fix (step-4 review): ``_FIRED`` is terminal; ``cancel_monitor``
    MUST NOT overwrite a successfully-fired wakeup with ``_CANCELLED``
    (the bookkeeping would lie to the agent). Returns the actual current
    status and ``list_monitors`` still reports ``fired``.
    """

    api = _FakeApi()
    mgr = _manager(api)

    res = await mgr.schedule_wakeup({"delay": 0.02})
    monitor_id = _payload(res)["monitor_id"]
    await asyncio.sleep(0.1)  # let the wakeup fire

    listing = _payload(await mgr.list_monitors({}))
    fired = next(m for m in listing["monitors"] if m["monitor_id"] == monitor_id)
    assert fired["status"] == "fired"

    cancel = await mgr.cancel_monitor({"monitor_id": monitor_id})
    assert _payload(cancel)["status"] == "fired"
    assert not cancel.is_error

    listing_after = _payload(await mgr.list_monitors({}))
    entry = next(m for m in listing_after["monitors"] if m["monitor_id"] == monitor_id)
    assert entry["status"] == "fired"


@pytest.mark.asyncio
async def test_shutdown_does_not_overwrite_fired_status() -> None:
    """MAJOR fix (step-4 review): ``on_session_shutdown`` MUST NOT flip a
    ``_FIRED`` wakeup to ``_CANCELLED``. Already-terminal monitors stay
    terminal across shutdown.
    """

    api = _FakeApi()
    mgr = _manager(api)

    res = await mgr.schedule_wakeup({"delay": 0.02})
    monitor_id = _payload(res)["monitor_id"]
    await asyncio.sleep(0.1)  # let it fire

    await mgr.on_session_shutdown(SessionShutdownEvent(cwd="."))

    listing = _payload(await mgr.list_monitors({}))
    entry = next(m for m in listing["monitors"] if m["monitor_id"] == monitor_id)
    assert entry["status"] == "fired"


@pytest.mark.asyncio
async def test_cancel_unknown_returns_tool_error_and_is_idempotent() -> None:
    api = _FakeApi()
    mgr = _manager(api)
    miss = await mgr.cancel_monitor({"monitor_id": "nope"})
    assert miss.is_error
    assert "unknown" in _payload(miss)["error"]

    # Cancelling twice the same monitor is idempotent.
    res = await mgr.schedule_wakeup({"delay": 60.0})
    monitor_id = _payload(res)["monitor_id"]
    first = await mgr.cancel_monitor({"monitor_id": monitor_id})
    second = await mgr.cancel_monitor({"monitor_id": monitor_id})
    assert _payload(first)["status"] == "cancelled"
    assert _payload(second)["status"] == "cancelled"
    assert not second.is_error


@pytest.mark.asyncio
async def test_session_shutdown_cancels_all_and_clears_subscriptions() -> None:
    """MAJOR (step-3 lesson #1): the shutdown handler cancels every pending
    wakeup task AND clears every channel subscription so nothing leaks past
    bus clearing. Both kinds of monitor go terminal."""

    api = _FakeApi()
    mgr = _manager(api)

    w1 = await mgr.schedule_wakeup({"delay": 60.0})
    w2 = await mgr.schedule_wakeup({"delay": 60.0})
    c1 = await mgr.create_monitor({"watch": "tool_call"})
    c2 = await mgr.create_monitor({"watch": "agent_end"})
    wake_ids = [_payload(w1)["monitor_id"], _payload(w2)["monitor_id"]]
    chan_ids = [_payload(c1)["monitor_id"], _payload(c2)["monitor_id"]]

    await mgr.on_session_shutdown(SessionShutdownEvent(cwd="."))

    for mid in wake_ids:
        state = mgr._monitors[mid]
        assert state.task is not None
        assert state.task.done()
        assert state.status == "cancelled"
    for mid in chan_ids:
        state = mgr._monitors[mid]
        assert state.status == "cancelled"
    assert api._handlers.get("tool_call", []) == []
    assert api._handlers.get("agent_end", []) == []
    assert mgr._shutting_down is True

    # Post-shutdown new monitors are refused (not stranded on a cleared bus).
    refused_wake = await mgr.schedule_wakeup({"delay": 0.0})
    refused_chan = await mgr.create_monitor({"watch": "tool_call"})
    assert refused_wake.is_error
    assert refused_chan.is_error


@pytest.mark.asyncio
async def test_post_inbox_stale_does_not_crash_wakeup() -> None:
    """MAJOR (step-3 lesson #3): the detached wakeup task swallows
    ``ExtensionStaleError`` from ``post_inbox`` rather than dying with an
    unretrieved exception."""

    class _StaleApi(_FakeApi):
        def post_inbox(self, *, source: str, payload: Any, dedup_key: Any = None) -> None:
            raise ExtensionStaleError("reloaded mid-flight")

    api = _StaleApi()
    mgr = _manager(api)
    res = await mgr.schedule_wakeup({"delay": 0.01})
    monitor_id = _payload(res)["monitor_id"]

    state = mgr._monitors[monitor_id]
    assert state.task is not None
    # Must NOT raise: the wakeup runner catches ExtensionStaleError.
    await state.task
    assert state.task.done()
    # The exception was swallowed cleanly.
    assert state.task.exception() is None


@pytest.mark.asyncio
async def test_post_inbox_stale_does_not_crash_channel_handler() -> None:
    """Same MAJOR-3 lesson on the channel side: firing a channel whose handler
    sees a stale api must not propagate the exception into the bus dispatch."""

    class _StaleApi(_FakeApi):
        def post_inbox(self, *, source: str, payload: Any, dedup_key: Any = None) -> None:
            raise ExtensionStaleError("reloaded mid-flight")

    api = _StaleApi()
    mgr = _manager(api)
    await mgr.create_monitor({"watch": "tool_call"})

    # Must NOT raise.
    api.fire("tool_call", {"event": 1})


@pytest.mark.asyncio
async def test_list_monitors_reports_live_and_terminal_state() -> None:
    api = _FakeApi()
    mgr = _manager(api)
    w = await mgr.schedule_wakeup({"delay": 60.0})
    c = await mgr.create_monitor({"watch": "tool_call"})
    wake_id = _payload(w)["monitor_id"]
    chan_id = _payload(c)["monitor_id"]

    res = await mgr.list_monitors({})
    rows = {row["monitor_id"]: row for row in _payload(res)["monitors"]}
    assert rows[wake_id]["kind"] == "wakeup"
    assert rows[wake_id]["status"] == "pending"
    assert rows[chan_id]["kind"] == "channel"
    assert rows[chan_id]["watch"] == "tool_call"
    assert rows[chan_id]["status"] == "active"

    await mgr.cancel_monitor({"monitor_id": wake_id})
    res2 = await mgr.list_monitors({})
    rows2 = {row["monitor_id"]: row for row in _payload(res2)["monitors"]}
    assert rows2[wake_id]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_create_monitor_condition_form_refused() -> None:
    """Condition-polling is deferred; the agent gets a tool-error result, not
    a silent no-op."""

    api = _FakeApi()
    mgr = _manager(api)
    res = await mgr.create_monitor({"condition": "x == 1"})
    assert res.is_error
    assert "not implemented" in _payload(res)["error"]


def test_install_registers_tools_and_shutdown_handler() -> None:
    """``install`` registers exactly the four monitor tools and the
    session_shutdown handler so MANIFEST.registers and the wiring agree."""

    api = _FakeApi()
    monitor.install(cast(ExtensionAPI, api), {})
    names = {t.name for t in api.tools}
    assert names == {
        "schedule_wakeup",
        "create_monitor",
        "list_monitors",
        "cancel_monitor",
    }
    assert "session_shutdown" in api._handlers


def test_render_item_monitor_source_renders() -> None:
    """``source="monitor"`` lands as a <system-reminder>-wrapped user message
    (cache-stable, same shape as ``background``)."""

    msg = render_item(
        InboxItem(source="monitor", payload="wakeup fired", dedup_key="k")
    )
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert "<system-reminder>" in msg.content[0].text
    assert "wakeup fired" in msg.content[0].text


def test_render_item_unhandled_source_still_raises() -> None:
    """Sources not yet wired (subagent) still fail loudly — step 4 only opens
    ``monitor``, not the whole catalog."""

    with pytest.raises(NotImplementedError):
        render_item(InboxItem(source="subagent", payload="x"))
