"""Fail-stop coverage for the ``monitor`` atom (Session Inbox step 4).

Design: ``.claude/designs/session-inbox.md`` (``monitor`` section). Same fail-stop
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
from agentm.core.abi import SessionShutdownEvent
from agentm.core.abi import ExtensionAPI, ExtensionStaleError
from agentm.core.runtime.session_inbox import InboxItem, render_item
from agentm.extensions.builtin import monitor
from agentm.extensions.builtin.monitor import MonitorConfig, _MonitorManager
from tests.unit.extensions._fake_api import FakeExtensionAPI

# Alias for diff continuity with the pre-B7 tests; the shared helper IS the
# minimal ExtensionAPI shim these tests rely on (subscribe/unsubscribe,
# post_inbox + dedup, and a ``fire(channel, event)`` test sugar to dispatch
# subscribed handlers without standing up the real EventBus).
_FakeApi = FakeExtensionAPI


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
    # ``tool_call`` is a legitimate agent-observable channel (not in the
    # kernel-control denylist that A1 added).
    res = await mgr.create_monitor({"watch": "tool_call"})
    monitor_id = _payload(res)["monitor_id"]

    api.fire("tool_call", {"event": 1})
    api.fire("tool_call", {"event": 2})

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
    # Both ``tool_call`` and ``plan_submitted`` are agent-observable channels
    # (not in A1's kernel-control denylist).
    c1 = await mgr.create_monitor({"watch": "tool_call"})
    c2 = await mgr.create_monitor({"watch": "plan_submitted"})
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
    assert api._handlers.get("plan_submitted", []) == []
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
        def post_inbox(
            self,
            *,
            source: str,
            payload: Any,
            dedup_key: Any = None,
            terminal: bool = False,
        ) -> None:
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
        def post_inbox(
            self,
            *,
            source: str,
            payload: Any,
            dedup_key: Any = None,
            terminal: bool = False,
        ) -> None:
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
async def test_create_monitor_refuses_kernel_control_channels() -> None:
    """A1 fail-stop: subscribing to a kernel-control channel must be refused
    at ``create_monitor`` so the agent cannot wedge the loop into an
    inescapable spin via ``watch="context"`` and friends.

    The canonical bug: a monitor on ``context`` posts to the inbox on every
    fire; the inbox-non-empty floor keeps the loop alive; the next turn fires
    ``context`` again. Without the denylist this is an unbreakable infinite
    loop. The refusal must NOT leave a subscription on the bus.
    """

    api = _FakeApi()
    mgr = _manager(api)

    # Pick a representative across the categories the denylist covers.
    # The list includes ``message_appended`` and ``session_header_emitted``
    # explicitly because they are the exact siblings A1's first revision
    # missed (boundary re-review) — regressing them out of the denylist
    # would reopen the same spin class.
    for channel in (
        "context",
        "decide_turn_action",
        "agent_start",
        "agent_end",
        "before_send_to_llm",
        "message_persisted",
        "message_appended",
        "session_header_emitted",
        "session_ready",
        "session_shutdown",
        "extension_install",
        "api_register",
    ):
        res = await mgr.create_monitor({"watch": channel})
        assert res.is_error, f"expected refusal for {channel!r}"
        err = _payload(res)["error"]
        assert channel in err, f"error message must name the channel: {err}"
        # No subscription leaked onto the bus.
        assert api._handlers.get(channel, []) == [], (
            f"refused channel {channel!r} must not leave a handler"
        )
        # No monitor entry was registered.
        assert not mgr._monitors


@pytest.mark.asyncio
async def test_create_monitor_condition_form_polls_inbox() -> None:
    """#178 fail-stop: the condition form starts a recurring poll that re-posts
    the free-text predicate as a ``source="monitor"`` item with
    ``kind="condition"`` and a stable ``monitor-cond-{id}`` dedup_key.

    Without the polling implementation ``create_monitor`` returned a tool-error
    refusal and NO inbox item ever appeared. Here we assert a real item lands.

    The poll period is driven to ~0 by a small floor so the test does not sleep
    a real interval; we then cancel to stop the metronome.
    """

    api = _FakeApi()
    mgr = _MonitorManager(
        api=cast(ExtensionAPI, api), condition_poll_min_seconds=0.01
    )
    res = await mgr.create_monitor(
        {"condition": "queue is drained", "poll_interval": 0.01, "note": "n"}
    )
    assert not res.is_error
    payload = _payload(res)
    monitor_id = payload["monitor_id"]
    assert payload["kind"] == "condition"
    assert payload["status"] == "active"
    assert payload["condition"] == "queue is drained"

    # Let at least one poll fire, then stop the metronome.
    item = None
    for _ in range(200):
        await asyncio.sleep(0.01)
        drained = api.inbox.drain()
        if drained:
            item = drained[-1]
            break
    await mgr.cancel_monitor({"monitor_id": monitor_id})
    assert item is not None, "condition poll never posted an inbox item"
    assert item.source == "monitor"
    assert item.dedup_key == f"monitor-cond-{monitor_id}"
    assert isinstance(item.payload, dict)
    assert item.payload["kind"] == "condition"
    assert item.payload["condition"] == "queue is drained"
    assert item.payload["monitor_id"] == monitor_id
    assert item.payload["note"] == "n"


@pytest.mark.asyncio
async def test_create_monitor_rejects_both_and_neither() -> None:
    """#178 guard: ``watch`` and ``condition`` are mutually exclusive; passing
    both, or neither, returns a tool-error so the agent sees the misuse — never
    a silent no-op or a half-registered monitor."""

    api = _FakeApi()
    mgr = _manager(api)

    both = await mgr.create_monitor({"watch": "tool_call", "condition": "x"})
    assert both.is_error
    assert not mgr._monitors

    neither = await mgr.create_monitor({})
    assert neither.is_error
    assert not mgr._monitors


@pytest.mark.asyncio
async def test_condition_poll_interval_clamped_to_floor() -> None:
    """#178 fail-stop: a per-call ``poll_interval`` below the configured floor is
    clamped UP so the agent cannot wedge the session into a tight re-poll spin.

    Without the clamp a near-0 interval would busy-spin posting every loop tick.
    """

    api = _FakeApi()
    mgr = _MonitorManager(
        api=cast(ExtensionAPI, api), condition_poll_min_seconds=5.0
    )
    res = await mgr.create_monitor({"condition": "x", "poll_interval": 0.001})
    monitor_id = _payload(res)["monitor_id"]
    assert _payload(res)["poll_interval"] == 5.0
    state = mgr._monitors[monitor_id]
    assert state.poll_interval == 5.0
    await mgr.cancel_monitor({"monitor_id": monitor_id})


@pytest.mark.asyncio
async def test_condition_monitor_cancel_stops_metronome() -> None:
    """#178 fail-stop: ``cancel_monitor`` on a condition monitor cancels ONLY
    its own task and posts no further items — the per-monitor cancel discipline
    (never a shared kernel signal)."""

    api = _FakeApi()
    mgr = _MonitorManager(
        api=cast(ExtensionAPI, api), condition_poll_min_seconds=0.01
    )
    res = await mgr.create_monitor({"condition": "x", "poll_interval": 0.01})
    monitor_id = _payload(res)["monitor_id"]
    state = mgr._monitors[monitor_id]

    await mgr.cancel_monitor({"monitor_id": monitor_id})
    assert state.status == "cancelled"

    # Let the cancellation propagate, then confirm the metronome is dead: no
    # further inbox items across several poll periods.
    api.inbox.drain()
    await asyncio.sleep(0.1)
    assert api.inbox.drain() == [], "cancelled condition monitor kept posting"
    assert state.task is not None and state.task.done()


@pytest.mark.asyncio
async def test_condition_monitor_shutdown_cancels_task() -> None:
    """#178 fail-stop: ``on_session_shutdown`` cancels a live condition poll so
    no detached task survives the session (same lifecycle as wakeup)."""

    api = _FakeApi()
    mgr = _MonitorManager(
        api=cast(ExtensionAPI, api),
        condition_poll_min_seconds=0.01,
        shutdown_grace_seconds=0.05,
    )
    res = await mgr.create_monitor({"condition": "x", "poll_interval": 60.0})
    monitor_id = _payload(res)["monitor_id"]
    state = mgr._monitors[monitor_id]

    await mgr.on_session_shutdown(SessionShutdownEvent(cwd="."))
    assert state.task is not None
    assert state.task.done()


@pytest.mark.asyncio
async def test_condition_poll_stale_does_not_crash() -> None:
    """#178 fail-stop: a detached condition poll whose ``post_inbox`` raises
    ``ExtensionStaleError`` (atom reloaded mid-flight) stops gracefully — no
    unretrieved task exception."""

    class _StaleApi(_FakeApi):
        def post_inbox(
            self,
            *,
            source: str,
            payload: Any,
            dedup_key: Any = None,
            terminal: bool = False,
        ) -> None:
            raise ExtensionStaleError("reloaded mid-flight")

    api = _StaleApi()
    mgr = _MonitorManager(
        api=cast(ExtensionAPI, api), condition_poll_min_seconds=0.01
    )
    res = await mgr.create_monitor({"condition": "x", "poll_interval": 0.01})
    monitor_id = _payload(res)["monitor_id"]
    state = mgr._monitors[monitor_id]
    assert state.task is not None
    await state.task  # must return, not raise


def test_install_registers_tools_and_shutdown_handler() -> None:
    """``install`` registers exactly the four monitor tools and the
    session_shutdown handler so MANIFEST.registers and the wiring agree."""

    api = _FakeApi()
    monitor.install(cast(ExtensionAPI, api), MonitorConfig())
    names = {t.name for t in api.tools}
    assert names == {
        "schedule_wakeup",
        "create_monitor",
        "list_monitors",
        "cancel_monitor",
    }
    assert "session_shutdown" in api._handlers


@pytest.mark.asyncio
async def test_install_propagates_shutdown_grace_config() -> None:
    """A3 fail-stop: a config override (``shutdown_grace_seconds=0.05``)
    propagates from ``install`` to the manager and bounds the actual drain.

    Drives a still-pending wakeup (60s sleep) through the shutdown handler.
    The drain awaits the wakeup task under the configured grace; with the
    default 5.0 this test would block ~5s. The override ensures it finishes
    well under a second (the wakeup task gets cancelled, so the gather is
    immediate).
    """

    api = _FakeApi()
    monitor.install(
        cast(ExtensionAPI, api),
        MonitorConfig(shutdown_grace_seconds=0.05),
    )
    # The manager isn't exposed by install(); reach it through the captured
    # shutdown handler closure (the install pattern wraps it in api.on()).
    shutdown_handlers = api._handlers["session_shutdown"]
    assert len(shutdown_handlers) == 1
    handler = shutdown_handlers[0]
    bound_manager = handler.__self__  # type: ignore[attr-defined]
    assert bound_manager._shutdown_grace_seconds == 0.05

    # Push a wakeup, then shut down — drain must complete promptly.
    await bound_manager.schedule_wakeup({"delay": 60.0})
    start = asyncio.get_event_loop().time()
    await asyncio.wait_for(
        bound_manager.on_session_shutdown(SessionShutdownEvent(cwd=".")),
        timeout=2.0,  # generous; if grace is honoured this returns near-instantly
    )
    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed < 1.0, f"shutdown drain took {elapsed:.3f}s — grace not honoured"


@pytest.mark.asyncio
async def test_install_propagates_event_summary_max_chars() -> None:
    """B5 fail-stop: ``event_summary_max_chars`` config knob propagates from
    ``install`` to the manager (no module-level constant override needed)."""

    api = _FakeApi()
    monitor.install(
        cast(ExtensionAPI, api),
        MonitorConfig(event_summary_max_chars=10),
    )
    handler = api._handlers["session_shutdown"][0]
    bound_manager = handler.__self__  # type: ignore[attr-defined]
    assert bound_manager._event_summary_max_chars == 10

    await bound_manager.create_monitor({"watch": "tool_call"})
    api.fire("tool_call", {"event": "x" * 100})
    drained = api.inbox.drain()
    assert len(drained) == 1
    summary = drained[0].payload["event_summary"]
    # Bounded to the configured cap (10 chars max, including the "..." tail).
    assert len(summary) <= 10


def test_render_item_monitor_source_renders() -> None:
    """``source="monitor"`` lands as a ``<system-reminder source="monitor">``-
    wrapped user message (cache-stable, same shape as ``background`` /
    ``subagent`` but distinguished by the wrapper's source attribute)."""

    msg = render_item(
        InboxItem(source="monitor", payload="wakeup fired", dedup_key="k")
    )
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert '<system-reminder source="monitor">' in msg.content[0].text
    assert "wakeup fired" in msg.content[0].text


def test_render_item_unhandled_source_still_raises() -> None:
    """Sources outside the catalog (user/background/monitor/subagent)
    still fail loudly — step 5 wired ``subagent``, but an unknown source
    must NOT silently land as a user message."""

    with pytest.raises(NotImplementedError):
        render_item(InboxItem(source="totally-unknown-source", payload="x"))
