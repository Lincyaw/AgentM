"""Fail-stop coverage for the ``background_exec`` atom (Session Inbox step 3).

Design: ``.claude/designs/session-inbox.md`` (the ``background_exec`` section +
the "Step-3 design decisions" block). These tests pin the load-bearing
positions only (quality over quantity):

* a sub-timeout call returns the wrapped tool's real result **unchanged**
  (transparency — fast tools must be byte-for-byte unaffected);
* an overrun returns a ticket ``ToolResult`` AND registers the still-running
  task; on completion the manager posts ``source="background"`` to the inbox;
* the ticker reuses one ``dedup_key`` so a new status REPLACES the prior
  undrained one (no stacking);
* ``cancel_background`` cancels via the registry (the first ``registry.cancel``
  caller);
* ``post_inbox`` round-trips through the fake API the same way the runtime
  impl does.

The atom is driven through a lightweight fake ``ExtensionAPI`` (the same
in-test stub style as ``test_tool_read_path_gate``) plus a real
:class:`SessionInbox` so ``post_inbox`` semantics (dedup replace) are exercised
end-to-end rather than mocked.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from agentm.core.abi import (
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.abi.extension import ExtensionAPI, ExtensionStaleError
from agentm.extensions.builtin import background_exec
from agentm.extensions.builtin.background_exec import _BgManager, _BgTool
from tests.unit.extensions._fake_api import FakeExtensionAPI

# Alias for diff continuity with the pre-B7 tests; the shared helper IS the
# minimal ExtensionAPI shim these tests rely on.
_FakeApi = FakeExtensionAPI


class _FakeTool:
    """Minimal Tool: runs ``fn(args)`` honouring an abort signal."""

    def __init__(self, name: str, fn: Any) -> None:
        self.name = name
        self.description = f"fake {name}"
        self.parameters = {"type": "object", "properties": {}}
        self._fn = fn

    async def execute(
        self, args: dict[str, Any], *, signal: asyncio.Event | None = None
    ) -> Any:
        return await self._fn(args, signal)


def _manager(api: _FakeApi, *, timeout: float = 60.0, **kw: Any) -> _BgManager:
    return _BgManager(
        api=cast(ExtensionAPI, api),
        timeout=timeout,
        heartbeat_interval=kw.get("heartbeat_interval", 120.0),
        silence_warning=kw.get("silence_warning", 300.0),
        denylist=kw.get("denylist", set()),
        shutdown_grace_seconds=kw.get("shutdown_grace_seconds", 5.0),
    )


def _payload(result: Any) -> dict[str, Any]:
    assert isinstance(result, ToolResult)
    text = "".join(b.text for b in result.content if isinstance(b, TextContent))
    return json.loads(text)


@pytest.mark.asyncio
async def test_subtimeout_returns_real_result_unchanged() -> None:
    """A fast call returns the wrapped tool's exact result object — no ticket,
    no inbox push, no registry entry."""

    sentinel = ToolResult(content=[TextContent(type="text", text="fast")], is_error=False)

    async def fast(_args: dict[str, Any], _sig: asyncio.Event | None) -> ToolResult:
        return sentinel

    api = _FakeApi()
    mgr = _manager(api, timeout=5.0)
    wrapped = _BgTool(_FakeTool("fast", fast), mgr)

    out = await wrapped.execute({})

    assert out is sentinel  # byte-for-byte: same object, unchanged
    assert api.inbox.is_empty()
    async with mgr._registry.lock:
        assert mgr._registry.values() == []


@pytest.mark.asyncio
async def test_foreground_terminate_passes_through() -> None:
    """A sub-timeout ToolTerminate is returned verbatim (must still end the
    loop foreground-side)."""

    term = ToolTerminate(
        result=ToolResult(content=[TextContent(type="text", text="done")]),
        reason="test:final",
    )

    async def terminator(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolTerminate:
        return term

    api = _FakeApi()
    mgr = _manager(api, timeout=5.0)
    wrapped = _BgTool(_FakeTool("term", terminator), mgr)

    out = await wrapped.execute({})
    assert out is term


@pytest.mark.asyncio
async def test_overrun_returns_ticket_registers_and_completes() -> None:
    """An overrun returns a running ticket, registers the task, and on
    completion posts a ``source="background"`` inbox item with the real result."""

    release = asyncio.Event()

    async def slow(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        await release.wait()
        return ToolResult(content=[TextContent(type="text", text="late result")])

    api = _FakeApi()
    # Tiny timeout so the slow call overruns immediately.
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("slow", slow), mgr)

    ticket = await wrapped.execute({})
    assert isinstance(ticket, ToolResult)
    payload = _payload(ticket)
    assert payload["status"] == "running"
    assert "task_id" in payload
    assert "moved to background" in payload["note"]

    task_id = payload["task_id"]
    async with mgr._registry.lock:
        states = mgr._registry.values()
    assert [s.task_id for s in states] == [task_id]

    # Let the inner tool finish; the watcher posts the completion.
    release.set()
    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    await state.task

    drained = api.inbox.drain()
    assert len(drained) == 1
    item = drained[0]
    assert item.source == "background"
    assert item.dedup_key == f"bg-complete-{task_id}"
    assert "late result" in item.payload
    assert state.status == "completed"


@pytest.mark.asyncio
async def test_ticker_dedup_replaces_no_stacking() -> None:
    """Repeated ticker pushes under one dedup_key collapse to a single
    undrained inbox item (replace, not stack)."""

    api = _FakeApi()
    key = "bg-ticker-T1"

    # Three rolling status lines for the same task, none drained in between.
    # post_inbox(dedup_key=...) is exactly what the per-task ticker calls.
    api.post_inbox(source="background", payload="still running 1s", dedup_key=key)
    api.post_inbox(source="background", payload="still running 2s", dedup_key=key)
    api.post_inbox(source="background", payload="still running 3s", dedup_key=key)

    drained = api.inbox.drain()
    assert len(drained) == 1
    assert drained[0].payload == "still running 3s"  # latest wins


@pytest.mark.asyncio
async def test_cancel_background_cancels_via_registry() -> None:
    """``cancel_background`` sets the abort signal through the registry; the
    inner tool observes it, the watcher flips status to cancelled and posts a
    completion item."""

    started = asyncio.Event()

    async def cancellable(_a: dict[str, Any], sig: asyncio.Event | None) -> ToolResult:
        assert sig is not None
        started.set()
        await sig.wait()  # cooperative cancellation hook
        raise asyncio.CancelledError()

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("cancellable", cancellable), mgr)

    ticket = await wrapped.execute({})
    task_id = _payload(ticket)["task_id"]
    await started.wait()

    res = await mgr.cancel_background({"task_id": task_id})
    assert _payload(res)["status"] == "cancelling"

    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    await state.task
    assert state.status == "cancelled"

    drained = api.inbox.drain()
    assert any(
        i.source == "background" and i.dedup_key == f"bg-complete-{task_id}"
        for i in drained
    )

    # Unknown id → error, not a crash.
    miss = await mgr.cancel_background({"task_id": "nope"})
    assert miss.is_error


@pytest.mark.asyncio
async def test_error_in_background_posts_error_completion() -> None:
    """A backgrounded tool that raises records an error status and posts an
    error completion to the inbox (not a silent drop)."""

    release = asyncio.Event()

    async def boom(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        await release.wait()
        raise RuntimeError("kaboom")

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("boom", boom), mgr)

    ticket = await wrapped.execute({})
    task_id = _payload(ticket)["task_id"]
    release.set()
    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    await state.task

    assert state.status == "error"
    assert state.error == "kaboom"
    drained = api.inbox.drain()
    assert any("kaboom" in str(i.payload) for i in drained)


@pytest.mark.asyncio
async def test_wrap_tools_skips_companions_and_denylist() -> None:
    """``wrap_tools`` wraps ordinary tools but leaves companion tools and
    denylisted names untouched; it is idempotent."""

    async def noop(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        return ToolResult(content=[])

    api = _FakeApi()
    api.tools = [
        _FakeTool("bash", noop),
        _FakeTool("never_me", noop),
        _FakeTool("check_background", noop),  # companion name
    ]
    mgr = _manager(api, denylist={"never_me"})

    mgr.wrap_tools()
    by_name = {t.name: t for t in api.tools}
    assert isinstance(by_name["bash"], _BgTool)
    assert not isinstance(by_name["never_me"], _BgTool)
    assert not isinstance(by_name["check_background"], _BgTool)

    # Idempotent: a second fire does not double-wrap.
    mgr.wrap_tools()
    assert sum(isinstance(t, _BgTool) for t in api.tools) == 1


@pytest.mark.asyncio
async def test_backgrounded_terminate_posts_terminal_inbox_item() -> None:
    """#177: a backgrounded tool returning ToolTerminate posts its completion
    with ``terminal=True`` so the runtime drain seam can route it through loop
    termination — NOT swallowed as an ordinary completion.

    Fail-stop: without the patch the completion lands as an ordinary
    ``terminal=False`` background item and the terminate intent is lost.
    """

    release = asyncio.Event()

    async def slow_terminate(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolTerminate:
        await release.wait()
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="bg terminal")]),
            reason="test:bg-final",
        )

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("slow_terminate", slow_terminate), mgr)

    ticket = await wrapped.execute({})
    task_id = _payload(ticket)["task_id"]
    release.set()
    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    await state.task

    assert state.status == "completed"
    assert isinstance(state.outcome, ToolTerminate)
    drained = api.inbox.drain()
    terminal_items = [
        i for i in drained if i.source == "background" and "bg terminal" in str(i.payload)
    ]
    assert len(terminal_items) == 1
    assert terminal_items[0].terminal is True


@pytest.mark.asyncio
async def test_backgrounded_non_terminate_completion_is_not_terminal() -> None:
    """#177 guard: an ordinary backgrounded completion stays ``terminal=False``
    so only a genuine ToolTerminate ends the loop."""

    release = asyncio.Event()

    async def slow(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        await release.wait()
        return ToolResult(content=[TextContent(type="text", text="ok")])

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("slow", slow), mgr)

    ticket = await wrapped.execute({})
    task_id = _payload(ticket)["task_id"]
    release.set()
    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    await state.task

    drained = api.inbox.drain()
    completions = [i for i in drained if i.source == "background" and "ok" in str(i.payload)]
    assert len(completions) == 1
    assert completions[0].terminal is False


@pytest.mark.asyncio
async def test_backgrounded_tool_tracks_work_until_completion() -> None:
    """#179 fail-stop: an auto-backgrounded tool brackets its detached lifetime
    in ``api.track_background`` so the session counts it as live work until the
    completion has been posted.

    Without the bracket a one-shot host (``agentm -p``) would see ``idle`` the
    instant the agent ends its turn — while this tool is still running — and
    exit, dropping the completion. We assert the inbox work-count is non-zero
    while the tool runs and back to zero (only) after it finishes.
    """

    release = asyncio.Event()

    async def slow(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        await release.wait()
        return ToolResult(content=[TextContent(type="text", text="done")])

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("slow", slow), mgr)

    ticket = await wrapped.execute({})
    task_id = _payload(ticket)["task_id"]

    # The tool overran the auto-bg timeout and is detached + still running:
    # the session must regard this as live background work.
    assert api.inbox.has_pending_work, (
        "a still-running backgrounded tool must keep the session non-idle"
    )

    release.set()
    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    await state.task

    # Completion posted AND the work count cleared — only now is the session
    # allowed to consider itself idle.
    assert not api.inbox.has_pending_work
    drained = api.inbox.drain()
    assert any(i.source == "background" and "done" in str(i.payload) for i in drained)


def test_install_registers_companion_tools() -> None:
    """``install`` registers exactly the three companion tools and the
    agent_start + session_shutdown handlers."""

    api = _FakeApi()
    background_exec.install(cast(ExtensionAPI, api), {})
    names = {t.name for t in api.tools}
    assert names == {"check_background", "wait_background", "cancel_background"}
    assert "agent_start" in api._handlers
    assert "session_shutdown" in api._handlers


@pytest.mark.asyncio
async def test_install_propagates_shutdown_grace_config() -> None:
    """A3 fail-stop: a config override (``shutdown_grace_seconds=0.05``)
    propagates from ``install`` to the manager and bounds the actual drain
    so a stuck inner tool cannot block shutdown past the configured window.
    """

    api = _FakeApi()
    background_exec.install(
        cast(ExtensionAPI, api),
        {"shutdown_grace_seconds": 0.05},
    )
    # Pull the manager off the bound shutdown handler so the test does not
    # reach into install internals.
    shutdown_handlers = api._handlers["session_shutdown"]
    assert len(shutdown_handlers) == 1
    bound_manager = shutdown_handlers[0].__self__  # type: ignore[attr-defined]
    assert bound_manager._shutdown_grace_seconds == 0.05

    # Drive a cooperative inner (honours the abort) through the overrun path
    # so the shutdown handler exercises the bounded ``asyncio.wait`` and then
    # the abort + gather sequence.
    started = asyncio.Event()

    async def cooperative(_a: dict[str, Any], sig: asyncio.Event | None) -> ToolResult:
        assert sig is not None
        started.set()
        await sig.wait()  # cooperative shutdown: returns when aborted
        return ToolResult(content=[TextContent(type="text", text="stopped")])

    wrapped = _BgTool(_FakeTool("cooperative", cooperative), bound_manager)
    # Tiny overrun timeout so the call promptly moves to background.
    bound_manager.timeout = 0.01
    await wrapped.execute({})
    await started.wait()

    start = asyncio.get_event_loop().time()
    await asyncio.wait_for(
        bound_manager.on_session_shutdown(SessionShutdownEvent(cwd=".")),
        timeout=2.0,
    )
    elapsed = asyncio.get_event_loop().time() - start
    # 0.05s grace + cooperative shutdown overhead — would have been ~5s
    # with the previous hard-coded constant.
    assert elapsed < 1.0, f"shutdown drain took {elapsed:.3f}s — grace not honoured"


@pytest.mark.asyncio
async def test_cancel_with_host_signal_does_not_touch_shared_signal() -> None:
    """MAJOR 2 regression: under a host-supplied kernel ``signal``,
    ``cancel_background`` must cancel ONLY the per-task work — never set the
    shared signal that the live turn and other in-flight calls share."""

    started = asyncio.Event()
    shared_signal = asyncio.Event()  # the kernel/session signal

    async def cancellable(_a: dict[str, Any], sig: asyncio.Event | None) -> ToolResult:
        assert sig is not None
        # The inner tool is handed the PER-TASK event, not the shared signal.
        assert sig is not shared_signal
        started.set()
        await sig.wait()
        raise asyncio.CancelledError()

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("cancellable", cancellable), mgr)

    # Drive execute with a non-None host signal — the path the old tests never
    # exercised (they all ran signal=None, which is why the bug slipped).
    ticket = await wrapped.execute({}, signal=shared_signal)
    task_id = _payload(ticket)["task_id"]
    await started.wait()

    res = await mgr.cancel_background({"task_id": task_id})
    assert _payload(res)["status"] == "cancelling"

    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    await state.task
    assert state.status == "cancelled"

    # The shared kernel signal was NOT tripped by cancelling this one task.
    assert not shared_signal.is_set()
    # And the per-task abort is a different object than the shared signal.
    assert state.abort_signal is not shared_signal


@pytest.mark.asyncio
async def test_host_signal_still_aborts_backgrounded_task() -> None:
    """The kernel signal must still abort a backgrounded call when it fires —
    the per-task isolation (Major 2) keeps the forward direction working."""

    started = asyncio.Event()
    shared_signal = asyncio.Event()

    async def waits(_a: dict[str, Any], sig: asyncio.Event | None) -> ToolResult:
        assert sig is not None
        started.set()
        await sig.wait()
        raise asyncio.CancelledError()

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("waits", waits), mgr)

    ticket = await wrapped.execute({}, signal=shared_signal)
    task_id = _payload(ticket)["task_id"]
    await started.wait()

    # Host fires the shared signal → forwarder propagates it into the per-task
    # abort → the inner tool observes it and the task goes terminal.
    shared_signal.set()
    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    await state.task
    assert state.status == "cancelled"


@pytest.mark.asyncio
async def test_session_shutdown_drains_running_tasks() -> None:
    """MAJOR 1: the shutdown handler cancels tickers/forwarders, aborts running
    tasks within the grace window, and gathers them so nothing leaks pending."""

    started = asyncio.Event()

    async def long_running(_a: dict[str, Any], sig: asyncio.Event | None) -> ToolResult:
        assert sig is not None
        started.set()
        await sig.wait()  # cooperative: returns when aborted
        return ToolResult(content=[TextContent(type="text", text="stopped")])

    api = _FakeApi()
    # Tiny grace window so the test does not wait the real 5s — passed via the
    # config knob the atom exposes (was a hidden module constant in PR #176).
    mgr = _manager(
        api, timeout=0.01, heartbeat_interval=1000.0, shutdown_grace_seconds=0.05
    )
    wrapped = _BgTool(_FakeTool("long_running", long_running), mgr)

    ticket = await wrapped.execute({})
    task_id = _payload(ticket)["task_id"]
    await started.wait()

    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    assert state.ticker is not None

    await mgr.on_session_shutdown(SessionShutdownEvent(cwd="."))

    # Inner watch task finished (abort fired), ticker cancelled — nothing left
    # pending that would warn "Task was destroyed but it is pending".
    assert state.task.done()
    assert state.ticker.done()
    assert mgr._shutting_down is True


@pytest.mark.asyncio
async def test_background_refused_after_shutdown() -> None:
    """Once shutdown has begun, a fresh overrun is refused (not stranded on a
    cleared bus): the inner work is aborted and an error ticket returned."""

    started = asyncio.Event()

    async def slow(_a: dict[str, Any], sig: asyncio.Event | None) -> ToolResult:
        assert sig is not None
        started.set()
        await sig.wait()
        return ToolResult(content=[TextContent(type="text", text="x")])

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    # Shutdown already ran (no tasks), then a late overrun arrives.
    await mgr.on_session_shutdown(SessionShutdownEvent(cwd="."))
    wrapped = _BgTool(_FakeTool("slow", slow), mgr)

    ticket = await wrapped.execute({})
    await started.wait()
    assert isinstance(ticket, ToolResult)
    payload = _payload(ticket)
    assert ticket.is_error
    assert "shutting down" in payload["error"]
    async with mgr._registry.lock:
        assert mgr._registry.values() == []


@pytest.mark.asyncio
async def test_background_refused_after_shutdown_terminates_noncooperative_inner() -> None:
    """MAJOR (refusal-branch leak): when ``background()`` is reached AFTER
    ``on_session_shutdown`` has drained the registry, the never-registered inner
    task must still be driven to a terminal state. A NON-cooperative inner tool
    (ignores the abort signal) would otherwise be detached and untracked → the
    Major-1 "Task was destroyed but it is pending" leak. The refusal branch must
    cancel it within the bounded grace and gather it before returning."""

    api = _FakeApi()
    # Tiny grace so the cancel path is taken quickly (inner never honours abort).
    mgr = _manager(
        api, timeout=0.01, heartbeat_interval=1000.0, shutdown_grace_seconds=0.05
    )
    await mgr.on_session_shutdown(SessionShutdownEvent(cwd="."))
    assert mgr._shutting_down is True

    # Non-cooperative inner: ignores the abort signal entirely, sleeps long.
    async def noncooperative(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        await asyncio.sleep(100.0)
        return ToolResult(content=[TextContent(type="text", text="never")])

    abort = asyncio.Event()
    inner: asyncio.Task[Any] = asyncio.create_task(noncooperative({}, abort))

    ticket = await mgr.background(
        tool_name="noncooperative",
        task=inner,
        abort_signal=abort,
        forwarder=None,
    )

    # Refusal ticket returned, registry still empty (never registered)...
    assert isinstance(ticket, ToolResult)
    assert ticket.is_error
    assert "shutting down" in _payload(ticket)["error"]
    async with mgr._registry.lock:
        assert mgr._registry.values() == []
    # ...and crucially the non-cooperative inner task reached a terminal state
    # (cancelled within the grace window) — no pending task survives the refusal.
    assert inner.done()


@pytest.mark.asyncio
async def test_post_inbox_stale_does_not_crash_watcher() -> None:
    """MAJOR 3: if the atom was reloaded mid-flight, ``post_inbox`` raises
    ``ExtensionStaleError``; the detached watcher must stop gracefully rather
    than die with an unretrieved exception."""

    release = asyncio.Event()

    async def slow(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        await release.wait()
        return ToolResult(content=[TextContent(type="text", text="late")])

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
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("slow", slow), mgr)

    ticket = await wrapped.execute({})
    task_id = _payload(ticket)["task_id"]
    release.set()
    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None
    # Must not raise: the watcher swallows ExtensionStaleError from post_inbox.
    await state.task
    assert state.status == "completed"


@pytest.mark.asyncio
async def test_check_background_marks_terminal_read_no_double_show() -> None:
    """NIT A: ``check_background`` marks a terminal task ``read`` so that the
    same completion reported in the tool result is NOT also re-injected into
    the inbox by ``_watch``.

    Deterministic: drive the state to terminal-but-unposted (the race window
    where ``check_background`` observes completion before ``_watch`` posts),
    then prove ``check_background`` claims it and a subsequent ``_watch`` post
    attempt (via ``_post_completion``) is a no-op."""

    release = asyncio.Event()

    async def slow(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        await release.wait()
        return ToolResult(content=[TextContent(type="text", text="ok")])

    api = _FakeApi()
    mgr = _manager(api, timeout=0.01, heartbeat_interval=1000.0)
    wrapped = _BgTool(_FakeTool("slow", slow), mgr)

    ticket = await wrapped.execute({})
    task_id = _payload(ticket)["task_id"]
    async with mgr._registry.lock:
        state = mgr._registry.get(task_id)
    assert state is not None

    # Simulate the race window: status is terminal but _watch has not yet
    # posted (read still False). check_background must claim it.
    state.status = "completed"  # type: ignore[assignment]
    state.outcome = None
    res = await mgr.check_background({})
    assert any(t["task_id"] == task_id for t in _payload(res)["tasks"])
    assert state.read is True

    # _watch's terminal post, arriving afterwards, is suppressed (no double).
    mgr._post_completion(state)
    assert api.inbox.is_empty()

    # Release the real inner task so it does not leak; its watcher also sees
    # read=True and stays silent.
    release.set()
    await state.task
    assert api.inbox.is_empty()


@pytest.mark.asyncio
async def test_wrap_tools_wraps_tools_registered_between_prompts() -> None:
    """NIT B: dropping the run-once flag lets a SECOND agent_start wrap tools
    that were registered after the first, with no double-wrapping."""

    async def noop(_a: dict[str, Any], _s: asyncio.Event | None) -> ToolResult:
        return ToolResult(content=[])

    api = _FakeApi()
    api.tools = [_FakeTool("first", noop)]
    mgr = _manager(api)

    mgr.wrap_tools()  # agent_start cycle 1
    assert isinstance(api.tools[0], _BgTool)

    # A later install_atom registers another tool between prompts.
    api.tools.append(_FakeTool("second", noop))

    mgr.wrap_tools()  # agent_start cycle 2
    by_name = {t.name: t for t in api.tools}
    assert isinstance(by_name["first"], _BgTool)
    assert isinstance(by_name["second"], _BgTool)
    # No double-wrap: still exactly one _BgTool per original tool name.
    assert sum(isinstance(t, _BgTool) for t in api.tools) == 2
