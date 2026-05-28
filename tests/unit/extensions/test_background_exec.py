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
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.runtime.session_inbox import InboxItem, SessionInbox
from agentm.extensions.builtin import background_exec
from agentm.extensions.builtin.background_exec import _BgManager, _BgTool


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


class _FakeApi:
    """Fake ExtensionAPI exposing the surface ``background_exec`` touches.

    ``post_inbox`` delegates to a real :class:`SessionInbox` so dedup-replace
    semantics are the genuine ones; the runtime impl does exactly this.
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
        self._handlers.setdefault(channel, []).append(handler)
        return lambda: None


def _manager(api: _FakeApi, *, timeout: float = 60.0, **kw: Any) -> _BgManager:
    return _BgManager(
        api=cast(ExtensionAPI, api),
        timeout=timeout,
        heartbeat_interval=kw.get("heartbeat_interval", 120.0),
        silence_warning=kw.get("silence_warning", 300.0),
        denylist=kw.get("denylist", set()),
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
async def test_backgrounded_terminate_injected_as_completion() -> None:
    """Step-3 simplification: a backgrounded tool returning ToolTerminate is
    injected as an ordinary completion (does NOT stop the loop yet)."""

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
    assert any(i.source == "background" and "bg terminal" in str(i.payload) for i in drained)


def test_install_registers_companion_tools() -> None:
    """``install`` registers exactly the three companion tools and an
    agent_start handler."""

    api = _FakeApi()
    background_exec.install(cast(ExtensionAPI, api), {})
    names = {t.name for t in api.tools}
    assert names == {"check_background", "wait_background", "cancel_background"}
    assert "agent_start" in api._handlers
