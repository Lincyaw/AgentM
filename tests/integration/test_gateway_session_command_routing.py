"""Gateway must forward session-registered slash commands to the session.

Fail-stop for the single-process gateway's command-routing seam. The gateway
has two command layers: its own builtins (``/status``, ``/help``, ...) routed
by ``CommandRouter``, and session-registered commands (``/compact`` and
friends), dispatched by the session itself.

Current behavior routes unknown gateway names to session submit (unless the
session command set is known and explicitly excludes the command, in which case
the gateway emits ``diagnostic_error``).
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm.core.runtime.session_inbox import InboxItem
from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.commands import CommandRouter, discover_commands
from agentm.gateway.outbox import SqliteOutbox
from agentm.gateway.runtime import GatewayRuntime
from agentm.gateway.wire import InboundBody


class _FakeInbox:
    def __init__(self) -> None:
        self.items: list[InboxItem] = []

    def push(self, item: InboxItem) -> None:
        self.items.append(item)


class _FakeSession:
    def __init__(self) -> None:
        self.inbox = _FakeInbox()


async def _fake_factory(*_a: Any) -> _FakeSession:
    return _FakeSession()


def _runtime(tmp_path: Any) -> tuple[GatewayRuntime, SqliteOutbox]:
    outbox = SqliteOutbox(str(tmp_path / "o.sqlite"))
    runtime = GatewayRuntime(
        cwd=".",
        scenario="local",
        outbox=outbox,
        chat_map=ChatSessionMap(tmp_path / "m.json"),
        session_factory=_fake_factory,
        command_router=CommandRouter(registry=discover_commands(".")),
        approval_policy=(frozenset(), frozenset(), 300.0),
        model_name="doubao",
        make_factory=lambda _n: _fake_factory,
    )
    return runtime, outbox


def _inbound(content: str) -> InboundBody:
    return InboundBody(
        channel="terminal",
        chat_id="c1",
        content=content,
        sender_id="u1",
    )


def _capture_outbound(runtime: GatewayRuntime, sink: list[dict[str, Any]]) -> None:
    async def _emit(body: dict[str, Any]) -> None:
        sink.append(body)

    runtime._emit_outbound = _emit  # type: ignore[method-assign]


def _capture_inbox_push(
    runtime: GatewayRuntime,
    pushes: list[str],
) -> None:
    orig_factory = runtime._sessions._factory

    async def _factory_with_capture(*args: Any) -> Any:
        sess = await orig_factory(*args)

        orig_push = sess.inbox.push

        def _push(item: InboxItem) -> None:
            pushes.append(item.payload)
            orig_push(item)

        sess.inbox.push = _push  # type: ignore[assignment]
        return sess

    runtime._sessions.set_factory(_factory_with_capture)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_known_session_command_is_forwarded_to_session(tmp_path: Any) -> None:
    runtime, outbox = _runtime(tmp_path)
    try:
        # The gateway learned /compact from a session_ready frame.
        runtime._session_commands["terminal:c1"] = {"compact"}
        pushes: list[str] = []
        _capture_inbox_push(runtime, pushes)

        outbound: list[dict[str, Any]] = []
        _capture_outbound(runtime, outbound)

        await runtime._run_command("terminal:c1", _inbound("/compact"))

        assert pushes == ["/compact"]
        assert not any(
            o.get("metadata", {}).get("kind") == "diagnostic_error"
            for o in outbound
        )
    finally:
        outbox.close()


@pytest.mark.asyncio
async def test_session_command_forwarded_when_set_unknown(tmp_path: Any) -> None:
    # Before any session_ready frame the gateway has no per-session set; it
    # forwards optimistically rather than rejecting a real command.
    runtime, outbox = _runtime(tmp_path)
    try:
        pushes: list[str] = []
        _capture_inbox_push(runtime, pushes)

        outbound: list[dict[str, Any]] = []
        _capture_outbound(runtime, outbound)

        await runtime._run_command("terminal:c1", _inbound("/compact"))

        assert pushes == ["/compact"]
        assert not any(
            o.get("metadata", {}).get("kind") == "diagnostic_error"
            for o in outbound
        )
    finally:
        outbox.close()


@pytest.mark.asyncio
async def test_gateway_builtin_still_dispatches_locally(tmp_path: Any) -> None:
    runtime, outbox = _runtime(tmp_path)
    try:
        outbound: list[dict[str, Any]] = []
        _capture_outbound(runtime, outbound)

        await runtime._run_command("terminal:c1", _inbound("/status"))

        assert outbound, "/status should emit a reply"
    finally:
        outbox.close()


@pytest.mark.asyncio
async def test_unknown_to_both_yields_diagnostic(tmp_path: Any) -> None:
    runtime, outbox = _runtime(tmp_path)
    try:
        # Session set is known and does NOT contain asdf.
        runtime._session_commands["terminal:c1"] = {"compact"}
        pushes: list[str] = []
        _capture_inbox_push(runtime, pushes)

        outbound: list[dict[str, Any]] = []
        _capture_outbound(runtime, outbound)

        await runtime._run_command("terminal:c1", _inbound("/asdf"))

        assert pushes == []
        kinds = [o.get("metadata", {}).get("kind") for o in outbound]
        assert "diagnostic_error" in kinds
    finally:
        outbox.close()
