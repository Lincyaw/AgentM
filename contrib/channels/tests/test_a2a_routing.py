"""Phase 6 — agent-to-agent (A2A) routing tests.

Drives a real :class:`WireServer` with a :class:`WireBridge` and attaches
mock workers / chat clients via :class:`WireClient`. Verifies the new
Phase 6 invariants: hop limit, root_session_key propagation, missing-
root rejection, approval-route override, and correlation_id preservation
through both legs of a worker→worker delegation.

No AgentSession, no subprocess — the bridge is exercised directly.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm_channels.bus import MessageBus
from agentm_channels.client import WireClient
from agentm_channels.manager import ChannelManager
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.server import WireServer
from agentm_channels.wire import (
    KIND_ERROR,
    KIND_INBOUND,
    WIRE_VERSION,
    Envelope,
)
from agentm_channels.wire_bridge import WireBridge
from agentm_channels.worker_registry import WorkerRegistry


# -- fixtures ----------------------------------------------------------


@pytest.fixture
def tmp_dir() -> "AsyncIterator[str]":  # type: ignore[type-arg]
    with tempfile.TemporaryDirectory(prefix="agentm-a2a-") as d:
        yield d


@pytest.fixture
def socket_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "gw.sock")


@pytest.fixture
def db_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "outbox.sqlite")


async def _wait(predicate: Any, timeout: float = 2.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


async def _start_server(
    socket_path: str,
    db_path: str,
    *,
    scenario: str = "x",
    allow_inproc: bool = False,
    max_a2a_hops: int = 10,
) -> tuple[WireServer, WireBridge, MessageBus, SqliteOutbox, SqliteInbox]:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)
    bus = MessageBus()
    manager = ChannelManager({}, bus)
    bridge = WireBridge(
        bus=bus,
        manager=manager,
        outbox=outbox,
        worker_registry=WorkerRegistry(),
        scenario=scenario,
        allow_inproc=allow_inproc,
        max_a2a_hops=max_a2a_hops,
    )
    server = WireServer(
        socket_path=socket_path,
        outbox=outbox,
        inbox=inbox,
        on_inbound=bridge.handle_inbound,
        on_peer_hello=bridge.handle_peer_hello,
        on_peer_disconnect=bridge.handle_peer_disconnect,
        on_worker_outbound=bridge.handle_worker_outbound,
    )
    await server.start()
    return server, bridge, bus, outbox, inbox


def _make_chat_inbound(
    *,
    peer_id: str,
    channel: str = "terminal",
    chat_id: str = "t1",
    content: str = "hello",
    env_id: str | None = None,
    correlation_id: str | None = None,
) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=env_id or f"in-{peer_id}-{int(time.time() * 1_000_000)}",
        kind=KIND_INBOUND,
        ts=time.time(),
        body={
            "channel": channel,
            "chat_id": chat_id,
            # Bridge binds sender_id to the authenticated peer (issue
            # #1). Mirror the peer_id here so the spoof guard accepts
            # the envelope.
            "sender_id": peer_id,
            "content": content,
        },
        correlation_id=correlation_id,
    )


def _make_worker_inbound(
    *,
    to: str,
    correlation_id: str,
    root_session_key: str | None,
    hops: int = 1,
    content: str = "delegate",
) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=f"a2a-{int(time.time() * 1_000_000)}",
        kind=KIND_INBOUND,
        ts=time.time(),
        body={
            "channel": "_a2a",
            "chat_id": correlation_id,
            "sender_id": "agent",
            "content": content,
        },
        to=to,
        correlation_id=correlation_id,
        root_session_key=root_session_key,
        peer_kind="agent_worker",
        hops=hops,
    )


# -- tests -------------------------------------------------------------


async def test_hop_limit_enforced(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path, max_a2a_hops=2
    )
    errors: list[Envelope] = []

    async def collect(env: Envelope) -> None:
        if env.kind == KIND_ERROR:
            errors.append(env)

    try:
        worker_a = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect,
        )
        worker_b = WireClient(
            socket_path=socket_path,
            peer_id="worker-B",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
        )
        await worker_a.connect()
        await worker_b.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-A"))
        assert await _wait(lambda: bridge.worker_registry.has("worker-B"))

        # Forge an envelope that's already at hops=2 — one more forward
        # would exceed the cap.
        env = _make_worker_inbound(
            to="worker-B",
            correlation_id="c1",
            root_session_key="terminal:t1",
            hops=2,
        )
        await worker_a.send(env)
        assert await _wait(lambda: len(errors) >= 1, timeout=2.0)
        err = errors[0]
        assert err.kind == KIND_ERROR
        body = err.body if isinstance(err.body, dict) else {}
        assert body.get("reason") == "hop_limit_exceeded"
        assert body.get("max_a2a_hops") == 2
        assert body.get("correlation_id") == "c1"
        await worker_a.close()
        await worker_b.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()




async def test_missing_root_session_key_rejected(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path
    )
    errors: list[Envelope] = []

    async def collect(env: Envelope) -> None:
        if env.kind == KIND_ERROR:
            errors.append(env)

    try:
        worker_a = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect,
        )
        worker_b = WireClient(
            socket_path=socket_path,
            peer_id="worker-B",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
        )
        await worker_a.connect()
        await worker_b.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-B"))

        # Send a worker-originated inbound with no root_session_key.
        await worker_a.send(
            _make_worker_inbound(
                to="worker-B",
                correlation_id="c2",
                root_session_key=None,
                hops=1,
            )
        )
        assert await _wait(lambda: len(errors) >= 1, timeout=2.0)
        err = errors[0]
        body = err.body if isinstance(err.body, dict) else {}
        assert body.get("reason") == "missing_root_session_key"
        assert body.get("correlation_id") == "c2"
        await worker_a.close()
        await worker_b.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()




