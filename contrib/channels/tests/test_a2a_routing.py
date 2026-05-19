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
    KIND_OUTBOUND,
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


async def test_root_session_key_propagated(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path
    )
    worker_a_received: list[Envelope] = []
    worker_b_received: list[Envelope] = []

    async def collect_a(env: Envelope) -> None:
        if env.kind == KIND_INBOUND:
            worker_a_received.append(env)

    async def collect_b(env: Envelope) -> None:
        if env.kind == KIND_INBOUND:
            worker_b_received.append(env)

    try:
        worker_a = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect_a,
        )
        worker_b = WireClient(
            socket_path=socket_path,
            peer_id="worker-B",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect_b,
        )
        chat = WireClient(
            socket_path=socket_path,
            peer_id="chat-A",
            peer_kind="chat_client",
        )
        await worker_a.connect()
        await worker_b.connect()
        await chat.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-A"))

        # Chat client → worker_A (Phase 5a routing: sticky-binds the
        # session_key to whichever matching worker registered first).
        await chat.send(_make_chat_inbound(peer_id="chat-A"))
        assert await _wait(lambda: len(worker_a_received) >= 1)
        fwd1 = worker_a_received[0]
        assert fwd1.root_session_key == "terminal:t1"
        # Hops bumped by 1 on the chat→worker forward.
        assert fwd1.hops == 1

        # Worker_A → worker_B delegation: copy root_session_key.
        await worker_a.send(
            _make_worker_inbound(
                to="worker-B",
                correlation_id="abc",
                root_session_key="terminal:t1",
                hops=1,
            )
        )
        assert await _wait(lambda: len(worker_b_received) >= 1)
        fwd2 = worker_b_received[0]
        assert fwd2.root_session_key == "terminal:t1"
        assert fwd2.correlation_id == "abc"
        # The gateway bumped hops from 1 to 2 on this forward.
        assert fwd2.hops == 2
        await worker_a.close()
        await worker_b.close()
        await chat.close()
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


async def test_approval_routes_to_root_chat(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path
    )
    worker_a_received: list[Envelope] = []
    chat_received: list[Envelope] = []

    async def chat_recv(env: Envelope) -> None:
        chat_received.append(env)

    async def worker_a_recv(env: Envelope) -> None:
        worker_a_received.append(env)

    try:
        worker_a = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=worker_a_recv,
        )
        worker_b = WireClient(
            socket_path=socket_path,
            peer_id="worker-B",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
        )
        chat = WireClient(
            socket_path=socket_path,
            peer_id="chat-A",
            peer_kind="chat_client",
            on_outbound=chat_recv,
        )
        await worker_a.connect()
        await worker_b.connect()
        await chat.connect()
        # Drive a chat→worker turn so the synthetic "terminal" channel
        # is bound to chat-A; the approval override needs that.
        await chat.send(_make_chat_inbound(peer_id="chat-A"))
        assert await _wait(lambda: "terminal" in bridge._peer_channels.values())

        # worker_B emits an approval_request whose root_session_key
        # points at the chat session.
        approval = Envelope(
            v=WIRE_VERSION,
            id="appr-1",
            kind=KIND_OUTBOUND,
            ts=time.time(),
            body={
                "channel": "_a2a",
                "chat_id": "abc",
                "content": "approve bash?",
                "metadata": {"kind": "approval_request", "id": "req-1"},
            },
            root_session_key="terminal:t1",
        )
        await worker_b.send(approval)
        # The chat client should see the approval card; worker_A should
        # NOT receive anything because the override skipped the chain.
        assert await _wait(lambda: len(chat_received) >= 1, timeout=2.0)
        card = next(
            (e for e in chat_received if e.kind == KIND_OUTBOUND), None
        )
        assert card is not None
        body = card.body if isinstance(card.body, dict) else {}
        assert body.get("channel") == "terminal"
        assert body.get("chat_id") == "t1"
        assert body.get("metadata", {}).get("kind") == "approval_request"
        # Worker_A received the original forwarded chat inbound but no
        # approval card.
        approvals_at_a = [
            e
            for e in worker_a_received
            if isinstance(e.body, dict)
            and isinstance(e.body.get("metadata"), dict)
            and e.body["metadata"].get("kind") == "approval_request"
        ]
        assert approvals_at_a == []
        await worker_a.close()
        await worker_b.close()
        await chat.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


async def test_correlation_id_preserved_round_trip(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path
    )
    worker_a_received: list[Envelope] = []
    worker_b_received: list[Envelope] = []

    async def collect_a(env: Envelope) -> None:
        worker_a_received.append(env)

    async def collect_b(env: Envelope) -> None:
        worker_b_received.append(env)

    try:
        worker_a = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect_a,
        )
        worker_b = WireClient(
            socket_path=socket_path,
            peer_id="worker-B",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect_b,
        )
        await worker_a.connect()
        await worker_b.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-B"))

        # Leg 1: worker_A → worker_B inbound with correlation_id "abc"
        await worker_a.send(
            _make_worker_inbound(
                to="worker-B",
                correlation_id="abc",
                root_session_key="terminal:t1",
                hops=1,
            )
        )
        assert await _wait(
            lambda: any(
                e.correlation_id == "abc"
                and e.kind == KIND_INBOUND
                for e in worker_b_received
            ),
            timeout=2.0,
        )

        # Leg 2: worker_B replies with KIND_OUTBOUND carrying same id +
        # to=worker-A. The gateway forwards it as an outbound to
        # worker_A.
        reply = Envelope(
            v=WIRE_VERSION,
            id="reply-1",
            kind=KIND_OUTBOUND,
            ts=time.time(),
            body={
                "channel": "_a2a",
                "chat_id": "abc",
                "content": "done",
            },
            to="worker-A",
            correlation_id="abc",
            root_session_key="terminal:t1",
        )
        await worker_b.send(reply)
        assert await _wait(
            lambda: any(
                e.correlation_id == "abc"
                and e.kind == KIND_OUTBOUND
                for e in worker_a_received
            ),
            timeout=2.0,
        )
        out = next(
            e
            for e in worker_a_received
            if e.correlation_id == "abc" and e.kind == KIND_OUTBOUND
        )
        body = out.body if isinstance(out.body, dict) else {}
        assert body.get("content") == "done"
        await worker_a.close()
        await worker_b.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()
