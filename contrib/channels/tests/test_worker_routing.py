"""Gateway-side ``agent_worker`` routing tests (Phase 5a).

Drives a real :class:`WireServer` with a :class:`WireBridge` and
attaches mock workers via :class:`WireClient`. Verifies the routing
policy: scenario-equality, sticky session_key → worker, fallback
behaviour with ``allow_inproc=False``, and the strict
"worker-disconnected" error path.

No AgentSession, no MessageBus consumer, no subprocess — the bridge
is exercised directly in-process. The Gateway is not instantiated; we
only check what lands on the wire (worker outboxes) and what the
bridge enqueues on the chat client's outbox.
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
    with tempfile.TemporaryDirectory(prefix="agentm-worker-rt-") as d:
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


def _make_inbound(
    *,
    peer_id: str,
    channel: str = "terminal",
    chat_id: str = "c1",
    content: str = "hello",
    env_id: str | None = None,
) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=env_id or f"in-{peer_id}-{int(time.time() * 1_000_000)}",
        kind=KIND_INBOUND,
        ts=time.time(),
        body={
            "channel": channel,
            "chat_id": chat_id,
            "sender_id": "u1",
            "content": content,
        },
    )


async def _start_server(
    socket_path: str,
    db_path: str,
    *,
    scenario: str,
    allow_inproc: bool,
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


# -- tests -------------------------------------------------------------


async def test_worker_registers_on_hello(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path, scenario="x", allow_inproc=False
    )
    try:
        worker = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x", "y"]},
        )
        await worker.connect()
        # Hello-side registration is async; wait for it to land.
        ok = await _wait(lambda: bridge.worker_registry.has("worker-A"))
        assert ok
        info = bridge.worker_registry.workers()[0]
        assert info.peer_id == "worker-A"
        assert info.scenarios == frozenset({"x", "y"})
        # find_worker should return this peer for both scenarios.
        assert bridge.worker_registry.find_worker("x", "k1") == "worker-A"
        # Sticky binding: second lookup returns the same answer.
        assert bridge.worker_registry.find_worker("x", "k1") == "worker-A"
        await worker.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


async def test_inbound_routed_to_matching_worker(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path, scenario="x", allow_inproc=False
    )
    received: list[Envelope] = []

    async def collect(env: Envelope) -> None:
        received.append(env)

    try:
        worker = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect,
        )
        await worker.connect()
        # Chat client peer that the bridge will register as a synthetic
        # channel on first inbound.
        chat_replies: list[Envelope] = []

        async def chat_recv(env: Envelope) -> None:
            chat_replies.append(env)

        chat = WireClient(
            socket_path=socket_path,
            peer_id="chat-A",
            peer_kind="chat_client",
            on_outbound=chat_recv,
        )
        await chat.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-A"))

        # Scenario "x" should route to the worker.
        await chat.send(_make_inbound(peer_id="chat-A", env_id="m1"))
        assert await _wait(lambda: len(received) >= 1, timeout=2.0)
        fwd = received[0]
        assert fwd.kind == KIND_INBOUND
        assert fwd.body["channel"] == "terminal"
        assert fwd.body["chat_id"] == "c1"
        assert fwd.body["content"] == "hello"

        await chat.close()
        await worker.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


async def test_no_worker_message_when_inproc_disabled(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path, scenario="y", allow_inproc=False
    )
    try:
        # A worker that advertises a *different* scenario.
        worker = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
        )
        await worker.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-A"))

        chat_replies: list[Envelope] = []

        async def chat_recv(env: Envelope) -> None:
            chat_replies.append(env)

        chat = WireClient(
            socket_path=socket_path,
            peer_id="chat-A",
            peer_kind="chat_client",
            on_outbound=chat_recv,
        )
        await chat.connect()
        await chat.send(_make_inbound(peer_id="chat-A", env_id="m1"))

        # The bridge should publish a "no worker available" outbound
        # on the bus → ChannelManager → … but with no in-process
        # channel manager wiring the synthetic channel back to the
        # bus consumer, we instead verify the outbound was published.
        # The synthetic _WireChannel injected by the bridge enqueues
        # the outbound onto the chat peer's outbox; the manager loop
        # forwards it. Without a started manager, no forwarding loop —
        # so we wait for either an outbound on the chat client
        # OR a publish on the bus.
        ok = await _wait(
            lambda: len(chat_replies) >= 1 or not _bus_empty(_bus_of(bridge)),
            timeout=2.0,
        )
        assert ok
        await chat.close()
        await worker.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


async def test_sticky_session_key(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path, scenario="x", allow_inproc=False
    )
    received_a: list[Envelope] = []
    received_b: list[Envelope] = []

    async def collect_a(env: Envelope) -> None:
        received_a.append(env)

    async def collect_b(env: Envelope) -> None:
        received_b.append(env)

    try:
        worker_a = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect_a,
        )
        await worker_a.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-A"))
        worker_b = WireClient(
            socket_path=socket_path,
            peer_id="worker-B",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect_b,
        )
        await worker_b.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-B"))

        chat = WireClient(
            socket_path=socket_path,
            peer_id="chat-A",
            peer_kind="chat_client",
        )
        await chat.connect()

        await chat.send(_make_inbound(peer_id="chat-A", env_id="m1"))
        assert await _wait(lambda: len(received_a) >= 1, timeout=2.0)
        # First-match-wins binds worker-A.
        assert bridge.worker_registry.sticky_owner("terminal:c1") == "worker-A"

        # Same session_key → still worker-A; worker-B sees nothing.
        await chat.send(_make_inbound(peer_id="chat-A", env_id="m2"))
        assert await _wait(lambda: len(received_a) >= 2, timeout=2.0)
        assert len(received_b) == 0

        await chat.close()
        await worker_a.close()
        await worker_b.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


async def test_worker_disconnect_releases_sticky_and_emits_error(
    socket_path: str, db_path: str
) -> None:
    server, bridge, _bus, outbox, inbox = await _start_server(
        socket_path, db_path, scenario="x", allow_inproc=False
    )
    received_a: list[Envelope] = []
    received_b: list[Envelope] = []
    chat_replies: list[Envelope] = []

    async def collect_a(env: Envelope) -> None:
        received_a.append(env)

    async def collect_b(env: Envelope) -> None:
        received_b.append(env)

    async def chat_recv(env: Envelope) -> None:
        chat_replies.append(env)

    try:
        worker_a = WireClient(
            socket_path=socket_path,
            peer_id="worker-A",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect_a,
        )
        await worker_a.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-A"))

        chat = WireClient(
            socket_path=socket_path,
            peer_id="chat-A",
            peer_kind="chat_client",
            on_outbound=chat_recv,
        )
        await chat.connect()

        await chat.send(_make_inbound(peer_id="chat-A", env_id="m1"))
        assert await _wait(lambda: len(received_a) >= 1, timeout=2.0)
        assert bridge.worker_registry.sticky_owner("terminal:c1") == "worker-A"

        # Worker disconnects mid-turn.
        await worker_a.close()
        # Wait for the bridge to observe the disconnect and emit the
        # strict-policy error outbound to the chat client.
        ok = await _wait(
            lambda: any(
                env.kind == KIND_OUTBOUND
                and "worker disconnected" in str(env.body.get("content", ""))
                for env in chat_replies
            ),
            timeout=3.0,
        )
        assert ok, f"chat replies: {[(e.kind, e.body) for e in chat_replies]}"
        # Sticky binding must be cleared.
        assert bridge.worker_registry.sticky_owner("terminal:c1") is None

        # Bring up worker-B and a fresh inbound binds to it.
        worker_b = WireClient(
            socket_path=socket_path,
            peer_id="worker-B",
            peer_kind="agent_worker",
            capabilities={"scenarios": ["x"]},
            on_outbound=collect_b,
        )
        await worker_b.connect()
        assert await _wait(lambda: bridge.worker_registry.has("worker-B"))
        await chat.send(_make_inbound(peer_id="chat-A", env_id="m2"))
        assert await _wait(lambda: len(received_b) >= 1, timeout=2.0)
        assert bridge.worker_registry.sticky_owner("terminal:c1") == "worker-B"

        await chat.close()
        await worker_b.close()
    finally:
        await server.stop()
        outbox.close()
        inbox.close()


# -- helpers ----------------------------------------------------------


def _bus_of(bridge: WireBridge) -> MessageBus:
    return bridge._bus  # type: ignore[attr-defined]


def _bus_empty(bus: MessageBus) -> bool:
    return bus.outbound.empty()
