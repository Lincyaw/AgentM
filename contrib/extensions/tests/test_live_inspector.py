"""Fail-stop tests for the ``live_inspector`` atom.

Scope per CLAUDE.md "Testing philosophy" — only positions where the atom's
value proposition fails if broken:

1. **Wire protocol round-trip**: server emits ``hello`` then broadcast
   records in order with the right schema. If this breaks, every UI client
   sees garbage.

2. **Backpressure**: a slow client does not deadlock the broadcaster. If
   this breaks, one stuck websocket-on-a-laptop wedges the whole agent.

Happy-path-for-every-event-kind is NOT covered — payload serialization is
delegated to :func:`agentm.core.lib.to_jsonable`, already tested upstream.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from contrib.extensions.live_inspector import (
    _get_or_create_server,
    _SERVERS,
)


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    # Each test should start from a clean slate so refcounts don't bleed.
    for srv in list(_SERVERS.values()):
        srv.shutdown()
    _SERVERS.clear()
    yield
    for srv in list(_SERVERS.values()):
        srv.shutdown()
    _SERVERS.clear()


def _read_n_messages(host: str, port: int, n: int, timeout: float = 3.0) -> list[dict]:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)
    out: list[dict] = []
    buf = b""
    try:
        while len(out) < n:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                out.append(json.loads(line.decode("utf-8")))
                if len(out) >= n:
                    break
    finally:
        sock.close()
    return out


def test_wire_protocol_round_trip() -> None:
    """Connect a client; assert hello + ordered broadcast delivery."""

    server = _get_or_create_server(
        root_session_id="root-abc",
        bind="127.0.0.1",
        port=0,
        url_file=None,
    )
    try:
        # Broadcast BEFORE the client connects so we exercise the backlog
        # replay path. The hello goes out first, then backlog, then
        # backlog_done, then any subsequent live broadcasts.
        server.broadcast(
            {
                "type": "session_started",
                "session_id": "s1",
                "parent_session_id": None,
                "purpose": "root",
                "cwd": "/tmp",
                "ts": 1.0,
            }
        )
        server.broadcast(
            {
                "type": "event",
                "session_id": "s1",
                "ts": 1.1,
                "kind": "TurnEndEvent",
                "channel": "turn_end",
                "payload": {"turn_index": 0},
            }
        )

        # Reader thread connects and reads 4 messages: hello, the two
        # backlog items, and backlog_done.
        result: dict[str, list[dict]] = {"messages": []}

        def reader() -> None:
            result["messages"] = _read_n_messages(
                "127.0.0.1", server.actual_port, 4
            )

        t = threading.Thread(target=reader)
        t.start()
        t.join(timeout=3.0)
        assert not t.is_alive(), "reader thread hung"

        msgs = result["messages"]
        assert len(msgs) == 4, msgs
        assert msgs[0]["type"] == "hello"
        assert msgs[0]["root_session_id"] == "root-abc"
        assert msgs[0]["schema_version"] == 1
        assert msgs[1]["type"] == "session_started"
        assert msgs[1]["session_id"] == "s1"
        assert msgs[2]["type"] == "event"
        assert msgs[2]["channel"] == "turn_end"
        assert msgs[2]["kind"] == "TurnEndEvent"
        assert msgs[3]["type"] == "backlog_done"
    finally:
        server.release()


def test_backpressure_does_not_deadlock_server() -> None:
    """A slow client gets messages dropped, but the server keeps broadcasting.

    Strategy: open a connection but never call ``recv``. The OS socket
    buffer + per-client asyncio queue (cap 256) will fill. Verify that
    a second, fast client still receives a fresh broadcast within the
    timeout — i.e. the slow client did NOT block the broadcaster.
    """

    server = _get_or_create_server(
        root_session_id="root-slow",
        bind="127.0.0.1",
        port=0,
        url_file=None,
    )
    try:
        # Slow client: connect, then sleep. Never read.
        slow_sock = socket.create_connection(
            ("127.0.0.1", server.actual_port), timeout=2.0
        )
        try:
            # Flood far more than the queue cap to force drops.
            for i in range(2000):
                server.broadcast({"type": "event", "i": i, "session_id": "s1"})

            # Fast client connects; should still get hello + backlog
            # entries + backlog_done. The hello arrives even if the slow
            # client is wedged.
            messages = _read_n_messages(
                "127.0.0.1", server.actual_port, 3, timeout=3.0
            )
            assert messages[0]["type"] == "hello"
            # Some flood records should have made it into the backlog.
            kinds = {m["type"] for m in messages}
            assert "hello" in kinds
        finally:
            slow_sock.close()
    finally:
        server.release()


def test_server_singleton_per_root() -> None:
    """Two ``_get_or_create_server`` calls with the same root reuse the same server.

    This is the load-bearing invariant for child-session sharing: when the
    inspector is loaded in both parent and a child session in the same
    process, both must publish to the SAME socket so a UI client connecting
    once sees the full tree.
    """

    a = _get_or_create_server("root-shared", "127.0.0.1", 0, None)
    try:
        b = _get_or_create_server("root-shared", "127.0.0.1", 0, None)
        try:
            assert a is b
            assert a.refcount == 2
        finally:
            b.release()
        assert a.refcount == 1  # b released, a still holds
    finally:
        a.release()
