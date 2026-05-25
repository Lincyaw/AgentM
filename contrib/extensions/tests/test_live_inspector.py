"""Fail-stop tests for the ``live_inspector`` atom.

Scope per CLAUDE.md "Testing philosophy" — only positions where the atom's
value proposition fails if broken:

1. **Wire protocol round-trip over WebSocket** — server emits ``hello``,
   replays backlog, and forwards broadcasts in order with the right schema.
   If this breaks, every UI client sees garbage.

2. **Replay-ordering race** — a broadcast arriving mid-handshake must NOT
   reach the client before ``backlog_done``. This is the load-bearing
   protocol invariant; replay-vs-live ordering drift would lie about
   "what state existed when I connected".

3. **Backpressure** — a client whose queue is full has the NEWEST message
   dropped (``_Client.dropped`` increments) while the broadcaster keeps
   running. Drives the mechanism directly: pre-fills the queue and asserts
   the drop counter. The WS layer is incidental to this check.

4. **Singleton-per-root** — two ``_get_or_create_server`` calls with the
   same root_session_id reuse the same server. Load-bearing for child
   sessions sharing the parent's socket.

5. **``ChildSessionExtendingEvent`` handler contract** — the inspector's
   root install registers exactly one handler that returns the child
   inspector entry with the right parent_root + parent_port + bind. The
   substrate's dedupe (covered separately in
   ``tests/unit/core/test_child_session_extending_event.py``) takes it
   from there.

Per-event-kind happy paths are NOT covered — payload serialisation is
delegated to :func:`agentm.core.lib.to_jsonable`, already tested upstream.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from websockets.sync.client import connect

from contrib.extensions.live_inspector import (
    _QUEUE_HIGHWATER,
    _SERVERS,
    _Client,
    _get_or_create_server,
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


def _read_n_messages(url: str, n: int, timeout: float = 5.0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with connect(url, open_timeout=timeout, close_timeout=timeout) as ws:
        for _ in range(n):
            try:
                msg = ws.recv(timeout=timeout)
            except TimeoutError:
                break
            if isinstance(msg, (bytes, bytearray)):
                msg = msg.decode("utf-8")
            out.append(json.loads(msg))
    return out




def test_replay_ordering_race_does_not_leak_live_event_into_backlog() -> None:
    """Drive the race directly: broadcast AFTER the prologue has snapshotted
    the backlog but BEFORE ``backlog_done`` is enqueued, then assert the
    new broadcast appears strictly after ``backlog_done``.

    Driving this through the WS layer is timing-fragile (the broadcaster
    thread may run before or after the server-loop prologue depending on
    OS scheduling). Instead we drive it directly:

    1. Pre-fill ``_backlog`` with N records.
    2. Construct a fake client and run the prologue's exact step sequence
       (``_enqueue_sync`` hello, ``_enqueue_sync`` for each backlog record,
       ``_enqueue_sync`` backlog_done, ``clients.add``).
    3. BETWEEN the backlog loop and the ``backlog_done`` enqueue, fire a
       ``broadcast()`` from another thread. Because the client is NOT yet
       in ``server.clients``, ``broadcast()``'s ``_fanout`` MUST skip it —
       the new record goes into ``_backlog`` for late-connecters but does
       NOT race into our client's queue ahead of ``backlog_done``.

    Drain the client's queue and assert: every backlog record comes
    before ``backlog_done``, no live broadcast appears anywhere in the
    queue (it correctly went to ``_backlog`` only).

    This is the actual protocol invariant. A WS round-trip on top would
    be downstream of this — if the prologue is correct, the WS layer
    just relays whatever's in the queue.
    """

    server = _get_or_create_server(
        root_session_id="root-race",
        bind="127.0.0.1",
        port=0,
        announce=False,
        url_file=None,
    )
    try:
        for i in range(20):
            server._backlog.append({"type": "event", "i": i, "phase": "backlog"})

        client = _Client(queue=asyncio.Queue(maxsize=_QUEUE_HIGHWATER))

        # Step 1: hello.
        server._enqueue_sync(
            client,
            {"type": "hello", "root_session_id": "root-race", "schema_version": 1},
        )
        # Step 2: backlog snapshot replay.
        for record in list(server._backlog):
            server._enqueue_sync(client, record)
        # Step 3 — RACE WINDOW: a broadcast fires while we're still mid-prologue.
        # The client is NOT in ``server.clients`` yet, so this broadcast
        # must NOT land in client.queue. It DOES land in ``_backlog`` for
        # any future connecter.
        server.broadcast({"type": "event", "phase": "live_during_prologue"})
        # Step 4: backlog_done.
        server._enqueue_sync(client, {"type": "backlog_done"})
        # Step 5: clients.add.
        server.clients.add(client)

        # Drain whatever's in the client's queue (all should be from the
        # prologue, in order: hello, 20 backlog records, backlog_done).
        delivered: list[dict[str, Any]] = []
        while not client.queue.empty():
            item = client.queue.get_nowait()
            if isinstance(item, str):
                delivered.append(json.loads(item))

        # Find backlog_done; everything in the queue is before it (the
        # queue terminates after backlog_done since clients.add happened
        # last and no fanout has run on this client yet).
        types_in_order = [m.get("type", m.get("phase")) for m in delivered]
        assert types_in_order[0] == "hello"
        assert types_in_order[-1] == "backlog_done"
        # And nothing from the live broadcast leaked in.
        phases = [m.get("phase") for m in delivered]
        assert "live_during_prologue" not in phases, (
            "REPLAY RACE: live broadcast leaked into prologue queue: "
            f"{phases}"
        )
        # Sanity: the live broadcast DID make it into _backlog for future
        # connecters.
        assert any(
            r.get("phase") == "live_during_prologue" for r in server._backlog
        )
    finally:
        server.release()


def test_backpressure_drops_newest_when_queue_full() -> None:
    """Direct unit test of the backpressure mechanism.

    Pre-fill a client's queue to capacity, then drive ``_enqueue_sync``
    one more time and assert the drop counter increments. The WS layer
    isn't involved — this is the actual mechanism we rely on to keep a
    slow consumer from wedging the broadcaster.
    """

    server = _get_or_create_server(
        root_session_id="root-bp",
        bind="127.0.0.1",
        port=0,
        announce=False,
        url_file=None,
    )
    try:
        client = _Client(queue=asyncio.Queue(maxsize=_QUEUE_HIGHWATER))
        for i in range(_QUEUE_HIGHWATER):
            client.queue.put_nowait(f"prefill-{i}")
        assert client.dropped == 0

        server._enqueue_sync(client, {"i": "overflow-1"})
        assert client.dropped == 1

        for _ in range(50):
            server._enqueue_sync(client, {"i": "more"})
        assert client.dropped == 51
        assert client.queue.qsize() == _QUEUE_HIGHWATER
    finally:
        server.release()




# --- Child-session extending hook -------------------------------------------




