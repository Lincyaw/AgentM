"""Asyncio wire server for the single-process gateway (§3.1).

Accepts chat-client connections, runs the hello/welcome handshake (with
wire-version negotiation and auth), reads inbound frames, and runs a
per-peer outbox delivery worker. There is one peer kind in v2 —
``chat_client`` — so the handshake carries no ``peer_kind``.

Delivery semantics — at-least-once
----------------------------------
The per-peer delivery worker leases rows from the outbox, writes them to
the socket, waits for ``writer.drain()``, then ``ack``s the rows. A peer
crash after receiving the frame but before processing means the row is
gone — the receiver-side ``InboxLog`` is the deduplication boundary.

Inbound semantics — at-most-once-with-ack-on-process
----------------------------------------------------
The server records each inbound id in the ``InboxLog`` before
dispatching; duplicates are skipped.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agentm.gateway.auth import (
    AllowAllAuthenticator,
    Authenticator,
    UnixPeerCredAuthenticator,
)
from agentm.gateway.outbox import (
    InboxLog,
    OutboxRecord,
    OutboxStore,
)
from agentm.gateway.peer import PeerRegistry, PeerSession
from agentm.gateway.transport import (
    ServerTransport,
    UnixServerTransport,
    WebSocketServerTransport,
)
from agentm.gateway.wire import (
    HEADER_BYTES,
    KIND_ACK,
    KIND_ERROR,
    KIND_HELLO,
    KIND_PING,
    KIND_PONG,
    KIND_INBOUND,
    KIND_WELCOME,
    WIRE_VERSION,
    Envelope,
    IncompleteFrame,
    InvalidEnvelope,
    WireError,
    decode_stream,
    encode,
)

log = logging.getLogger("agentm.gateway.server")

SERVER_VERSION: str = "0.2.0"
MAX_DELIVERY_ATTEMPTS: int = 5
# Safety-net poll: the delivery worker waits on an asyncio.Event woken by
# outbox.enqueue, but it also re-checks every LEASE_REFRESH_INTERVAL
# seconds so leases that expired without a fresh enqueue (crash recovery)
# eventually get re-leased.
LEASE_REFRESH_INTERVAL: float = 5.0

InboundHandler = Callable[[PeerSession, Envelope], Awaitable[None]]


class _ConnectionLost(Exception):
    """Internal sentinel — socket dead, exit the per-peer delivery loop."""


class WireServer:
    """Asyncio wire server.

    Each accepted connection: handshake → register peer → spawn a
    delivery worker that drains the outbox → read inbound frames in the
    connection task.

    Lifecycle: ``await start()``; ``await stop()`` (idempotent — removes
    the socket file).
    """

    def __init__(
        self,
        *,
        outbox: OutboxStore,
        inbox: InboxLog,
        on_inbound: InboundHandler,
        transport: ServerTransport | None = None,
        socket_path: str | None = None,
        authenticator: Authenticator | None = None,
        delivery_batch_max: int = 32,
        slow_consumer_high_water: int = 1000,
        max_delivery_attempts: int = MAX_DELIVERY_ATTEMPTS,
    ) -> None:
        # ``socket_path=`` is a Unix-socket convenience shortcut equivalent to
        # ``transport=UnixServerTransport(socket_path)``; pass exactly one.
        if transport is None:
            if socket_path is None:
                raise TypeError(
                    "WireServer requires either transport= or socket_path="
                )
            transport = UnixServerTransport(socket_path)
        elif socket_path is not None:
            raise TypeError(
                "WireServer: pass either transport= or socket_path=, not both"
            )
        # Peer-cred auth depends on AF_UNIX kernel credentials; pairing it
        # with a WebSocket transport would silently degrade to
        # rejection-of-everything (no socket extra_info). Fail fast.
        if isinstance(transport, WebSocketServerTransport) and isinstance(
            authenticator, UnixPeerCredAuthenticator
        ):
            raise ValueError(
                "UnixPeerCredAuthenticator is incompatible with "
                "WebSocketServerTransport; use TokenAuthenticator over WS"
            )
        self._transport = transport
        self._outbox = outbox
        self._inbox = inbox
        self._on_inbound = on_inbound
        self._auth: Authenticator = authenticator or AllowAllAuthenticator()
        self._delivery_batch_max = delivery_batch_max
        self._high_water = slow_consumer_high_water
        self._low_water = max(1, slow_consumer_high_water // 2)
        self._max_attempts = max_delivery_attempts
        # First-class Protocol methods (§6) — no getattr probing.
        self._registry = PeerRegistry()
        self._started = False
        self._stopped = False
        self._conn_tasks: set[asyncio.Task[None]] = set()
        self._delivery_tasks: dict[str, asyncio.Task[None]] = {}
        self._wake_events: dict[str, asyncio.Event] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        await self._transport.serve(self._handle_connection)
        self._started = True

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self._transport.close()
        for task in list(self._delivery_tasks.values()):
            task.cancel()
        for task in list(self._conn_tasks):
            task.cancel()
        all_tasks = list(self._delivery_tasks.values()) + list(self._conn_tasks)
        for task in all_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._delivery_tasks.clear()
        self._conn_tasks.clear()

    @property
    def registry(self) -> PeerRegistry:
        return self._registry

    # -- connection handling -----------------------------------------

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._conn_tasks.add(task)
        peer_id: str | None = None
        try:
            hello = await _read_one_frame(reader)
            if hello is None:
                # No valid v2 frame — either EOF or a v1 envelope that the
                # decoder rejected on the version check. Tell the peer the
                # wire version is unsupported (§2).
                await _send(
                    writer,
                    _error_env(
                        "unsupported_wire_version",
                        f"expected a v{WIRE_VERSION} hello as the first frame",
                    ),
                )
                return
            if hello.kind != KIND_HELLO:
                await _send(
                    writer,
                    _error_env("expected_hello", "expected hello as first frame"),
                )
                return
            body = hello.body if isinstance(hello.body, dict) else {}
            peer_id = str(body.get("peer_name") or body.get("peer_id") or "")
            auth_raw = body.get("auth")
            token_raw = body.get("token") or (
                auth_raw if isinstance(auth_raw, str) else None
            )
            token = str(token_raw) if isinstance(token_raw, str) else None
            if not peer_id:
                await _send(
                    writer,
                    _error_env("bad_hello", "hello body missing peer_name"),
                )
                return
            ok = await self._auth.authenticate(peer_id, token, writer)
            if not ok:
                await _send(
                    writer,
                    _error_env("auth_failed", f"auth failed for peer {peer_id!r}"),
                )
                return
            if peer_id in self._registry:
                await _send(
                    writer,
                    _error_env(
                        "duplicate_peer", f"peer {peer_id!r} already connected"
                    ),
                )
                return

            session = PeerSession(
                peer_id=peer_id,
                transport_writer=writer,
                last_seen=time.time(),
                capabilities=dict(body.get("capabilities") or {}),
            )
            self._registry.register(session)
            await _send(
                writer,
                _make_env(
                    KIND_WELCOME,
                    {
                        "server_version": SERVER_VERSION,
                        "wire_version": WIRE_VERSION,
                        "peer_id": peer_id,
                        "session_resume": [],
                    },
                ),
            )

            wake = asyncio.Event()
            self._wake_events[peer_id] = wake
            self._register_outbox_notifier(peer_id, wake)
            # An enqueue may have landed before we registered: prime the
            # event so the first drain pass runs.
            wake.set()
            delivery = asyncio.create_task(
                self._delivery_loop(session, wake), name=f"deliver:{peer_id}"
            )
            self._delivery_tasks[peer_id] = delivery
            try:
                await self._read_loop(session, reader)
            finally:
                delivery.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await delivery
                self._delivery_tasks.pop(peer_id, None)
                self._wake_events.pop(peer_id, None)
                self._unregister_outbox_notifier(peer_id)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("connection handler crashed")
        finally:
            if peer_id is not None:
                self._registry.deregister(peer_id)
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()
            if task is not None:
                self._conn_tasks.discard(task)

    async def _read_loop(
        self, session: PeerSession, reader: asyncio.StreamReader
    ) -> None:
        buf = b""
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                return  # peer closed
            buf += chunk
            try:
                envelopes, buf = decode_stream(buf)
            except (InvalidEnvelope, WireError) as exc:
                await _send(
                    session.transport_writer,
                    _error_env("bad_frame", f"frame rejected: {exc}"),
                )
                return
            for env in envelopes:
                session.last_seen = time.time()
                await self._dispatch_frame(session, env)

    async def _dispatch_frame(self, session: PeerSession, env: Envelope) -> None:
        if env.kind == KIND_INBOUND:
            inserted = await asyncio.to_thread(
                self._inbox.record_seen, session.peer_id, env.id, env.ts
            )
            if not inserted:
                return  # duplicate — skip handler
            try:
                await self._on_inbound(session, env)
            except Exception:
                log.exception("on_inbound handler raised for env id=%s", env.id)
            return
        if env.kind == KIND_ACK:
            return  # server-side acks are implicit; tolerate client-sent ones.
        if env.kind == KIND_PING:
            await _send(session.transport_writer, _make_env(KIND_PONG, {}))
            return
        if env.kind == KIND_PONG:
            return
        log.debug(
            "server: ignoring unexpected kind %s from %s", env.kind, session.peer_id
        )

    # -- delivery worker ---------------------------------------------

    async def _delivery_loop(
        self, session: PeerSession, wake: asyncio.Event
    ) -> None:
        """Drain outbox rows for ``session.peer_id`` until cancelled.

        Wakeup model: blocks on ``wake`` (set by ``outbox.enqueue`` via
        the registered notifier) with a ``LEASE_REFRESH_INTERVAL``
        timeout (the crash-recovery safety net). Pipelined push: each
        leased record ships as one ``outbound`` envelope, one wire write
        per record.

        Slow-consumer accounting is observational: we log a diagnostic
        when the pending queue crosses ``slow_consumer_high_water`` and
        clear the flag below ``high_water / 2``; we never stop draining.
        """
        peer_id = session.peer_id
        try:
            while True:
                pending = await asyncio.to_thread(self._outbox.pending_count, peer_id)
                session.pending_count_hint = pending
                if pending > self._high_water and not session.backpressure:
                    log.warning(
                        "slow_consumer peer=%s pending=%d high_water=%d",
                        peer_id,
                        pending,
                        self._high_water,
                    )
                    session.backpressure = True
                elif session.backpressure and pending <= self._low_water:
                    session.backpressure = False
                wake.clear()
                now = time.time()
                records = await asyncio.to_thread(
                    self._outbox.lease, peer_id, self._delivery_batch_max, now
                )
                if not records:
                    next_retry = await asyncio.to_thread(
                        self._outbox.next_retry_at_min, peer_id
                    )
                    if next_retry is not None and next_retry > now:
                        timeout = min(
                            max(next_retry - now, 0.0), LEASE_REFRESH_INTERVAL
                        )
                    else:
                        timeout = LEASE_REFRESH_INTERVAL
                    try:
                        await asyncio.wait_for(wake.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        pass
                    continue
                await self._deliver(session, records)
        except _ConnectionLost:
            with contextlib.suppress(Exception):
                session.transport_writer.close()
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("delivery loop for peer=%s crashed", peer_id)

    async def _deliver(
        self, session: PeerSession, records: list[OutboxRecord]
    ) -> None:
        """Write each record as one ``outbound`` envelope and ack the set.

        On socket failure: account the records (nack or dead-letter per
        attempts) and re-raise ``_ConnectionLost`` so the delivery loop
        exits until the peer reconnects.
        """
        writer = session.transport_writer
        try:
            # Hold the per-peer write lock across the batch so durable frames
            # never interleave with ephemeral live frames on the same writer
            # (one wire frame == one WS message; see PeerSession.write_lock).
            async with session.write_lock:
                for r in records:
                    writer.write(encode(r.envelope))
                    await writer.drain()
        except asyncio.CancelledError:
            await self._nack_records(records)
            raise
        except (ConnectionResetError, BrokenPipeError) as exc:
            await self._handle_delivery_failure(session, records, repr(exc))
            raise _ConnectionLost() from exc
        except Exception as exc:
            await self._handle_delivery_failure(session, records, repr(exc))
            raise _ConnectionLost() from exc
        await asyncio.to_thread(self._outbox.ack, [r.id for r in records])

    async def _nack_records(self, records: list[OutboxRecord]) -> None:
        now = time.time()
        for r in records:
            delay = self._outbox.backoff_delay(r.attempts)
            await asyncio.to_thread(self._outbox.nack, [r.id], now + delay)

    async def _handle_delivery_failure(
        self,
        session: PeerSession,
        records: list[OutboxRecord],
        reason: str,
    ) -> None:
        now = time.time()
        for r in records:
            if r.attempts >= self._max_attempts:
                await asyncio.to_thread(self._outbox.dead_letter, r.id, reason)
                log.warning(
                    "dead_letter peer=%s env_id=%s reason=%s",
                    session.peer_id,
                    r.envelope.id,
                    reason,
                )
            else:
                delay = self._outbox.backoff_delay(r.attempts)
                await asyncio.to_thread(self._outbox.nack, [r.id], now + delay)

    # -- outbox notifier wiring --------------------------------------

    def _register_outbox_notifier(self, peer_id: str, event: asyncio.Event) -> None:
        """Register a per-peer wakeup with the outbox (§6 first-class)."""
        loop = self._loop
        if loop is None:
            return

        def notify(_pid: str) -> None:
            # outbox.enqueue runs in worker threads (via asyncio.to_thread).
            # Bounce the set() through the loop.
            if not loop.is_closed():
                loop.call_soon_threadsafe(event.set)

        self._outbox.set_notifier(peer_id, notify)

    def _unregister_outbox_notifier(self, peer_id: str) -> None:
        self._outbox.set_notifier(peer_id, None)


# -- helpers ---------------------------------------------------------


async def _read_one_frame(reader: asyncio.StreamReader) -> Envelope | None:
    """Read exactly one framed envelope. Returns None on EOF/parse error."""
    try:
        header = await reader.readexactly(HEADER_BYTES)
    except asyncio.IncompleteReadError:
        return None
    length = int.from_bytes(header, "big")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None
    try:
        envelopes, _rest = decode_stream(header + body)
    except (InvalidEnvelope, WireError, IncompleteFrame):
        return None
    return envelopes[0] if envelopes else None


def _make_env(kind: str, body: dict[str, Any]) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=f"srv-{kind}-{os.urandom(6).hex()}",
        kind=kind,
        ts=time.time(),
        body=body,
    )


def _error_env(code: str, message: str) -> Envelope:
    return _make_env(KIND_ERROR, {"code": code, "message": message, "fatal": True})


async def _send(writer: asyncio.StreamWriter, env: Envelope) -> None:
    writer.write(encode(env))
    try:
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass


__all__ = [
    "MAX_DELIVERY_ATTEMPTS",
    "SERVER_VERSION",
    "WireServer",
]
