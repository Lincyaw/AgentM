"""Asyncio Unix-socket server for the channels gateway/client split.

See ``.claude/designs/client-server-architecture.md`` §3, §4, §4.5.
This is a *parallel* mechanism to the v0 in-process gateway — Phase 1.4
will wire it up as ``agentm-gateway --bind``. Until then this module
is consumed only by integration tests and (in subsequent phases) by
extracted channel processes via :class:`WireClient`.

Delivery semantics — at-least-once
----------------------------------
The per-peer delivery worker leases rows from the outbox, writes them
to the socket, waits for ``writer.drain()``, then ``ack``s the rows.
A peer crash *after* receiving the frame but *before* application
processing means the row is gone — the receiver-side ``InboxLog`` is
the deduplication boundary. The simpler interpretation (ack on
``drain()``) is chosen deliberately:

* The alternative (server waits for an explicit per-row ``ack`` over
  the wire) doubles wire chatter and requires per-row state on the
  receiving peer; for v1, receiver-side idempotency is enough.
* If the application needs ack-on-process semantics it sends a fresh
  ``inbound`` envelope back through the gateway; the inbox dedupes.

Authenticator hook
------------------
:class:`Authenticator` is a Protocol; v1 ships :class:`AllowAllAuthenticator`
as the default. Phase 1.4 ships :class:`UnixPeerCredAuthenticator`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from agentm_channels.outbox import (
    InboxLog,
    OutboxRecord,
    OutboxStore,
    exponential_backoff,
)
from agentm_channels.peer import PeerRegistry, PeerSession
from agentm_channels.wire import (
    HEADER_BYTES,
    KIND_ACK,
    KIND_ACK_BATCH,
    KIND_BYE,
    KIND_DELIVERY_BATCH,
    KIND_ERROR,
    KIND_HELLO,
    KIND_INBOUND,
    KIND_PING,
    KIND_PONG,
    KIND_WELCOME,
    WIRE_VERSION,
    Envelope,
    IncompleteFrame,
    InvalidEnvelope,
    WireError,
    decode_stream,
    encode,
)

log = logging.getLogger("agentm_channels.server")

SERVER_VERSION: str = "0.1.0"
MAX_DELIVERY_ATTEMPTS: int = 5
DELIVERY_IDLE_POLL_SECONDS: float = 0.05

InboundHandler = Callable[[PeerSession, Envelope], Awaitable[None]]


class _ConnectionLost(Exception):
    """Internal sentinel — socket dead, exit the per-peer delivery loop."""


@runtime_checkable
class Authenticator(Protocol):
    """Server-side hook for accepting/rejecting a ``hello``."""

    async def authenticate(
        self,
        peer_kind: str,
        peer_id: str,
        token: str | None,
        transport: asyncio.StreamWriter,
    ) -> bool:
        ...


class AllowAllAuthenticator:
    """Default — accept every hello. Suitable for tests + Unix socket
    on trusted hosts. Phase 1.4 replaces this with a peer-cred check.
    """

    async def authenticate(
        self,
        peer_kind: str,
        peer_id: str,
        token: str | None,
        transport: asyncio.StreamWriter,
    ) -> bool:
        return True


class WireServer:
    """Asyncio Unix-socket server.

    Each accepted connection: handshake → register peer → spawn a
    delivery worker that drains the outbox → read inbound frames in
    the connection task.

    Lifecycle: ``await start()``; ``await stop()`` (idempotent — removes
    the socket file).
    """

    def __init__(
        self,
        socket_path: str,
        outbox: OutboxStore,
        inbox: InboxLog,
        on_inbound: InboundHandler,
        *,
        authenticator: Authenticator | None = None,
        delivery_batch_max: int = 32,
        lease_ttl: float = 30.0,
        slow_consumer_high_water: int = 1000,
        max_delivery_attempts: int = MAX_DELIVERY_ATTEMPTS,
    ) -> None:
        self._socket_path = socket_path
        self._outbox = outbox
        self._inbox = inbox
        self._on_inbound = on_inbound
        self._auth: Authenticator = authenticator or AllowAllAuthenticator()
        self._delivery_batch_max = delivery_batch_max
        self._lease_ttl = lease_ttl
        self._high_water = slow_consumer_high_water
        self._low_water = max(1, slow_consumer_high_water // 2)
        self._max_attempts = max_delivery_attempts
        self._registry = PeerRegistry()
        self._server: asyncio.base_events.Server | None = None
        self._stopped = False
        self._conn_tasks: set[asyncio.Task[None]] = set()
        self._delivery_tasks: dict[str, asyncio.Task[None]] = {}

    # -- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        if self._server is not None:
            return
        # Clean any stale socket left by a crashed predecessor.
        with contextlib.suppress(FileNotFoundError):
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        # Cancel all delivery workers and connection tasks.
        for task in list(self._delivery_tasks.values()):
            task.cancel()
        for task in list(self._conn_tasks):
            task.cancel()
        # Drain the cancellations.
        all_tasks = list(self._delivery_tasks.values()) + list(self._conn_tasks)
        for task in all_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._delivery_tasks.clear()
        self._conn_tasks.clear()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._socket_path)

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
            if hello is None or hello.kind != KIND_HELLO:
                await _send(writer, _error_env("expected_hello", "expected hello as first frame"))
                return
            body = hello.body if isinstance(hello.body, dict) else {}
            peer_kind = str(body.get("peer_kind", ""))
            peer_id = str(body.get("peer_id", ""))
            token_raw = body.get("token")
            token = str(token_raw) if isinstance(token_raw, str) else None
            if not peer_kind or not peer_id:
                await _send(writer, _error_env("bad_hello", "hello body missing peer_kind/peer_id"))
                return
            ok = await self._auth.authenticate(peer_kind, peer_id, token, writer)
            if not ok:
                await _send(writer, _error_env("auth_failed", f"auth failed for peer {peer_id!r}"))
                return
            if peer_id in self._registry:
                await _send(writer, _error_env("duplicate_peer", f"peer {peer_id!r} already connected"))
                return

            session = PeerSession(
                peer_id=peer_id,
                peer_kind=peer_kind,
                transport_writer=writer,
                last_seen=time.time(),
                capabilities=dict(body.get("capabilities") or {}),
            )
            self._registry.register(session)
            await _send(
                writer,
                _make_env(KIND_WELCOME, {"server_version": SERVER_VERSION, "peer_id_echo": peer_id}),
            )

            delivery = asyncio.create_task(
                self._delivery_loop(session), name=f"deliver:{peer_id}"
            )
            self._delivery_tasks[peer_id] = delivery
            try:
                await self._read_loop(session, reader)
            finally:
                delivery.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await delivery
                self._delivery_tasks.pop(peer_id, None)
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
                if env.kind == KIND_BYE:
                    return

    async def _dispatch_frame(self, session: PeerSession, env: Envelope) -> None:
        if env.kind == KIND_INBOUND:
            inserted = await asyncio.to_thread(
                self._inbox.record_seen, session.peer_id, env.id, env.ts
            )
            if not inserted:
                # Duplicate — skip handler, but keep transport happy.
                return
            try:
                await self._on_inbound(session, env)
            except Exception:
                log.exception("on_inbound handler raised for env id=%s", env.id)
            return
        if env.kind in (KIND_ACK, KIND_ACK_BATCH):
            return  # server-side acks are implicit; tolerate client-sent ones.
        if env.kind == KIND_PING:
            await _send(
                session.transport_writer,
                _make_env(KIND_PONG, {"echo_id": env.id}),
            )
            return
        if env.kind == KIND_BYE:
            return
        log.debug("server: ignoring unexpected kind %s from %s", env.kind, session.peer_id)

    # -- delivery worker ---------------------------------------------

    async def _delivery_loop(self, session: PeerSession) -> None:
        """Drain outbox rows for ``session.peer_id`` until cancelled.

        Slow-consumer accounting is observational: we *log a diagnostic*
        when the pending queue crosses ``slow_consumer_high_water`` and
        clear the flag when it falls below ``high_water / 2``. We do
        **not** stop draining — back-pressure belongs upstream of the
        outbox (the enqueue side); the delivery worker's job is to
        empty the queue as fast as the peer will accept. Per §0
        simplicity, no ``slow_consumer`` wire event.
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
                now = time.time()
                records = await asyncio.to_thread(
                    self._outbox.lease, peer_id, self._delivery_batch_max, now
                )
                if not records:
                    await asyncio.sleep(DELIVERY_IDLE_POLL_SECONDS)
                    continue
                await self._deliver(session, records)
        except _ConnectionLost:
            # Socket dead — stop draining. The connection task will
            # close the reader side and trigger reconnect logic peer-side.
            try:
                session.transport_writer.close()
            except Exception:  # noqa: BLE001
                pass
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("delivery loop for peer=%s crashed", peer_id)

    async def _deliver(
        self, session: PeerSession, records: list[OutboxRecord]
    ) -> None:
        """Write one batch (or single envelope) to the peer.

        On socket write/drain failure: account the records (nack or
        dead-letter per attempts) and *re-raise* a sentinel so the
        delivery loop exits — the connection is dead, sit out until
        the peer reconnects. Without this the loop would re-lease the
        same records every tick and exhaust ``max_attempts`` against
        a closed socket.
        """
        writer = session.transport_writer
        try:
            if len(records) == 1:
                env = records[0].envelope
                writer.write(encode(env))
                await writer.drain()
            else:
                items = [r.envelope.to_dict() for r in records]
                batch = _make_env(KIND_DELIVERY_BATCH, {"items": items})
                writer.write(encode(batch))
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
        # Success: ack the whole set.
        await asyncio.to_thread(self._outbox.ack, [r.id for r in records])

    async def _nack_records(self, records: list[OutboxRecord]) -> None:
        now = time.time()
        for r in records:
            delay = exponential_backoff(r.attempts)
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
                root = r.envelope.root_session_key
                if root is not None:
                    log.warning(
                        "dead_letter peer=%s env_id=%s reason=%s root=%s",
                        session.peer_id, r.envelope.id, reason, root,
                    )
                else:
                    log.warning(
                        "dead_letter peer=%s env_id=%s reason=%s",
                        session.peer_id, r.envelope.id, reason,
                    )
            else:
                delay = exponential_backoff(r.attempts)
                await asyncio.to_thread(self._outbox.nack, [r.id], now + delay)


# -- helpers ---------------------------------------------------------

async def _read_one_frame(reader: asyncio.StreamReader) -> Envelope | None:
    """Read exactly one framed envelope. Returns None on EOF/parse error."""
    buf = b""
    try:
        header = await reader.readexactly(HEADER_BYTES)
    except asyncio.IncompleteReadError:
        return None
    buf += header
    # Peek body length from the header.
    length = int.from_bytes(header, "big")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None
    buf += body
    try:
        envelopes, _rest = decode_stream(buf)
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
    return _make_env(KIND_ERROR, {"code": code, "message": message})


async def _send(writer: asyncio.StreamWriter, env: Envelope) -> None:
    writer.write(encode(env))
    try:
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass


__all__ = [
    "AllowAllAuthenticator",
    "Authenticator",
    "MAX_DELIVERY_ATTEMPTS",
    "SERVER_VERSION",
    "WireServer",
]
