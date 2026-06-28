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
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from agentm.gateway.auth import (
    AllowAllAuthenticator,
    Authenticator,
    UnixPeerCredAuthenticator,
)
from agentm.gateway.outbox import (
    InboxLog,
    OutboxStore,
)
from agentm.gateway.peer import PeerRegistry, PeerSession
from agentm.gateway.send_queue import SendItem, SendQueue
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

SERVER_VERSION: str = "0.2.0"
# Dead-letter cap for durable replay. ``attempts`` advances once per
# reconnect-prefill lease (not per live retry — there is no live retry
# loop), so this bounds reconnect-replays of a row, i.e. "5 reconnects",
# not "5 rapid retries".
MAX_DELIVERY_ATTEMPTS: int = 5

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
        capabilities_provider: Callable[[], dict[str, Any]] | None = None,
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
        self._capabilities_provider = capabilities_provider
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
        # peer_id -> the per-peer sender task draining its SendQueue.
        self._delivery_tasks: dict[str, asyncio.Task[None]] = {}

    # -- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        await self._transport.serve(self._handle_connection)
        self._started = True

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        # Cancel per-peer connection + delivery tasks FIRST. Each read loop's
        # finally then closes its writer, so the listening socket is left with
        # no live connections. Otherwise transport.close()'s wait_closed()
        # (py3.12 waits for active connections to finish) blocks forever on a
        # peer whose read loop never returns on its own — e.g. an
        # auto-reconnecting client — hanging `systemctl stop` until
        # TimeoutStopSec SIGKILLs the process.
        for task in list(self._delivery_tasks.values()):
            task.cancel()
        for task in list(self._conn_tasks):
            task.cancel()
        all_tasks = list(self._delivery_tasks.values()) + list(self._conn_tasks)
        for task in all_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                # Draining cancelled delivery/conn tasks during shutdown.
                logger.debug("gateway: task raised while draining on shutdown: {}", exc)
        self._delivery_tasks.clear()
        self._conn_tasks.clear()
        await self._transport.close()

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
            peer_name = str(body.get("peer_name") or body.get("peer_id") or "")
            peer_id = peer_name
            auth_raw = body.get("auth")
            token_raw = body.get("token") or (
                auth_raw if isinstance(auth_raw, str) else None
            )
            token = str(token_raw) if isinstance(token_raw, str) else None
            if not peer_name:
                await _send(
                    writer,
                    _error_env("bad_hello", "hello body missing peer_name"),
                )
                return
            ok = await self._auth.authenticate(peer_name, token, writer)
            if not ok:
                await _send(
                    writer,
                    _error_env("auth_failed", f"auth failed for peer {peer_name!r}"),
                )
                return
            # Peer names identify the client implementation (e.g. terminal-go),
            # not a singleton process. Multiple terminal peers may connect to the
            # same gateway concurrently, so derive an effective registry/outbox id
            # when the requested name is already live instead of rejecting it.
            if peer_id in self._registry:
                peer_id = f"{peer_name}#{uuid.uuid4().hex[:8]}"
                logger.info(
                    "gateway: peer name {} already connected; assigned peer_id {}",
                    peer_name,
                    peer_id,
                )

            peer_cwd_raw = body.get("cwd")
            peer_cwd = str(peer_cwd_raw) if isinstance(peer_cwd_raw, str) and peer_cwd_raw.strip() else None
            session = PeerSession(
                peer_id=peer_id,
                transport_writer=writer,
                cwd=peer_cwd,
                last_seen=time.time(),
                capabilities=dict(body.get("capabilities") or {}),
            )
            self._registry.register(session)
            welcome_body: dict[str, Any] = {
                "server_version": SERVER_VERSION,
                "wire_version": WIRE_VERSION,
                "peer_id": peer_id,
                "session_resume": [],
            }
            if self._capabilities_provider is not None:
                try:
                    welcome_body["capabilities"] = self._capabilities_provider()
                except Exception:
                    logger.exception(
                        "gateway: capabilities_provider raised; welcome sent without it"
                    )
            await _send(writer, _make_env(KIND_WELCOME, welcome_body))

            # Size the unified send queue to the slow-consumer high-water,
            # then prefill any durable rows left unacked by a prior
            # connection (reconnect replay) in FIFO order *before* the sender
            # starts — replayed durable frames thus precede new live frames.
            session.send_q = SendQueue(high_water=self._high_water)
            await self._prefill_from_outbox(session)
            sender = asyncio.create_task(
                self._sender_loop(session), name=f"send:{peer_id}"
            )
            self._delivery_tasks[peer_id] = sender
            try:
                await self._read_loop(session, reader)
            finally:
                sender.cancel()
                try:
                    await sender
                except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                    logger.debug("gateway: sender task drain raised for {}: {}", peer_id, exc)
                self._delivery_tasks.pop(peer_id, None)
        except (ConnectionResetError, BrokenPipeError) as exc:
            # Peer dropped the connection — normal client disconnect.
            logger.debug("gateway: peer {} connection reset: {}", peer_id, exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("connection handler crashed")
        finally:
            if peer_id is not None:
                self._registry.deregister(peer_id)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as exc:  # noqa: BLE001
                logger.debug("gateway: writer close failed for {}: {}", peer_id, exc)
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
                logger.exception(f"on_inbound handler raised for env id={env.id}")
            return
        if env.kind == KIND_ACK:
            return  # server-side acks are implicit; tolerate client-sent ones.
        if env.kind == KIND_PING:
            await _send(session.transport_writer, _make_env(KIND_PONG, {}))
            return
        if env.kind == KIND_PONG:
            return
        logger.debug(f"server: ignoring unexpected kind {env.kind} from {session.peer_id}")

    # -- unified sender (§2.6 ordered delivery channel) --------------

    async def _prefill_from_outbox(self, session: PeerSession) -> None:
        """Replay durable rows left unacked from a prior connection.

        On reconnect, every durable frame that was never acked is still in
        the outbox. We lease them all (FIFO by row id) and push them onto
        the fresh send queue *before* the sender starts, so a reconnecting
        client receives its missed durable frames in order, ahead of any
        new live frames. Rows whose attempts exceed the cap are dead-
        lettered instead of replayed forever (poison-pill guard).
        """
        peer_id = session.peer_id
        now = time.time()
        while True:
            records = await asyncio.to_thread(
                self._outbox.lease, peer_id, self._delivery_batch_max, now
            )
            if not records:
                return
            for r in records:
                if r.attempts > self._max_attempts:
                    await asyncio.to_thread(
                        self._outbox.dead_letter, r.id, "max delivery attempts"
                    )
                    logger.warning(f"dead_letter peer={peer_id} env_id={r.envelope.id} reason=max_attempts")
                    continue
                session.send_q.put(r.envelope, r.id)

    async def _sender_loop(self, session: PeerSession) -> None:
        """Drain the per-peer send queue in order until cancelled.

        This is the single writer for the peer: durable and ephemeral
        frames share this one ordered path, so delivery order == enqueue
        order (the ordering guarantee — see ``send_queue``). A durable item
        (``outbox_id is not None``) is acked only after a successful write;
        on socket failure it is left in the outbox for reconnect replay and
        the loop exits via ``_ConnectionLost``. An ephemeral item is dropped
        on failure (best-effort).

        Slow-consumer accounting is observational: flag when the queue
        depth crosses ``high_water`` and clear below ``high_water / 2``.
        """
        peer_id = session.peer_id
        writer = session.transport_writer
        try:
            while True:
                item = await session.send_q.get()
                depth = len(session.send_q)
                session.pending_count_hint = depth
                if depth > self._high_water and not session.backpressure:
                    logger.warning(f"slow_consumer peer={peer_id} queued={depth} high_water={self._high_water} dropped_ephemeral={session.send_q.dropped_ephemeral}")
                    session.backpressure = True
                elif session.backpressure and depth <= self._low_water:
                    session.backpressure = False
                try:
                    # One wire frame == one WS message; the write lock keeps
                    # each frame mapped to exactly one flush.
                    async with session.write_lock:
                        writer.write(encode(item.envelope))
                        await writer.drain()
                except (ConnectionResetError, BrokenPipeError) as exc:
                    await self._account_failure(session, item, repr(exc))
                    raise _ConnectionLost() from exc
                except Exception as exc:
                    await self._account_failure(session, item, repr(exc))
                    raise _ConnectionLost() from exc
                if item.outbox_id is not None:
                    await asyncio.to_thread(self._outbox.ack, [item.outbox_id])
        except _ConnectionLost:
            try:
                session.transport_writer.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("gateway: transport writer close failed on lost conn: {}", exc)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"sender loop for peer={peer_id} crashed")

    async def _account_failure(
        self, session: PeerSession, item: SendItem, reason: str
    ) -> None:
        """On write failure, release a durable item for reconnect replay.

        Ephemeral items (``outbox_id is None``) are best-effort and simply
        dropped. A durable item is nacked with an immediate retry time so
        the next connection's prefill re-leases it promptly (the sender has
        already exited, so there is no tight-loop to throttle); the lease
        TTL is the crash-recovery backstop for items still queued behind it.
        """
        if item.outbox_id is None:
            return
        await asyncio.to_thread(self._outbox.nack, [item.outbox_id], time.time())


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
    except (ConnectionResetError, BrokenPipeError) as exc:
        # Peer gone mid-send — normal disconnect; the read loop will clean up.
        logger.debug("gateway: drain failed, peer disconnected: {}", exc)


__all__ = [
    "MAX_DELIVERY_ATTEMPTS",
    "SERVER_VERSION",
    "WireServer",
]
