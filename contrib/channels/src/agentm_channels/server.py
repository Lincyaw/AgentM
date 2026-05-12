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

from agentm_channels.auth import UnixPeerCredAuthenticator
from agentm_channels.transport import (
    ServerTransport,
    UnixServerTransport,
    WebSocketServerTransport,
)
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
    KIND_OUTBOUND,
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
# Safety-net poll: the delivery worker waits on an asyncio.Event woken by
# outbox.enqueue, but it also re-checks every LEASE_REFRESH_INTERVAL
# seconds so leases that expired without a fresh enqueue (crash recovery)
# eventually get re-leased. See §4.5.3 of the design.
LEASE_REFRESH_INTERVAL: float = 5.0
# Reason strings for delivery_batch envelopes (§4.5.3).
BATCH_REASON_RECONNECT_CATCHUP: str = "reconnect_catchup"

InboundHandler = Callable[[PeerSession, Envelope], Awaitable[None]]
# Optional hooks: ``hello`` fires after a successful handshake and
# before any inbound is dispatched; ``disconnect`` fires when the peer
# read loop exits (clean BYE, socket close, or crash). Both are
# optional — the server keeps working when they are ``None``. Used by
# :class:`WorkerRegistry` to track ``agent_worker`` peers.
PeerHelloHandler = Callable[[PeerSession], Awaitable[None]]
PeerDisconnectHandler = Callable[[PeerSession], Awaitable[None]]
# Worker → gateway outbound: emitted by an ``agent_worker`` peer with
# ``kind=outbound``; the server forwards it here so the gateway can
# route it back to the originating chat client.
WorkerOutboundHandler = Callable[[PeerSession, Envelope], Awaitable[None]]


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
        outbox_or_socket: OutboxStore | str | None = None,
        outbox: OutboxStore | None = None,
        inbox: InboxLog | None = None,
        on_inbound: InboundHandler | None = None,
        *,
        socket_path: str | None = None,
        transport: ServerTransport | None = None,
        authenticator: Authenticator | None = None,
        delivery_batch_max: int = 32,
        lease_ttl: float = 30.0,
        slow_consumer_high_water: int = 1000,
        max_delivery_attempts: int = MAX_DELIVERY_ATTEMPTS,
        on_peer_hello: PeerHelloHandler | None = None,
        on_peer_disconnect: PeerDisconnectHandler | None = None,
        on_worker_outbound: WorkerOutboundHandler | None = None,
    ) -> None:
        # Back-compat positional shim: callers that historically passed
        # ``WireServer(socket_path, outbox, inbox, on_inbound)`` still work.
        # New callers pass ``transport=`` (or ``socket_path=``) by keyword
        # along with ``outbox=``/``inbox=``/``on_inbound=``.
        if isinstance(outbox_or_socket, str):
            if socket_path is None:
                socket_path = outbox_or_socket
        elif outbox_or_socket is not None:
            outbox = outbox_or_socket
        if outbox is None or inbox is None or on_inbound is None:
            raise TypeError(
                "WireServer requires outbox, inbox, and on_inbound arguments"
            )
        if transport is None:
            if socket_path is None:
                raise TypeError(
                    "WireServer requires either transport= or socket_path="
                )
            transport = UnixServerTransport(socket_path)
        # Peer-cred auth depends on AF_UNIX kernel credentials; pairing
        # it with a WebSocket transport would silently degrade to
        # rejection-of-everything (no socket extra_info), which is
        # confusing. Fail fast at construction.
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
        self._on_peer_hello = on_peer_hello
        self._on_peer_disconnect = on_peer_disconnect
        self._on_worker_outbound = on_worker_outbound
        self._auth: Authenticator = authenticator or AllowAllAuthenticator()
        self._delivery_batch_max = delivery_batch_max
        self._lease_ttl = lease_ttl
        self._high_water = slow_consumer_high_water
        self._low_water = max(1, slow_consumer_high_water // 2)
        self._max_attempts = max_delivery_attempts
        # Retry-delay policy. Outbox stores may expose ``backoff_delay``
        # (the default :class:`SqliteOutbox` does, and accepts an
        # injected callable via its constructor); fall back to the
        # module-level ``exponential_backoff``. No dedicated knob on
        # the server — the outbox already owns this for ``set_notifier``
        # symmetry.
        outbox_backoff = getattr(outbox, "backoff_delay", None)
        self._backoff: Callable[[int], float] = (
            outbox_backoff if callable(outbox_backoff) else exponential_backoff
        )
        # Duck-typed: SqliteOutbox exposes next_retry_at_min for tighter
        # delivery-loop wakeups when pending rows are under backoff.
        # Alternate backends that don't implement it fall through to
        # the safety-net LEASE_REFRESH_INTERVAL only — correctness is
        # unaffected, just slightly higher latency on retry-only wakes.
        _next_retry_at_min = getattr(outbox, "next_retry_at_min", None)
        self._next_retry_at_min: Callable[[str], float | None] = (
            _next_retry_at_min if callable(_next_retry_at_min) else lambda _peer_id: None
        )
        self._registry = PeerRegistry()
        self._started = False
        self._stopped = False
        self._conn_tasks: set[asyncio.Task[None]] = set()
        self._delivery_tasks: dict[str, asyncio.Task[None]] = {}
        # Per-peer wakeup events: outbox.enqueue calls our notifier, which
        # sets the peer's Event so the delivery worker drains immediately
        # instead of polling. Steady-state latency for an enqueue is now
        # bounded by writer.drain(), not the old 50 ms poll.
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
            if self._on_peer_hello is not None:
                try:
                    await self._on_peer_hello(session)
                except Exception:
                    log.exception("on_peer_hello raised for peer=%s", peer_id)

            wake = asyncio.Event()
            self._wake_events[peer_id] = wake
            self._register_outbox_notifier(peer_id, wake)
            # An enqueue may have landed before we registered: prime
            # the event so the first drain pass runs.
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
                session = self._registry.get(peer_id)
                self._registry.deregister(peer_id)
                if session is not None and self._on_peer_disconnect is not None:
                    try:
                        await self._on_peer_disconnect(session)
                    except Exception:
                        log.exception(
                            "on_peer_disconnect raised for peer=%s", peer_id
                        )
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
        if env.kind == KIND_OUTBOUND:
            # Workers send ``outbound`` envelopes for the gateway to
            # relay back to the originating chat client. Non-worker
            # peers (chat_client) have no reason to emit outbound — we
            # ignore those defensively, the gateway never asked them to
            # forward replies.
            if (
                self._on_worker_outbound is not None
                and session.peer_kind == "agent_worker"
            ):
                try:
                    await self._on_worker_outbound(session, env)
                except Exception:
                    log.exception(
                        "on_worker_outbound raised for env id=%s peer=%s",
                        env.id,
                        session.peer_id,
                    )
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

    async def _delivery_loop(
        self, session: PeerSession, wake: asyncio.Event
    ) -> None:
        """Drain outbox rows for ``session.peer_id`` until cancelled.

        Wakeup model: blocks on ``wake`` (set by ``outbox.enqueue`` via
        the registered notifier) with a ``LEASE_REFRESH_INTERVAL``
        timeout. The timeout is the safety net for crash recovery —
        leases that expired without an enqueue event still get retried.

        Batching policy: while in the *catch-up window* — every drain
        from (re)connection until the first empty drain — multi-record
        leases ship as one ``delivery_batch`` envelope per §4.5.3.
        After that we are in steady state: subsequent drains push
        single ``outbound`` envelopes one-by-one. The design contract
        is that batches mean "catch up after gap", not "spool whatever
        happens to be queued right now".

        Slow-consumer accounting is observational: we *log a diagnostic*
        when the pending queue crosses ``slow_consumer_high_water`` and
        clear the flag when it falls below ``high_water / 2``. We do
        **not** stop draining — back-pressure belongs upstream of the
        outbox (the enqueue side); the delivery worker's job is to
        empty the queue as fast as the peer will accept. Per §0
        simplicity, no ``slow_consumer`` wire event.
        """
        peer_id = session.peer_id
        # ``in_catchup`` starts True and stays True until we see an
        # empty drain — i.e. we've fully consumed whatever was queued
        # at the moment the peer (re)connected. After the first empty
        # drain we are in steady state and single enqueues should
        # arrive as ``outbound``, not wrapped in a single-item batch.
        in_catchup = True
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
                    # Empty drain → catch-up window closes; any later
                    # enqueues will ship as single envelopes. Sleep on
                    # the wake event; the LEASE_REFRESH_INTERVAL timeout
                    # backs up crash recovery (expired leases without a
                    # fresh enqueue event).
                    #
                    # If there are pending rows whose ``next_retry_at``
                    # is still in the future (post-nack backoff), cap
                    # the wait at the earliest retry instant so we
                    # don't sit blind on the safety-net interval. The
                    # MIN query is cheap; this turns the empty-drain
                    # branch from "wait 5 s blind" into "wake exactly
                    # when the next row becomes leasable".
                    in_catchup = False
                    next_retry = await asyncio.to_thread(
                        self._next_retry_at_min, peer_id
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
                await self._deliver(session, records, in_catchup=in_catchup)
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
        self,
        session: PeerSession,
        records: list[OutboxRecord],
        *,
        in_catchup: bool,
    ) -> None:
        """Write one batch (or single envelope) to the peer.

        Frame shape:

        * ``in_catchup and len(records) > 1`` → one ``delivery_batch``
          envelope carrying ``items`` plus ``reason`` and (when all
          records share one) ``session_key`` (§4.5.3).
        * Otherwise → individual ``outbound`` envelopes, one wire
          write per record. Includes the case where the catch-up
          drain finds exactly one ready row — no reason to wrap one
          item.

        On socket write/drain failure: account the records (nack or
        dead-letter per attempts) and *re-raise* a sentinel so the
        delivery loop exits — the connection is dead, sit out until
        the peer reconnects. Without this the loop would re-lease the
        same records every tick and exhaust ``max_attempts`` against
        a closed socket.
        """
        writer = session.transport_writer
        try:
            if in_catchup and len(records) > 1:
                items = [r.envelope.to_dict() for r in records]
                body: dict[str, Any] = {
                    "items": items,
                    "reason": BATCH_REASON_RECONNECT_CATCHUP,
                    "session_key": _common_session_key(records),
                }
                batch = _make_env(KIND_DELIVERY_BATCH, body)
                writer.write(encode(batch))
                await writer.drain()
            else:
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
        # Success: ack the whole set.
        await asyncio.to_thread(self._outbox.ack, [r.id for r in records])

    async def _nack_records(self, records: list[OutboxRecord]) -> None:
        now = time.time()
        for r in records:
            delay = self._backoff(r.attempts)
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
                delay = self._backoff(r.attempts)
                await asyncio.to_thread(self._outbox.nack, [r.id], now + delay)


    # -- outbox notifier wiring --------------------------------------

    def _register_outbox_notifier(self, peer_id: str, event: asyncio.Event) -> None:
        """Register a per-peer wakeup with the outbox if it supports it.

        Outbox implementations may expose ``set_notifier(peer_id, fn)``
        as an optional protocol method. The default :class:`SqliteOutbox`
        does; alternative stores can omit it and fall back to the
        timeout-based safety-net poll. Missing method is *not* an
        error — the loop still makes progress, just at
        ``LEASE_REFRESH_INTERVAL`` granularity.
        """
        loop = self._loop
        if loop is None:
            return
        set_notifier = getattr(self._outbox, "set_notifier", None)
        if set_notifier is None:
            return

        def notify(_pid: str) -> None:
            # outbox.enqueue is called from worker threads (via
            # asyncio.to_thread). Bounce the set() through the loop.
            if not loop.is_closed():
                loop.call_soon_threadsafe(event.set)

        set_notifier(peer_id, notify)

    def _unregister_outbox_notifier(self, peer_id: str) -> None:
        set_notifier = getattr(self._outbox, "set_notifier", None)
        if set_notifier is None:
            return
        set_notifier(peer_id, None)


# -- helpers ---------------------------------------------------------

def _common_session_key(records: list[OutboxRecord]) -> str | None:
    """Return the session key shared by all records, or ``None``.

    Used to annotate ``delivery_batch`` envelopes (§4.5.3) so clients
    that fan out per-session can route the batch in one hop. We read
    :attr:`Envelope.root_session_key` — the only session-scoped field
    on the wire today.
    """
    first = records[0].envelope.root_session_key
    if first is None:
        return None
    for r in records[1:]:
        if r.envelope.root_session_key != first:
            return None
    return first


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
    "PeerDisconnectHandler",
    "PeerHelloHandler",
    "SERVER_VERSION",
    "WireServer",
    "WorkerOutboundHandler",
]
