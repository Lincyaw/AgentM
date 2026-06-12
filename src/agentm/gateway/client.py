"""In-tree asyncio wire client (v2).

The shared client library chat-client peers (terminal, feishu) and tests
use to talk to a :class:`WireServer`.

Reconnect
---------
:meth:`WireClient.connect` connects once and the read loop exits when the
connection drops; a caller may then re-dial. The server-side outbox makes
that safe: a re-dial with the *same* ``peer_name`` triggers a FIFO replay
of every durable frame the prior connection left unacked, ahead of new
live frames.

For long-lived daemon peers this client also offers an **opt-in**
supervisor, :meth:`WireClient.run_reconnecting`, which owns that re-dial
loop: it awaits the read loop, and on an unexpected drop re-dials the
transport, re-sends the same hello (same ``peer_name`` → outbox replay),
and restarts the read loop, with exponential backoff + jitter. A
deliberate :meth:`close` stops the supervisor; an :class:`AuthError`
(including a transient ``duplicate_peer`` that does not clear within the
backoff window) is the only non-retryable surface. Existing callers that
only use :meth:`connect` see no behaviour change — the supervisor is
additive and the disconnect signalling it relies on is inert unless used.

The ``on_outbound`` callback fires once per delivered ``outbound``
envelope. v2 has no ``delivery_batch`` / ``bye`` frames — pipelined
single-envelope push is enough at this scale (§2.3).
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from agentm.gateway.transport import ClientTransport, UnixClientTransport
from agentm.gateway.wire import (
    KIND_ERROR,
    KIND_HELLO,
    KIND_INBOUND,
    KIND_OUTBOUND,
    KIND_PING,
    KIND_PONG,
    KIND_WELCOME,
    WIRE_VERSION,
    Envelope,
    InvalidEnvelope,
    WireError,
    decode_stream,
    encode,
)

OutboundHandler = Callable[[Envelope], Awaitable[None]]


class AuthError(Exception):
    """Server rejected our hello (auth_failed / duplicate_peer / etc.)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Disconnected(RuntimeError):
    """A ``send`` was attempted while the client was not connected.

    Retryable: the reconnect supervisor (or a caller's own loop) will
    re-establish the link and the message can be re-sent. Distinct from a
    never-connected :class:`RuntimeError`, which is a programming error.
    """


class WireClient:
    """Asyncio wire client to a :class:`WireServer` (v2)."""

    def __init__(
        self,
        *,
        peer_name: str,
        peer_version: str = "0.2.0",
        transport: ClientTransport | None = None,
        socket_path: str | None = None,
        token: str | None = None,
        on_outbound: OutboundHandler | None = None,
        capabilities: dict[str, Any] | None = None,
        backoff_base: float = 0.5,
        backoff_cap: float = 5.0,
    ) -> None:
        # ``socket_path=`` is a Unix-socket convenience shortcut equivalent to
        # ``transport=UnixClientTransport(socket_path)``; pass exactly one.
        if transport is None:
            if socket_path is None:
                raise TypeError(
                    "WireClient requires either transport= or socket_path="
                )
            transport = UnixClientTransport(socket_path)
        elif socket_path is not None:
            raise TypeError(
                "WireClient: pass either transport= or socket_path=, not both"
            )
        self._transport = transport
        self._peer_name = peer_name
        self._peer_version = peer_version
        self._token = token
        self._on_outbound = on_outbound
        self._capabilities: dict[str, Any] = dict(capabilities or {})
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._errors: asyncio.Queue[Envelope] = asyncio.Queue()
        self._closed = False
        self._welcome: Envelope | None = None
        # Set whenever the read loop ends on a *transport* EOF (not a
        # deliberate ``close()`` cancel). The reconnect supervisor waits on
        # this; for plain ``connect()`` callers it is simply never observed.
        self._disconnected: asyncio.Event = asyncio.Event()
        # True between a successful connect and the next disconnect/close.
        # ``send`` consults it to fail fast with a retryable error during a
        # reconnect gap instead of writing to a dead/closing writer.
        self._connected = False

    # -- lifecycle ----------------------------------------------------

    async def connect(self) -> None:
        self._reader, self._writer = await self._transport.connect()
        body: dict[str, Any] = {
            "peer_name": self._peer_name,
            "peer_version": self._peer_version,
            "capabilities": dict(self._capabilities),
        }
        if self._token is not None:
            body["auth"] = self._token
        hello = Envelope(
            v=WIRE_VERSION,
            id=f"hello-{self._peer_name}-{int(time.time() * 1000)}",
            kind=KIND_HELLO,
            ts=time.time(),
            body=body,
        )
        self._writer.write(encode(hello))
        await self._writer.drain()
        # Await first server frame: welcome or error.
        first = await _read_one_frame(self._reader)
        if first is None:
            raise ConnectionRefusedError("server closed before welcome")
        if first.kind == KIND_ERROR:
            err_body = first.body if isinstance(first.body, dict) else {}
            code = str(err_body.get("code", "unknown"))
            message = str(err_body.get("message", ""))
            with contextlib.suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
            raise AuthError(code, message)
        if first.kind != KIND_WELCOME:
            raise ConnectionRefusedError(f"expected welcome, got {first.kind!r}")
        self._welcome = first
        self._disconnected.clear()
        self._connected = True
        self._read_task = asyncio.create_task(
            self._read_loop(), name=f"client-read:{self._peer_name}"
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connected = False
        # Wake any reconnect supervisor waiting on a drop so it can observe
        # ``_closed`` and exit instead of re-dialling.
        self._disconnected.set()
        if self._writer is not None and not self._writer.is_closing():
            with contextlib.suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._read_task

    # -- reconnect supervisor ----------------------------------------

    async def run_reconnecting(
        self,
        *,
        connect_first: bool = True,
        on_connect: Callable[[bool], Awaitable[None]] | None = None,
    ) -> None:
        """Connect and keep the link alive across unexpected drops.

        Opt-in supervisor for long-lived daemon peers (feishu). When
        ``connect_first`` is true it performs the initial :meth:`connect`
        (surfacing its exceptions so the caller keeps the usual first-connect
        semantics — auth → exit, connect-failed → exit); set it false when the
        caller has already connected and wants reconnect-only supervision
        (preserving its own first-connect error handling). Then it loops: wait
        for the read loop to end on an unexpected drop, and re-dial with the
        *same* ``peer_name`` so the server replays missed durable frames from
        the outbox. Returns only when :meth:`close` is called.

        Backoff is exponential from ``backoff_base`` to ``backoff_cap`` with
        full jitter, reset to base after each successful reconnect. An
        :class:`AuthError` on *reconnect* is non-retryable and re-raised —
        except a transient ``duplicate_peer`` (the prior session is still
        deregistering), which is retried within the backoff schedule only
        until the backoff reaches ``backoff_cap`` — i.e. a bounded window of
        roughly ``log2(backoff_cap/backoff_base)`` attempts (~4 over ~7.5s at
        the 0.5s/5s defaults), after which it is surfaced as fatal. If a
        gateway's deregistration can lag longer than that under load, raise
        ``backoff_cap``.

        ``on_connect(is_initial)`` fires after every successful (re)connect —
        the peer can use it to resume any per-connection work. It is awaited;
        an exception from it propagates.

        NOTE: ``peer_name`` is fixed per process today, so replay works for
        transient drops and full gateway restarts within this process's
        lifetime. Durable identity across a *peer-process* restart (stable,
        configurable ``peer_name``) is a separate future improvement.
        """
        # Initial connect: let exceptions propagate unchanged.
        if connect_first:
            await self.connect()
            if on_connect is not None:
                await on_connect(True)

        backoff = self._backoff_base
        while not self._closed:
            await self._disconnected.wait()
            if self._closed:
                return
            # Tear down the dead read task before re-dialling.
            if self._read_task is not None:
                self._read_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._read_task
                self._read_task = None
            with contextlib.suppress(Exception):
                if self._writer is not None and not self._writer.is_closing():
                    self._writer.close()

            delay = min(backoff, self._backoff_cap)
            delay += random.uniform(0.0, delay)  # full jitter
            logger.warning(
                "wire client peer=%s disconnected; reconnecting in %.2fs",
                self._peer_name,
                delay,
            )
            await asyncio.sleep(delay)
            if self._closed:
                return
            try:
                await self.connect()
            except AuthError as exc:
                if exc.code == "duplicate_peer" and backoff < self._backoff_cap:
                    # Prior session still deregistering — back off and retry.
                    backoff = min(backoff * 2, self._backoff_cap)
                    self._disconnected.set()
                    continue
                logger.error(
                    "wire client peer=%s reconnect rejected (code=%s); giving up",
                    self._peer_name,
                    exc.code,
                )
                raise
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                logger.warning(
                    "wire client peer=%s reconnect failed (%s); retrying",
                    self._peer_name,
                    exc.__class__.__name__,
                )
                backoff = min(backoff * 2, self._backoff_cap)
                self._disconnected.set()
                continue
            # Success — reset backoff and notify.
            backoff = self._backoff_base
            logger.info("wire client peer=%s reconnected", self._peer_name)
            if on_connect is not None:
                await on_connect(False)

    # -- IO -----------------------------------------------------------

    async def send(self, env: Envelope) -> None:
        if self._writer is None:
            raise RuntimeError("client not connected")
        if not self._connected:
            # Reconnect gap: the writer is dead/closing. Fail fast with a
            # retryable error rather than crash the process — the caller (or
            # the reconnect supervisor) re-dials and re-sends. We keep this
            # simple: no in-client buffering of inbound frames across a gap.
            raise Disconnected("client disconnected; retry after reconnect")
        try:
            self._writer.write(encode(env))
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            self._connected = False
            self._disconnected.set()
            raise Disconnected("connection dropped mid-send; retry") from exc

    async def send_inbound(
        self,
        body: dict[str, Any],
        *,
        session_key: str,
        scenario: str | None = None,
        env_id: str | None = None,
    ) -> None:
        """Convenience: build + send an ``inbound`` envelope (§2.4).

        ``session_key`` is required — it is how the gateway routes the
        message to (or creates) the right session. ``scenario`` is set on
        the first inbound for a chat the gateway has not seen.
        """
        env = Envelope(
            v=WIRE_VERSION,
            id=env_id or f"in-{self._peer_name}-{int(time.time() * 1_000_000)}",
            kind=KIND_INBOUND,
            ts=time.time(),
            session_key=session_key,
            scenario=scenario,
            body=body,
        )
        await self.send(env)

    async def ping(self) -> None:
        env = Envelope(
            v=WIRE_VERSION,
            id=f"ping-{self._peer_name}-{int(time.time() * 1000)}",
            kind=KIND_PING,
            ts=time.time(),
            body={},
        )
        await self.send(env)

    async def errors(self) -> Envelope:
        """Block until the server emits an ``error`` envelope, then return it."""
        return await self._errors.get()

    def welcome(self) -> Envelope | None:
        return self._welcome

    # -- read loop ----------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._reader is not None
        buf = b""
        try:
            while True:
                chunk = await self._reader.read(65536)
                if not chunk:
                    return
                buf += chunk
                try:
                    envelopes, buf = decode_stream(buf)
                except (InvalidEnvelope, WireError):
                    logger.exception("client received malformed frame")
                    return
                for env in envelopes:
                    await self._dispatch(env)
        except asyncio.CancelledError:
            # Deliberate teardown (close() cancels this task). Do NOT signal
            # an unexpected disconnect — the supervisor must not re-dial.
            raise
        except Exception:
            logger.exception("client read loop crashed for peer=%s", self._peer_name)
        finally:
            # Any non-cancel exit is an unexpected drop. Mark disconnected so
            # the reconnect supervisor (if any) wakes and re-dials; harmless
            # for plain connect() callers, who never wait on the event.
            self._connected = False
            self._disconnected.set()

    async def _dispatch(self, env: Envelope) -> None:
        if env.kind == KIND_OUTBOUND:
            if self._on_outbound is not None:
                await self._on_outbound(env)
            return
        if env.kind == KIND_PONG:
            return
        if env.kind == KIND_ERROR:
            await self._errors.put(env)
            if self._on_outbound is not None:
                await self._on_outbound(env)
            return
        if env.kind == KIND_PING:
            if self._writer is not None:
                pong = Envelope(
                    v=WIRE_VERSION,
                    id=f"pong-{self._peer_name}-{int(time.time() * 1000)}",
                    kind=KIND_PONG,
                    ts=time.time(),
                    body={},
                )
                self._writer.write(encode(pong))
                with contextlib.suppress(Exception):
                    await self._writer.drain()
            return


async def _read_one_frame(reader: asyncio.StreamReader) -> Envelope | None:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    length = int.from_bytes(header, "big")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None
    try:
        envelopes, _rest = decode_stream(header + body)
    except (InvalidEnvelope, WireError):
        return None
    return envelopes[0] if envelopes else None


__all__ = ["AuthError", "Disconnected", "WireClient"]
