"""In-tree asyncio wire client (v2).

The shared client library chat-client peers (terminal, feishu) and tests
use to talk to a :class:`WireServer`. No reconnect logic — callers handle
reconnect; the server-side outbox makes that safe.

The ``on_outbound`` callback fires once per delivered ``outbound``
envelope. v2 has no ``delivery_batch`` / ``bye`` frames — pipelined
single-envelope push is enough at this scale (§2.3).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

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

log = logging.getLogger("agentm.gateway.client")

OutboundHandler = Callable[[Envelope], Awaitable[None]]


class AuthError(Exception):
    """Server rejected our hello (auth_failed / duplicate_peer / etc.)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


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
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._errors: asyncio.Queue[Envelope] = asyncio.Queue()
        self._closed = False
        self._welcome: Envelope | None = None

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
        self._read_task = asyncio.create_task(
            self._read_loop(), name=f"client-read:{self._peer_name}"
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None and not self._writer.is_closing():
            with contextlib.suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._read_task

    # -- IO -----------------------------------------------------------

    async def send(self, env: Envelope) -> None:
        if self._writer is None:
            raise RuntimeError("client not connected")
        self._writer.write(encode(env))
        await self._writer.drain()

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
                    log.exception("client received malformed frame")
                    return
                for env in envelopes:
                    await self._dispatch(env)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("client read loop crashed for peer=%s", self._peer_name)

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


__all__ = ["AuthError", "WireClient"]
