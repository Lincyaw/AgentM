"""In-tree asyncio wire client.

Used by Phase 1 integration tests and (in Phase 2+) by extracted
channel processes. No reconnect logic — callers handle reconnect; the
server-side outbox makes that safe (see §4.5 of the design doc).

The on_outbound callback fires once per delivered envelope. For
``delivery_batch`` frames (catch-up), the client iterates ``items``
in order and invokes the callback for each item synchronously
inside the read loop, so the application sees ordering as published.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agentm_channels.wire import (
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
    InvalidEnvelope,
    WireError,
    decode_stream,
    encode,
)

log = logging.getLogger("agentm_channels.client")

OutboundHandler = Callable[[Envelope], Awaitable[None]]


class AuthError(Exception):
    """Server rejected our hello (auth_failed / duplicate_peer / etc.)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class WireClient:
    """Asyncio Unix-socket client to a :class:`WireServer`."""

    def __init__(
        self,
        socket_path: str,
        peer_id: str,
        peer_kind: str,
        *,
        token: str | None = None,
        on_outbound: OutboundHandler | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._peer_id = peer_id
        self._peer_kind = peer_kind
        self._token = token
        self._on_outbound = on_outbound
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._errors: asyncio.Queue[Envelope] = asyncio.Queue()
        self._closed = False
        self._welcome: Envelope | None = None

    # -- lifecycle ----------------------------------------------------

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_unix_connection(
            path=self._socket_path
        )
        hello = Envelope(
            v=WIRE_VERSION,
            id=f"hello-{self._peer_id}-{int(time.time() * 1000)}",
            kind=KIND_HELLO,
            ts=time.time(),
            body={
                "peer_id": self._peer_id,
                "peer_kind": self._peer_kind,
                "token": self._token,
                "capabilities": {},
            },
        )
        self._writer.write(encode(hello))
        await self._writer.drain()
        # Await first server frame: welcome or error.
        first = await _read_one_frame(self._reader)
        if first is None:
            raise ConnectionRefusedError("server closed before welcome")
        if first.kind == KIND_ERROR:
            body = first.body if isinstance(first.body, dict) else {}
            code = str(body.get("code", "unknown"))
            message = str(body.get("message", ""))
            with contextlib.suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
            raise AuthError(code, message)
        if first.kind != KIND_WELCOME:
            raise ConnectionRefusedError(f"expected welcome, got {first.kind!r}")
        self._welcome = first
        self._read_task = asyncio.create_task(self._read_loop(), name=f"client-read:{self._peer_id}")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None and not self._writer.is_closing():
            bye = Envelope(
                v=WIRE_VERSION,
                id=f"bye-{self._peer_id}-{int(time.time() * 1000)}",
                kind=KIND_BYE,
                ts=time.time(),
                body={},
            )
            with contextlib.suppress(Exception):
                self._writer.write(encode(bye))
                await self._writer.drain()
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

    async def send_inbound(self, body: dict[str, Any], env_id: str | None = None) -> None:
        """Convenience: build + send an ``inbound`` envelope."""
        env = Envelope(
            v=WIRE_VERSION,
            id=env_id or f"in-{self._peer_id}-{int(time.time() * 1000_000)}",
            kind=KIND_INBOUND,
            ts=time.time(),
            body=body,
        )
        await self.send(env)

    async def ping(self) -> None:
        env = Envelope(
            v=WIRE_VERSION,
            id=f"ping-{self._peer_id}-{int(time.time() * 1000)}",
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
            log.exception("client read loop crashed for peer=%s", self._peer_id)

    async def _dispatch(self, env: Envelope) -> None:
        if env.kind == KIND_OUTBOUND:
            if self._on_outbound is not None:
                await self._on_outbound(env)
            return
        if env.kind == KIND_DELIVERY_BATCH:
            items = (env.body or {}).get("items", []) if isinstance(env.body, dict) else []
            if self._on_outbound is not None:
                for item in items:
                    try:
                        sub = Envelope.from_dict(item)
                    except (InvalidEnvelope, WireError):
                        log.exception("client: bad item in delivery_batch")
                        continue
                    await self._on_outbound(sub)
            return
        if env.kind == KIND_PONG:
            return
        if env.kind == KIND_ERROR:
            await self._errors.put(env)
            return
        # ping from server (future use): reply pong
        if env.kind == KIND_PING:
            if self._writer is not None:
                pong = Envelope(
                    v=WIRE_VERSION,
                    id=f"pong-{self._peer_id}-{int(time.time() * 1000)}",
                    kind=KIND_PONG,
                    ts=time.time(),
                    body={"echo_id": env.id},
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
