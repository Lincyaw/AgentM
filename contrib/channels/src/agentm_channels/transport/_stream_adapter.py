"""asyncio StreamReader/StreamWriter shims over a ``websockets`` connection.

The wire layer expects an :class:`asyncio.StreamReader` /
:class:`asyncio.StreamWriter` byte stream — see ``wire/framing.py`` and
``server._read_one_frame``. WebSocket connections speak in discrete
*messages*, not a byte stream, so we bridge the two while preserving
the "one envelope per WebSocket message" invariant (Phase 2 design,
``.claude/plans/2026-05-12-gateway-websocket-transport.md``).

The reader pulls binary messages on demand and exposes them as a
contiguous byte buffer to ``readexactly``. The writer accumulates the
framing bytes a caller writes and flushes them as a single
``ws.send(bytes)`` call when ``drain()`` is invoked — one wire frame
(length-prefix header + JSON body) maps to one WS binary message.

We target the modern :mod:`websockets.asyncio` connection class
(``websockets>=13``) — that's what :func:`websockets.serve` and
:func:`websockets.connect` return.
"""

from __future__ import annotations

import asyncio
from typing import Any

from websockets.asyncio.connection import Connection
from websockets.exceptions import ConnectionClosed


class WebSocketStreamReader:
    """asyncio-StreamReader-shaped reader over a WebSocket connection.

    Only the subset that the framing reader needs is implemented —
    today that is :meth:`readexactly` (see ``server._read_one_frame``).
    Closure surfaces as :class:`asyncio.IncompleteReadError`, matching
    ``StreamReader.readexactly`` semantics.
    """

    def __init__(self, ws: Connection) -> None:
        self._ws = ws
        self._buf = bytearray()
        self._eof = False

    async def read(self, n: int = -1) -> bytes:
        """Up-to-``n`` bytes; returns ``b""`` on close (StreamReader contract).

        Used by ``server._read_loop`` / ``client._read_loop`` for the
        bulk streaming path. If the buffer is empty we await one
        binary message; otherwise we return whatever's buffered (up to
        ``n``) immediately to keep latency low.
        """
        if not self._buf:
            if self._eof:
                return b""
            try:
                msg = await self._ws.recv()
            except ConnectionClosed:
                self._eof = True
                return b""
            if isinstance(msg, str):
                self._eof = True
                return b""
            self._buf.extend(msg)
        if n < 0 or n >= len(self._buf):
            out = bytes(self._buf)
            self._buf.clear()
            return out
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    async def readexactly(self, n: int) -> bytes:
        while len(self._buf) < n:
            if self._eof:
                raise asyncio.IncompleteReadError(bytes(self._buf), n)
            try:
                msg = await self._ws.recv()
            except ConnectionClosed:
                self._eof = True
                raise asyncio.IncompleteReadError(bytes(self._buf), n) from None
            if isinstance(msg, str):
                # Wire is binary-only by contract; a text frame is a
                # protocol violation → behave like EOF.
                self._eof = True
                raise asyncio.IncompleteReadError(bytes(self._buf), n)
            self._buf.extend(msg)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


class WebSocketStreamWriter:
    """asyncio-StreamWriter-shaped writer over a WebSocket connection.

    ``write()`` is synchronous and just buffers bytes. ``drain()``
    flushes everything accumulated so far as ONE binary message, which
    preserves the wire invariant that one framed envelope == one WS
    message. The wire ``_send`` helper writes the full encoded
    envelope before awaiting drain, so the buffer holds exactly one
    frame at that point.
    """

    def __init__(self, ws: Connection) -> None:
        self._ws = ws
        self._buf = bytearray()
        self._closed = False

    def write(self, data: bytes) -> None:
        if self._closed:
            raise ConnectionResetError("writer is closed")
        self._buf.extend(data)

    async def drain(self) -> None:
        if not self._buf:
            return
        payload = bytes(self._buf)
        self._buf.clear()
        try:
            await self._ws.send(payload)
        except ConnectionClosed as exc:
            raise ConnectionResetError("websocket closed") from exc

    def is_closing(self) -> bool:
        # ``Connection`` does not expose a public ``closed`` flag on all
        # versions, but ``close_code`` becomes non-None once closing
        # starts.
        return self._closed or self._ws.close_code is not None

    def close(self) -> None:
        self._closed = True
        if self._ws.close_code is None:
            # ``Connection.close()`` is async; fire-and-forget here to
            # match the sync ``StreamWriter.close()`` contract.
            asyncio.get_event_loop().create_task(self._ws.close())

    async def wait_closed(self) -> None:
        try:
            await self._ws.wait_closed()
        except Exception:
            pass

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        # Peer-cred auth calls ``get_extra_info("socket")``; WS has no
        # AF_UNIX peer credential, so we return ``default`` and rely on
        # the server's startup-time isinstance check to refuse the
        # pairing entirely.
        if name == "peername":
            return self._ws.remote_address
        return default


__all__ = ["WebSocketStreamReader", "WebSocketStreamWriter"]
