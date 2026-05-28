"""WebSocket :class:`ServerTransport` and :class:`ClientTransport`.

Each WebSocket *binary message* carries exactly one already-framed
envelope (length-prefix header + JSON body bytes from
``wire/framing.py``). The framing reader / writer never see WebSockets
directly — :mod:`._stream_adapter` exposes the connection as
asyncio-StreamReader/StreamWriter-shaped objects so
``server._read_one_frame`` and ``server._send`` work unchanged.
"""

from __future__ import annotations

import asyncio
import ssl

from websockets.asyncio.client import ClientConnection, connect as ws_connect
from websockets.asyncio.server import ServerConnection, serve as ws_serve

from ._stream_adapter import WebSocketStreamReader, WebSocketStreamWriter
from .base import ConnectionHandler


class WebSocketServerTransport:
    """Bind a WebSocket listener on ``host:port``.

    URL path routing is intentionally not enforced here — a reverse
    proxy in front of the listener owns the URL surface today, and the
    handshake accepts any path the client requests.
    """

    def __init__(
        self,
        host: str,
        port: int,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._ssl = ssl_context
        self._server: object | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        # Resolved port after binding (useful when caller passed 0).
        if self._server is not None:
            sockets = getattr(self._server, "sockets", None)
            if sockets:
                return int(sockets[0].getsockname()[1])
        return self._port

    async def serve(self, handle: ConnectionHandler) -> None:
        if self._server is not None:
            return

        async def _handler(ws: ServerConnection) -> None:
            reader = WebSocketStreamReader(ws)
            writer = WebSocketStreamWriter(ws)
            try:
                # The wire framing helpers expect StreamReader /
                # StreamWriter; our adapters duck-type those.
                await handle(reader, writer)  # type: ignore[arg-type]
            finally:
                if ws.close_code is None:
                    await ws.close()

        # ``websockets.serve`` returns an awaitable that resolves to a
        # ``Server`` once the listener is bound.
        self._server = await ws_serve(
            _handler,
            host=self._host,
            port=self._port,
            ssl=self._ssl,
        )

    async def close(self) -> None:
        if self._server is None:
            return
        server = self._server
        self._server = None
        close_fn = getattr(server, "close", None)
        if close_fn is not None:
            close_fn()
        wait_fn = getattr(server, "wait_closed", None)
        if wait_fn is not None:
            try:
                await wait_fn()
            except Exception:
                pass


class WebSocketClientTransport:
    """Connect to ``uri`` (``ws://...`` or ``wss://...``)."""

    def __init__(self, uri: str, ssl_context: ssl.SSLContext | None = None) -> None:
        self._uri = uri
        self._ssl = ssl_context
        self._ws: ClientConnection | None = None

    @property
    def uri(self) -> str:
        return self._uri

    async def connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        kwargs: dict[str, object] = {}
        if self._ssl is not None:
            kwargs["ssl"] = self._ssl
        ws: ClientConnection = await ws_connect(self._uri, **kwargs)  # type: ignore[arg-type]
        self._ws = ws
        reader = WebSocketStreamReader(ws)
        writer = WebSocketStreamWriter(ws)
        # Duck-typed return; callers use only the StreamReader/Writer
        # subset that the wire helpers exercise.
        return reader, writer  # type: ignore[return-value]


__all__ = ["WebSocketClientTransport", "WebSocketServerTransport"]
