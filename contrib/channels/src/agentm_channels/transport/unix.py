"""Unix-socket implementations of :class:`ServerTransport` and
:class:`ClientTransport`.

Behaviour is identical to the pre-PR1 inline logic in ``server.py`` /
``client.py``: the server cleans a stale socket file on bind and on
close; the client opens via :func:`asyncio.open_unix_connection`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os

from .base import ConnectionHandler


class UnixServerTransport:
    """Bind to an `AF_UNIX` socket at ``socket_path``."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._server: asyncio.base_events.Server | None = None

    @property
    def socket_path(self) -> str:
        return self._socket_path

    async def serve(self, handle: ConnectionHandler) -> None:
        if self._server is not None:
            return
        # Stale-socket cleanup: a crashed predecessor may have left the
        # socket file on disk; start_unix_server would then fail with
        # EADDRINUSE.
        with contextlib.suppress(FileNotFoundError):
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        self._server = await asyncio.start_unix_server(
            handle, path=self._socket_path
        )

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._socket_path)


class UnixClientTransport:
    """Connect to an `AF_UNIX` socket at ``socket_path``."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path

    @property
    def socket_path(self) -> str:
        return self._socket_path

    async def connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await asyncio.open_unix_connection(path=self._socket_path)


__all__ = ["UnixClientTransport", "UnixServerTransport"]
