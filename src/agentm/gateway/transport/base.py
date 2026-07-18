"""Transport protocols for the channels wire layer.

The wire framing (length-prefixed JSON envelopes, see
:mod:`agentm.gateway.wire`) is transport-agnostic; this module pins
down the seam between the framing-aware server/client and the
underlying byte stream. Implementations expose the connection as an
:class:`asyncio.StreamReader` / :class:`asyncio.StreamWriter` pair so
the existing ``_read_one_frame`` / ``_send`` helpers work unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

ConnectionHandler = Callable[
    [asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]
]


@runtime_checkable
class ServerTransport(Protocol):
    """Bind-side transport: accepts connections and dispatches them."""

    async def serve(self, handle: ConnectionHandler) -> None:
        """Start accepting connections; return once listening.

        Each accepted connection invokes ``handle(reader, writer)`` as a
        task owned by the transport.
        """
        ...

    async def close(self) -> None:
        """Stop accepting; release listener-level resources. Idempotent."""
        ...


@runtime_checkable
class ClientTransport(Protocol):
    """Connect-side transport: opens a single byte-stream connection."""

    async def connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        ...


__all__ = ["ClientTransport", "ConnectionHandler", "ServerTransport"]
