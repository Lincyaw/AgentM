"""Transport abstraction for the channels wire layer.

PR1 of `.claude/plans/2026-05-12-gateway-websocket-transport.md`:
extract the Unix-socket plumbing out of :class:`WireServer` /
:class:`WireClient` behind a transport-agnostic interface. PR2 will
add a WebSocket implementation alongside :mod:`unix`.
"""

from __future__ import annotations

from .base import ClientTransport, ConnectionHandler, ServerTransport
from .unix import UnixClientTransport, UnixServerTransport

__all__ = [
    "ClientTransport",
    "ConnectionHandler",
    "ServerTransport",
    "UnixClientTransport",
    "UnixServerTransport",
]
