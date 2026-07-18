"""Transport abstraction for the gateway wire layer.

The Unix-socket plumbing is split out of :class:`WireServer` /
:class:`WireClient` behind a transport-agnostic interface, so a
WebSocket transport (:mod:`websocket`) sits alongside :mod:`unix`.
See ``.claude/designs/single-process-gateway.md``.
"""

from __future__ import annotations

from .base import ClientTransport, ConnectionHandler, ServerTransport
from .unix import UnixClientTransport, UnixServerTransport
from .websocket import WebSocketClientTransport, WebSocketServerTransport

__all__ = [
    "ClientTransport",
    "ConnectionHandler",
    "ServerTransport",
    "UnixClientTransport",
    "UnixServerTransport",
    "WebSocketClientTransport",
    "WebSocketServerTransport",
]
