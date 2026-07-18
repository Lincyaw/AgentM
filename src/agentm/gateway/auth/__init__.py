"""Authenticator surface for the single-process gateway.

The :class:`Authenticator` :class:`typing.Protocol` is one of the three
documented pluggability axes (§6). Hello-time identity check only — no
``peer_kind`` is passed because only ``chat_client`` peers exist in v2
(§2.3). Default impls:

* :class:`AllowAllAuthenticator` — accept every hello (tests, trusted hosts).
* :class:`UnixPeerCredAuthenticator` — kernel-vouched peer uid allow-list.
* :class:`TokenAuthenticator` — bearer-token allow-list (ws/wss).
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from .peercred import UnixPeerCredAuthenticator
from .token import TokenAuthenticator


@runtime_checkable
class Authenticator(Protocol):
    """Server-side hook for accepting/rejecting a ``hello``."""

    async def authenticate(
        self,
        peer_id: str,
        token: str | None,
        transport: asyncio.StreamWriter,
    ) -> bool:
        ...


class AllowAllAuthenticator:
    """Default — accept every hello. Suitable for tests + Unix socket on
    trusted hosts.
    """

    async def authenticate(
        self,
        peer_id: str,
        token: str | None,
        transport: asyncio.StreamWriter,
    ) -> bool:
        return True


__all__ = [
    "AllowAllAuthenticator",
    "Authenticator",
    "TokenAuthenticator",
    "UnixPeerCredAuthenticator",
]
