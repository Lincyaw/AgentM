"""Bearer-token :class:`Authenticator`.

Reads ``hello.body.token`` (passed into ``authenticate`` by the server
— see ``server._handle_connection``) and matches it against a static
allow-list. An empty allow-list rejects every connection, mirroring
the ``UnixPeerCredAuthenticator(set())`` "circuit breaker" shape.

Tokens are matched verbatim; rotation/expiry is out of scope (see
``.claude/plans/2026-05-12-gateway-websocket-transport.md``).
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("agentm_channels.auth")


class TokenAuthenticator:
    """Accept connections whose ``hello.body.token`` is in
    ``allowed_tokens``.

    Args:
        allowed_tokens: A set of acceptable bearer tokens. An empty
            set denies all (circuit-breaker).
    """

    def __init__(self, allowed_tokens: set[str]) -> None:
        self._allowed_tokens = set(allowed_tokens)

    async def authenticate(
        self,
        peer_kind: str,
        peer_id: str,
        token: str | None,
        transport: asyncio.StreamWriter,  # noqa: ARG002 — token auth ignores transport
    ) -> bool:
        if token is None or token == "":
            log.warning(
                "token reject: peer=%s kind=%s — no token presented",
                peer_id,
                peer_kind,
            )
            return False
        if token in self._allowed_tokens:
            log.info("token accept: peer=%s kind=%s", peer_id, peer_kind)
            return True
        log.warning(
            "token reject: peer=%s kind=%s — token not in allow-list",
            peer_id,
            peer_kind,
        )
        return False


__all__ = ["TokenAuthenticator"]
