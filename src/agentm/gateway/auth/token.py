"""Bearer-token :class:`Authenticator`.

Reads ``hello.body.token`` (passed into ``authenticate`` by the server
— see ``server._handle_connection``) and matches it against a static
allow-list. An empty allow-list rejects every connection, mirroring
the ``UnixPeerCredAuthenticator(set())`` "circuit breaker" shape.

Tokens are matched verbatim; rotation/expiry is out of scope.
"""

from __future__ import annotations

import asyncio
import hmac
import logging

log = logging.getLogger("agentm.gateway.auth")


def _constant_time_membership(token: str, allowed: tuple[str, ...]) -> bool:
    """Return True iff ``token`` matches any entry in ``allowed``.

    Iterates the entire allow-list and OR-accumulates the per-entry
    :func:`hmac.compare_digest` result without an early return so the
    time taken does not leak which entry matched (or how many leading
    bytes of a non-matching token were correct). Both sides are encoded
    to ``utf-8`` because :func:`hmac.compare_digest` rejects mixed
    bytes/str on some Python builds.
    """
    token_bytes = token.encode("utf-8")
    matched = False
    for candidate in allowed:
        # `|=` keeps both branches executed for every candidate; the
        # bool() coercion is the OR accumulator.
        matched |= hmac.compare_digest(token_bytes, candidate.encode("utf-8"))
    return matched


class TokenAuthenticator:
    """Accept connections whose ``hello.body.token`` is in
    ``allowed_tokens``.

    Args:
        allowed_tokens: A set of acceptable bearer tokens. An empty
            set denies all (circuit-breaker).
    """

    def __init__(self, allowed_tokens: set[str]) -> None:
        # Freeze as a tuple so the comparison loop walks a deterministic
        # sequence — the constant-time guarantee assumes a fixed order
        # of probes per call.
        self._allowed_tokens: tuple[str, ...] = tuple(str(t) for t in allowed_tokens)

    async def authenticate(
        self,
        peer_id: str,
        token: str | None,
        transport: asyncio.StreamWriter,  # noqa: ARG002 — token auth ignores transport
    ) -> bool:
        if token is None or token == "":
            log.warning("token reject: peer=%s — no token presented", peer_id)
            return False
        # Constant-time membership check (issue #3): naive ``in`` against
        # a set leaks a timing oracle on the comparing bytes. Walk the
        # whole allow-list with ``hmac.compare_digest`` and only branch
        # on the OR-accumulated result.
        if _constant_time_membership(str(token), self._allowed_tokens):
            log.info("token accept: peer=%s", peer_id)
            return True
        log.warning("token reject: peer=%s — token not in allow-list", peer_id)
        return False


__all__ = ["TokenAuthenticator"]
