"""Authenticator implementations for the wire server.

The :class:`Authenticator` :class:`typing.Protocol` lives in
:mod:`agentm_channels.server` (the consumer). Concrete impls go here
so we can grow the set without bloating ``server.py``.

v1 ships a single concrete impl: :class:`UnixPeerCredAuthenticator`.
Per ``.claude/designs/client-server-architecture.md`` §6, bearer-token
and mTLS auth are deferred — Unix peer-cred + single-host gateway is
the only supported deployment shape in v1.
"""

from __future__ import annotations

from .peercred import UnixPeerCredAuthenticator

__all__ = ["UnixPeerCredAuthenticator"]
