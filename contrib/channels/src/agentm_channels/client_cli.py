"""Shared client-side ``--connect`` resolver.

The terminal / worker / feishu CLIs all parse the same set of URL
schemes and TLS knobs. Centralizing the dispatch keeps their per-CLI
help text consistent and the error messages identical.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from urllib.parse import urlparse

from .transport import ClientTransport, UnixClientTransport, WebSocketClientTransport


class ConnectError(ValueError):
    """Raised when the ``--connect`` URL is malformed or paired with
    incompatible flags (e.g. ``--tls-ca`` on a unix:// URL)."""


@dataclass(frozen=True)
class ConnectSpec:
    """Resolved client-side endpoint."""

    scheme: str
    socket_path: str = ""  # unix
    uri: str = ""  # ws/wss


def resolve_connect(
    url: str,
    *,
    tls_ca: str | None = None,
) -> tuple[ConnectSpec, ClientTransport]:
    """Parse ``url`` and return (spec, transport).

    Raises :class:`ConnectError` with a human-readable message on bad input.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme == "unix":
        if tls_ca:
            raise ConnectError(
                "--tls-ca is only meaningful with wss:// (got unix://)."
            )
        socket_path = parsed.path or parsed.netloc
        if not socket_path or not socket_path.startswith("/"):
            raise ConnectError(
                f"--connect URL {url!r} has no absolute socket path"
            )
        return (
            ConnectSpec(scheme="unix", socket_path=socket_path),
            UnixClientTransport(socket_path),
        )
    if scheme in ("ws", "wss"):
        if scheme == "ws" and tls_ca:
            raise ConnectError(
                "--tls-ca is only meaningful with wss:// (got ws://)."
            )
        ssl_context: ssl.SSLContext | None = None
        if scheme == "wss":
            ssl_context = ssl.create_default_context()
            if tls_ca:
                ssl_context.load_verify_locations(cafile=tls_ca)
        return (
            ConnectSpec(scheme=scheme, uri=url),
            WebSocketClientTransport(url, ssl_context=ssl_context),
        )
    raise ConnectError(
        f"--connect scheme {scheme!r} is not supported; "
        "use unix://, ws://, or wss://."
    )


__all__ = ["ConnectError", "ConnectSpec", "resolve_connect"]
