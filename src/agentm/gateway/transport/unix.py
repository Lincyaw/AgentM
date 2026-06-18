"""Unix-socket implementations of :class:`ServerTransport` and
:class:`ClientTransport`.

Behaviour is identical to the pre-PR1 inline logic in ``server.py`` /
``client.py``: the server cleans a stale socket file on bind and on
close; the client opens via :func:`asyncio.open_unix_connection`.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import re
import socket
import tempfile

from loguru import logger

from .base import ConnectionHandler


_MAX_UNIX_SOCKET_PATH_BYTES: int = 104


def _resolve_unix_socket_path(socket_path: str) -> str:
    """Return a bind-ready unix socket path.

    macOS/AF_UNIX sockets cannot exceed a fixed byte limit (~104 bytes) in
    ``sun_path``. Pytest temp roots can exceed this with nested names, and
    direct bind/connect then fails with ``OSError: AF_UNIX path too long``.
    We keep the user-facing long path for API stability and map to a stable
    short fallback under the process temp dir when needed.
    """
    if len(socket_path.encode("utf-8")) <= _MAX_UNIX_SOCKET_PATH_BYTES:
        return socket_path
    digest = hashlib.sha256(socket_path.encode("utf-8")).hexdigest()[:12]
    base = os.path.basename(socket_path)
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", stem)[:12] or "gateway"
    short_name = f"agentm-gw-{stem}-{digest}.sock"
    fallback = os.path.join(tempfile.gettempdir(), short_name)
    # The fallback path stays safely short on normal hosts.
    if len(fallback.encode("utf-8")) > _MAX_UNIX_SOCKET_PATH_BYTES:
        # If tmpdir itself is already unexpectedly long, keep the final fallback
        # short by dropping stem suffix.
        short_name = f"agentm-{digest}.sock"
        fallback = os.path.join(tempfile.gettempdir(), short_name)
    return fallback


# Short connect timeout for the stale-socket liveness probe (issue #4).
# 200ms is more than enough for an in-host AF_UNIX connect to either
# succeed or trip ECONNREFUSED; longer would penalise legitimate
# gateway restarts after a clean shutdown that didn't unlink.
_STALE_PROBE_TIMEOUT_S: float = 0.2


def _is_stale_socket(socket_path: str) -> bool:
    """Return True iff a socket file at ``socket_path`` has no listener.

    Tries a non-blocking AF_UNIX connect. ECONNREFUSED / ENOENT mean
    the file is leftover from a crashed predecessor and is safe to
    unlink. A successful connect means a live gateway owns this path
    and we MUST NOT steal it. Any other errno (e.g. EACCES) bubbles up
    as ``False`` (treat as live to be safe).
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(_STALE_PROBE_TIMEOUT_S)
            try:
                probe.connect(socket_path)
            except (FileNotFoundError, ConnectionRefusedError):
                return True
            except OSError as exc:
                if exc.errno in {errno.ECONNREFUSED, errno.ENOENT}:
                    return True
                # Unknown failure — assume live and refuse to clobber.
                return False
            # connect() succeeded → a live peer accepted us.
            return False
    except OSError:
        return False


class UnixServerTransport:
    """Bind to an `AF_UNIX` socket at ``socket_path``."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = _resolve_unix_socket_path(socket_path)
        self._server: asyncio.base_events.Server | None = None

    @property
    def socket_path(self) -> str:
        return self._socket_path

    async def serve(self, handle: ConnectionHandler) -> None:
        if self._server is not None:
            return
        # Stale-socket cleanup (issue #4): only unlink when the
        # existing socket file is provably abandoned. If a live
        # gateway answers, refuse to start instead of stealing its
        # bind — two gateways on the same path silently split traffic.
        if os.path.exists(self._socket_path):
            if not _is_stale_socket(self._socket_path):
                raise RuntimeError(
                    f"socket {self._socket_path} is already in use by "
                    "another gateway"
                )
            try:
                os.unlink(self._socket_path)
            except FileNotFoundError as exc:
                # Stale socket vanished between the check and the unlink — fine.
                logger.debug("unix transport: stale socket already gone: {}", exc)
        # Restrict the socket to the bind user (issue #4): with the
        # default umask the AF_UNIX node would be world-connectable,
        # letting any local user impersonate a peer. Set 0o077 around
        # the bind so the file is created mode 0o600 — chmod after-
        # the-fact has a TOCTOU window where another process may
        # connect before we tighten the perms.
        prev_umask = os.umask(0o077)
        try:
            self._server = await asyncio.start_unix_server(
                handle, path=self._socket_path
            )
        finally:
            os.umask(prev_umask)
        # Belt-and-braces: assert the resulting perms actually exclude
        # group/other. asyncio.start_unix_server honours the umask we
        # set, but the explicit chmod makes the invariant testable.
        try:
            os.chmod(self._socket_path, 0o600)
        except OSError as exc:
            # Perms-tightening failure could leave the socket group/other
            # connectable — surface it as a security-relevant warning.
            logger.warning(
                "unix transport: could not chmod 0600 socket {}; it may be "
                "over-permissive: {}",
                self._socket_path,
                exc,
            )

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception as exc:  # noqa: BLE001
                logger.debug("unix transport: server wait_closed raised: {}", exc)
            self._server = None
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError as exc:
            logger.debug("unix transport: socket already removed on close: {}", exc)


class UnixClientTransport:
    """Connect to an `AF_UNIX` socket at ``socket_path``."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = _resolve_unix_socket_path(socket_path)

    @property
    def socket_path(self) -> str:
        return self._socket_path

    async def connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await asyncio.open_unix_connection(path=self._socket_path)


__all__ = ["UnixClientTransport", "UnixServerTransport"]
