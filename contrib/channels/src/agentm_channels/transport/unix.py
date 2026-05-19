"""Unix-socket implementations of :class:`ServerTransport` and
:class:`ClientTransport`.

Behaviour is identical to the pre-PR1 inline logic in ``server.py`` /
``client.py``: the server cleans a stale socket file on bind and on
close; the client opens via :func:`asyncio.open_unix_connection`.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import socket

from .base import ConnectionHandler


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
        self._socket_path = socket_path
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
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self._socket_path)
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
        with contextlib.suppress(OSError):
            os.chmod(self._socket_path, 0o600)

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
