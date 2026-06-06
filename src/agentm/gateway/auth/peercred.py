"""Unix peer-credential :class:`Authenticator`.

The connecting process's uid is read from the kernel (no token, no
trust in client-supplied identity claims) and matched against an
allow-list. Linux uses ``SO_PEERCRED``; macOS / *BSD bind libc
``getpeereid(2)`` via ctypes (CPython has no ``os.getpeereid``).

Design rationale (``.claude/designs/client-server-architecture.md``
§5, §6): a Unix-socket gateway on a trusted host doesn't need bearer
tokens — the kernel is a stronger authenticator than anything we can
ship in user space. The token field on ``hello`` is ignored in this
mode by design (defense-in-depth tokens are a v2 follow-up).
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import logging
import socket
import struct
import sys
from typing import Any

log = logging.getLogger("agentm.gateway.auth")


# Linux SO_PEERCRED struct ucred = (pid, uid, gid), three native ints.
_UCRED_FMT = "3i"
_UCRED_SIZE = struct.calcsize(_UCRED_FMT)


def _libc_getpeereid(fd: int) -> int | None:
    """Read peer uid via libc ``getpeereid(2)`` (macOS / *BSD).

    CPython's stdlib has no ``os.getpeereid``, so we bind the C symbol
    directly. Returns ``None`` if the symbol is missing or the call fails.
    """
    libc_name = ctypes.util.find_library("c")
    try:
        libc = ctypes.CDLL(libc_name or "libc.so.6", use_errno=True)
        getpeereid = libc.getpeereid
    except (OSError, AttributeError):
        return None
    # uid_t / gid_t are 32-bit unsigned on macOS and the BSDs.
    getpeereid.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    getpeereid.restype = ctypes.c_int
    uid = ctypes.c_uint32()
    gid = ctypes.c_uint32()
    if getpeereid(fd, ctypes.byref(uid), ctypes.byref(gid)) != 0:
        return None
    return int(uid.value)


def _peer_uid(sock: Any) -> int | None:
    """Return the connecting process's uid, or ``None`` on failure.

    Linux: ``SO_PEERCRED`` (kernel-vouched). macOS / *BSD: libc
    ``getpeereid``. Wrapped so an unsupported platform / closed socket
    simply yields ``None`` instead of crashing the handshake.
    """
    if sys.platform.startswith("linux"):
        try:
            data = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED_SIZE)
        except OSError:
            return None
        try:
            _pid, uid, _gid = struct.unpack(_UCRED_FMT, data)
        except struct.error:
            return None
        return int(uid)
    # macOS / *BSD: kernel-vouched peer uid via libc getpeereid(2).
    try:
        return _libc_getpeereid(sock.fileno())
    except OSError:
        return None


class UnixPeerCredAuthenticator:
    """Allow connections by kernel-vouched peer uid.

    Args:
        allowed_uids: ``None`` accepts any local uid; an explicit empty
            set denies all (useful for a circuit-breaker config); a
            populated set is the allow-list.

    Tokens in the ``hello`` envelope are ignored on purpose — the
    kernel is the source of truth in peer-cred mode.
    """

    def __init__(self, allowed_uids: set[int] | None = None) -> None:
        self._allowed_uids = allowed_uids

    async def authenticate(
        self,
        peer_id: str,
        token: str | None,  # noqa: ARG002 — peer-cred ignores tokens by design
        transport: asyncio.StreamWriter,
    ) -> bool:
        sock = transport.get_extra_info("socket")
        # asyncio wraps the kernel socket in :class:`asyncio.TransportSocket`,
        # which proxies ``getsockopt`` and ``fileno`` — both of which is
        # all peer-cred lookup needs.
        if sock is None or not hasattr(sock, "fileno"):
            log.warning("peer-cred reject: peer=%s — no socket on transport", peer_id)
            return False
        uid = _peer_uid(sock)
        if uid is None:
            log.warning(
                "peer-cred reject: peer=%s — could not read peer uid "
                "(unsupported platform?)",
                peer_id,
            )
            return False
        if self._allowed_uids is None:
            log.info("peer-cred accept: peer=%s uid=%d (any-uid policy)", peer_id, uid)
            return True
        if uid in self._allowed_uids:
            log.info("peer-cred accept: peer=%s uid=%d", peer_id, uid)
            return True
        log.warning(
            "peer-cred reject: peer=%s uid=%d not in allow-list %s",
            peer_id,
            uid,
            sorted(self._allowed_uids),
        )
        return False


__all__ = ["UnixPeerCredAuthenticator"]
