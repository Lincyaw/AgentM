"""UnixPeerCredAuthenticator over a real Unix socket pair."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import pytest

from agentm_channels.auth.peercred import UnixPeerCredAuthenticator


_PEERCRED_SUPPORTED = sys.platform.startswith("linux") or hasattr(os, "getpeereid")

pytestmark = pytest.mark.skipif(
    not _PEERCRED_SUPPORTED,
    reason="SO_PEERCRED / getpeereid not available on this platform",
)


async def _open_pair() -> tuple[
    asyncio.base_events.Server,
    asyncio.StreamWriter,  # server-side writer (carries the accepted socket)
    asyncio.StreamWriter,  # client-side writer
    asyncio.Event,  # set when handler should release
]:
    tmp = tempfile.mkdtemp(prefix="agentm-peercred-")
    path = os.path.join(tmp, "s.sock")
    ready = asyncio.Event()
    release = asyncio.Event()
    captured: dict[str, asyncio.StreamWriter] = {}

    async def handle(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        captured["w"] = w
        ready.set()
        await release.wait()
        w.close()
        try:
            await w.wait_closed()
        except Exception:
            pass

    server = await asyncio.start_unix_server(handle, path=path)
    _r, cw = await asyncio.open_unix_connection(path=path)
    await asyncio.wait_for(ready.wait(), timeout=2.0)
    return server, captured["w"], cw, release


async def _teardown(
    server: asyncio.base_events.Server,
    client_writer: asyncio.StreamWriter,
    release: asyncio.Event,
) -> None:
    release.set()
    client_writer.close()
    try:
        await client_writer.wait_closed()
    except Exception:
        pass
    server.close()
    await server.wait_closed()






async def test_authenticator_rejects_unknown_uid() -> None:
    server, sw, cw, release = await _open_pair()
    try:
        auth = UnixPeerCredAuthenticator(allowed_uids={os.geteuid() + 1})
        assert await auth.authenticate("chat_client", "p1", None, sw) is False
    finally:
        await _teardown(server, cw, release)




async def test_authenticator_empty_allow_set_denies_all() -> None:
    server, sw, cw, release = await _open_pair()
    try:
        auth = UnixPeerCredAuthenticator(allowed_uids=set())
        assert await auth.authenticate("chat_client", "p1", None, sw) is False
    finally:
        await _teardown(server, cw, release)
