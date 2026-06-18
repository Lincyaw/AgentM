"""Transport-specific regression tests for long AF_UNIX socket paths."""

from __future__ import annotations

import asyncio

import pytest
from agentm.gateway.transport import UnixClientTransport, UnixServerTransport


def test_unix_transport_shortens_long_path() -> None:
    long_path = "/tmp/" + "a" * 220 + ".sock"
    transport = UnixClientTransport(long_path)
    assert len(transport.socket_path) < len(long_path)
    assert len(transport.socket_path) < 104
    assert transport.socket_path != long_path


async def _echo_server(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    writer.close()


@pytest.mark.asyncio
async def test_unix_transport_roundtrip_with_shortened_path() -> None:
    long_path = "/tmp/" + "b" * 220 + ".sock"
    server = UnixServerTransport(long_path)
    try:
        await server.serve(_echo_server)
        reader, writer = await UnixClientTransport(long_path).connect()
        writer.write(b"hi")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        assert reader.at_eof()
    finally:
        await server.close()
