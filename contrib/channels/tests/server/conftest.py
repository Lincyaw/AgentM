"""Shared fixtures for server integration tests.

Real Unix sockets in a per-test :class:`tempfile.TemporaryDirectory`
to keep /tmp clean and avoid cross-test path collisions.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from agentm_channels.peer import PeerSession
from agentm_channels.wire import WIRE_VERSION, Envelope

InboundHandler = Callable[[PeerSession, Envelope], Awaitable[None]]


@pytest.fixture
def tmp_sock_dir() -> "AsyncIterator[str]":  # type: ignore[type-arg]
    with tempfile.TemporaryDirectory(prefix="agentm-wire-") as d:
        yield d


@pytest.fixture
def db_path(tmp_sock_dir: str) -> str:
    return os.path.join(tmp_sock_dir, "outbox.sqlite")


@pytest.fixture
def socket_path(tmp_sock_dir: str) -> str:
    return os.path.join(tmp_sock_dir, "gateway.sock")


async def make_envelope(env_id: str, body: dict[str, object]) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=env_id,
        kind="outbound",
        ts=time.time(),
        body=body,
    )


async def wait_for(
    predicate: Callable[[], bool], timeout: float = 3.0, interval: float = 0.02
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()
