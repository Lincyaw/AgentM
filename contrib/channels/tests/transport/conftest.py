"""Shared fixtures for WebSocket transport tests."""

from __future__ import annotations

import os
import socket
import tempfile
from collections.abc import Iterator

import pytest


@pytest.fixture
def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def tmp_dir() -> Iterator[str]:
    with tempfile.TemporaryDirectory(prefix="agentm-wsx-") as d:
        yield d


@pytest.fixture
def db_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "outbox.sqlite")


@pytest.fixture
def socket_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "gateway.sock")
