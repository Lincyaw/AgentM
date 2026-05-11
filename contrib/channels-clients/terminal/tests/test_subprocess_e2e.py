"""Subprocess E2E for ``agentm-terminal``.

We don't spin up the full ``agentm-gateway`` CLI here — it pulls the
LLM session factory, which is heavy and not what we're testing.
Instead, we run a minimal ``WireServer`` in-process inside the test
loop (the same pattern Phase 1 used in ``tests/server/test_echo.py``)
and connect the terminal CLI as a real subprocess via its installed
console script.

Each test uses ``tmp_path`` for the Unix socket so we never collide
in /tmp.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agentm_channels.auth import UnixPeerCredAuthenticator
from agentm_channels.outbox import SqliteInbox, SqliteOutbox
from agentm_channels.peer import PeerSession
from agentm_channels.server import WireServer
from agentm_channels.wire import WIRE_VERSION, Envelope


# ----- helpers --------------------------------------------------------


async def _start_echo_server(
    socket_path: str, db_path: str, *, allow_uids: set[int] | None = None
) -> tuple[WireServer, SqliteOutbox, SqliteInbox]:
    outbox = SqliteOutbox(db_path)
    inbox = SqliteInbox(db_path)

    async def on_inbound(session: PeerSession, env: Envelope) -> None:
        # Echo the inbound back as a plain outbound message. The
        # renderer expects ``content`` / optional ``buttons`` /
        # optional ``metadata``; we mirror the inbound body's
        # ``content`` so the test can assert round-trip.
        body = env.body if isinstance(env.body, dict) else {}
        echo = Envelope(
            v=WIRE_VERSION,
            id=f"echo-{env.id}",
            kind="outbound",
            ts=time.time(),
            body={"content": str(body.get("content") or ""), "kind": "message"},
        )
        outbox.enqueue(session.peer_id, echo)

    authenticator = (
        UnixPeerCredAuthenticator(allowed_uids=allow_uids)
        if allow_uids is not None
        else None
    )
    server = WireServer(
        socket_path=socket_path,
        outbox=outbox,
        inbox=inbox,
        on_inbound=on_inbound,
        authenticator=authenticator,
    )
    await server.start()
    return server, outbox, inbox


def _spawn_terminal(
    *,
    socket_path: str,
    extra: list[str] | None = None,
    stdin: int | None = subprocess.PIPE,
) -> subprocess.Popen[str]:
    # Invoke through ``python -m agentm_terminal.cli`` to avoid relying
    # on the console-script entry being on PATH inside the workspace.
    cmd = [
        sys.executable,
        "-m",
        "agentm_terminal.cli",
        "--connect",
        f"unix://{socket_path}",
        "--format",
        "json",
        *(extra or []),
    ]
    return subprocess.Popen(
        cmd,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _wait_proc(proc: subprocess.Popen[str], timeout: float) -> int:
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)
        raise


def _read_until_kind(
    proc: subprocess.Popen[str], kind: str, timeout: float
) -> dict[str, Any]:
    """Read stdout JSON lines until we see one with ``kind == kind``."""
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("kind") == kind:
            return obj
    raise AssertionError(f"timed out waiting for kind={kind!r}")


# ----- tests ----------------------------------------------------------


async def test_echo_round_trip(tmp_path: Path) -> None:
    sock = str(tmp_path / "gw.sock")
    db = str(tmp_path / "outbox.sqlite")
    server, outbox, inbox = await _start_echo_server(
        sock, db, allow_uids={os.geteuid()}
    )
    proc = _spawn_terminal(socket_path=sock)
    try:
        # Wait for the ``ready`` line (announced after WELCOME).
        ready = await asyncio.to_thread(_read_until_kind, proc, "ready", 5.0)
        assert ready == {"kind": "ready"}
        # Send a line on stdin; expect an echo back as a ``message``.
        assert proc.stdin is not None
        proc.stdin.write("hello\n")
        proc.stdin.flush()
        msg = await asyncio.to_thread(_read_until_kind, proc, "message", 5.0)
        assert msg["content"] == "hello"
    finally:
        if proc.poll() is None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        await server.stop()
        outbox.close()
        inbox.close()


async def test_clean_eof_shutdown(tmp_path: Path) -> None:
    sock = str(tmp_path / "gw.sock")
    db = str(tmp_path / "outbox.sqlite")
    server, outbox, inbox = await _start_echo_server(
        sock, db, allow_uids={os.geteuid()}
    )
    proc = _spawn_terminal(socket_path=sock)
    try:
        await asyncio.to_thread(_read_until_kind, proc, "ready", 5.0)
        # Close stdin → client should EOF its read loop and exit 0.
        assert proc.stdin is not None
        proc.stdin.close()
        rc = await asyncio.to_thread(_wait_proc, proc, 5.0)
        assert rc == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)
        await server.stop()
        outbox.close()
        inbox.close()


async def test_handshake_rejection(tmp_path: Path) -> None:
    sock = str(tmp_path / "gw.sock")
    db = str(tmp_path / "outbox.sqlite")
    # Allow only an impossible uid → peer-cred auth rejects every
    # local connection with ``auth_failed``.
    server, outbox, inbox = await _start_echo_server(
        sock, db, allow_uids={99999}
    )
    proc = _spawn_terminal(socket_path=sock)
    try:
        rc = await asyncio.to_thread(_wait_proc, proc, 5.0)
        assert rc == 4, f"expected EXIT_AUTH=4, got {rc}; stderr={proc.stderr.read() if proc.stderr else ''}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)
        await server.stop()
        outbox.close()
        inbox.close()


def test_bad_connect_url_subprocess() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "agentm_terminal.cli", "--connect", "http://x"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2
    assert "agentm-terminal: error" in proc.stderr


def test_missing_socket_subprocess(tmp_path: Path) -> None:
    missing = tmp_path / "definitely-not-there.sock"
    assert not missing.exists()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentm_terminal.cli",
            "--connect",
            f"unix://{missing}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 7, proc.stderr
    assert "connect-failed" in proc.stderr
