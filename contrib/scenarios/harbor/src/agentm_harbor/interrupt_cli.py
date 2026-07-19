"""CLI to send interrupt messages to a running AgentM harbor session.

Usage::

    python -m agentm_harbor.interrupt_cli <session_id> <message>
    python -m agentm_harbor.interrupt_cli <session_id>  # reads from stdin
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _default_inbox_root() -> Path:
    return Path.home() / ".agentm" / "inbox"


def _socket_path(session_id: str) -> Path:
    return _default_inbox_root() / f"{session_id}.sock"


async def _send(session_id: str, message: str) -> None:
    sock = _socket_path(session_id)
    if not sock.exists():
        print(f"error: no active session {session_id} (socket not found)", file=sys.stderr)
        sys.exit(1)
    reader, writer = await asyncio.open_unix_connection(str(sock))
    writer.write(message.encode("utf-8"))
    writer.write_eof()
    await writer.drain()
    writer.close()
    await writer.wait_closed()
    print(f"sent {len(message)} chars to session {session_id}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: interrupt_cli <session_id> [message]", file=sys.stderr)
        sys.exit(1)
    session_id = sys.argv[1]
    if len(sys.argv) >= 3:
        message = " ".join(sys.argv[2:])
    else:
        message = sys.stdin.read().strip()
    if not message:
        print("error: empty message", file=sys.stderr)
        sys.exit(1)
    asyncio.run(_send(session_id, message))


if __name__ == "__main__":
    main()
