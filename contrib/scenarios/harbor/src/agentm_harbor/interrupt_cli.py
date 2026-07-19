"""CLI to send interrupt messages to a running AgentM harbor session.

Usage::

    python -m agentm_harbor.interrupt_cli <session_id> <message>
    python -m agentm_harbor.interrupt_cli <session_id>  # reads from stdin
"""

from __future__ import annotations

import asyncio
import sys

from agentm_harbor.human_interrupt import socket_path


class InterruptDeliveryError(RuntimeError):
    """The target session rejected an interrupt message."""


async def _send(session_id: str, message: str) -> None:
    sock = socket_path(session_id)
    if not sock.exists():
        raise InterruptDeliveryError(f"no active session {session_id} (socket not found)")
    reader, writer = await asyncio.open_unix_connection(str(sock))
    try:
        writer.write(message.encode("utf-8"))
        writer.write_eof()
        await writer.drain()
        response = (
            (await reader.readline())
            .decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        )
    finally:
        writer.close()
        await writer.wait_closed()
    if response != "ok":
        raise InterruptDeliveryError(response or "session returned no acknowledgement")
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
    try:
        asyncio.run(_send(session_id, message))
    except (InterruptDeliveryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
