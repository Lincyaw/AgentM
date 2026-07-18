"""Unix-socket inbox listener: external processes can send messages to a live session.

Binds a Unix domain socket at ``~/.agentm/live/<session-id>.sock`` on
session start.  Any process can connect and send a UTF-8 line to inject a
user message into the running session's inbox.  The socket is removed on
session shutdown.

Delivery modes (controlled by the first line of the payload):

* ``!interrupt\\n<message>`` — abort the current tool execution, then
  deliver the message immediately.  The interrupted tool returns an error
  so the model knows the previous action was cancelled.
* ``!wait\\n<message>`` (default) — queue the message and deliver it at
  the next turn boundary, after the current tool finishes naturally.
  This avoids the "backgrounded tool result competes with injected
  message" problem.
* ``!now\\n<message>`` — deliver immediately (original behavior).  The
  current tool gets backgrounded.

External usage::

    agentm send --session <id> "stop exploring and start writing code"
    agentm send --session <id> --interrupt "stop now and write the fix"

Or manually::

    echo "focus on the fix now" | socat - UNIX-CONNECT:~/.agentm/live/<id>.sock
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict

from agentm.core.abi import ExtensionAPI, SessionShutdownEvent, TurnEndEvent
from agentm.core.lib import agentm_home_dir
from agentm.extensions import ExtensionManifest


class InboxSocketConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


MANIFEST = ExtensionManifest(
    name="inbox_socket",
    description="Unix-socket listener for external message injection into a live session.",
    registers=(),
    config_schema=InboxSocketConfig,
    requires=(),
    api_version=1,
    tier=1,
)


def _socket_dir() -> Path:
    return agentm_home_dir() / "live"


def _socket_path(session_id: str) -> Path:
    return _socket_dir() / f"{session_id}.sock"


class _InboxSocketRuntime:
    __slots__ = (
        "_api", "_session_id", "_server", "_sock_path",
        "_shutdown", "_pending",
    )

    def __init__(self, api: ExtensionAPI, session_id: str) -> None:
        self._api = api
        self._session_id = session_id
        self._sock_path = _socket_path(session_id)
        self._server: asyncio.AbstractServer | None = None
        self._shutdown = False
        self._pending: deque[str] = deque()

    async def start(self) -> None:
        sock_dir = _socket_dir()
        sock_dir.mkdir(parents=True, exist_ok=True)

        if self._sock_path.exists():
            self._sock_path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(self._sock_path),
        )
        os.chmod(str(self._sock_path), 0o600)
        logger.info("inbox_socket: listening on {}", self._sock_path)

    def _parse_mode(self, raw: str) -> tuple[str, str]:
        """Parse ``!mode\\nmessage`` prefix. Returns (mode, message)."""
        if raw.startswith("!interrupt\n"):
            return "interrupt", raw[len("!interrupt\n"):]
        if raw.startswith("!now\n"):
            return "now", raw[len("!now\n"):]
        if raw.startswith("!wait\n"):
            return "wait", raw[len("!wait\n"):]
        return "wait", raw

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            data = await asyncio.wait_for(reader.read(65536), timeout=5.0)
            text = data.decode("utf-8", errors="replace").strip()
            if not text or self._shutdown:
                writer.write(b"empty\n")
                await writer.drain()
                return

            mode, message = self._parse_mode(text)

            if mode == "now":
                self._api.post_inbox(source="user", payload=message)
                writer.write(b"ok:now\n")
                logger.info("inbox_socket: injected {} chars (now)", len(message))

            elif mode == "interrupt":
                self._api.post_inbox(source="user", payload=message)
                session = self._api.get_service("_session")
                if session is not None and hasattr(session, "interrupt"):
                    session.interrupt()
                    writer.write(b"ok:interrupted\n")
                    logger.info("inbox_socket: injected {} chars + interrupt", len(message))
                else:
                    writer.write(b"ok:now (interrupt unavailable)\n")
                    logger.info("inbox_socket: injected {} chars (interrupt unavailable)", len(message))

            else:
                self._pending.append(message)
                writer.write(b"ok:queued\n")
                logger.info("inbox_socket: queued {} chars for next turn boundary", len(message))

            await writer.drain()
        except Exception as exc:
            logger.debug("inbox_socket: client error: {}", exc)
        finally:
            writer.close()

    async def _on_turn_end(self, _event: TurnEndEvent) -> None:
        """Drain queued messages at the turn boundary."""
        while self._pending:
            message = self._pending.popleft()
            self._api.post_inbox(source="user", payload=message)
            logger.info("inbox_socket: delivered queued message ({} chars) at turn boundary", len(message))

    async def stop(self, _event: SessionShutdownEvent) -> None:
        self._shutdown = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._sock_path.exists():
            try:
                self._sock_path.unlink()
            except OSError:
                pass
        logger.debug("inbox_socket: stopped")


async def install(api: ExtensionAPI, config: InboxSocketConfig) -> None:
    session_id = api.session_id
    runtime = _InboxSocketRuntime(api, session_id)
    await runtime.start()

    api.on("turn_end", runtime._on_turn_end)
    api.on("session_shutdown", runtime.stop)
