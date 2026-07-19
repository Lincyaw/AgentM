"""Human interrupt atom — injects external feedback into the running session.

Listens on a Unix domain socket for the session. When the CLI sends a
message, it is queued in-memory and injected via DecideEvent -> Inject
after the current round's tool calls finish. The injection is persisted
in the trajectory (Turn.outcome.injected) and visible in context replay.

Socket path::

    ~/.agentm/inbox/<session_id>.sock

Protocol: each connection sends UTF-8 text terminated by EOF (close).
One message per connection.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agentm.core.abi import AtomAPI, ExtensionManifest
from agentm.core.abi.events import DecideEvent, Inject, SessionShutdownEvent
from agentm.core.abi.messages import synthetic_user_message
from loguru import logger
from pydantic import BaseModel


class HumanInterruptConfig(BaseModel):
    inbox_dir: str | None = None


MANIFEST = ExtensionManifest(
    name="human_interrupt",
    description="Injects external human feedback between rounds via Unix socket.",
    registers=(),
    config_schema=HumanInterruptConfig,
    requires=(),
    api_version=1,
    tier=2,
)


def _default_inbox_root() -> Path:
    return Path.home() / ".agentm" / "inbox"


def _socket_path(session_id: str, root: Path | None = None) -> Path:
    base = root if root else _default_inbox_root()
    return base / f"{session_id}.sock"


class _InboxServer:
    __slots__ = ("_path", "_queue", "_server")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self._path),
        )
        os.chmod(self._path, 0o600)
        logger.info("human_interrupt: listening on {}", self._path)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            data = await reader.read(64 * 1024)
            text = data.decode("utf-8").strip()
            if text:
                self._queue.put_nowait(text)
                logger.info("human_interrupt: received {} chars", len(text))
        except Exception as exc:
            logger.debug("human_interrupt: connection error: {}", exc)
        finally:
            writer.close()
            await writer.wait_closed()

    def drain(self) -> list[str]:
        messages: list[str] = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._path.unlink(missing_ok=True)


async def install(api: AtomAPI, config: HumanInterruptConfig) -> None:
    session_id = api.ctx.session_id
    inbox_root = Path(config.inbox_dir) if config.inbox_dir else _default_inbox_root()
    sock_path = _socket_path(session_id, inbox_root)

    server = _InboxServer(sock_path)
    await server.start()

    async def _on_decide(event: DecideEvent) -> Inject | None:
        pending = server.drain()
        if not pending:
            return None
        inject_messages = []
        for text in pending:
            logger.info("human_interrupt: injecting feedback ({} chars) at round {}", len(text), event.observation.turn_index)
            inject_messages.append(
                synthetic_user_message(
                    f"[Human Interrupt] The human operator has sent you the following message while you are working. Read it carefully and adjust your approach:\n\n{text}",
                    kind="human_interrupt",
                    origin="human",
                    visibility="visible",
                )
            )
        return Inject(messages=tuple(inject_messages))

    api.on(DecideEvent.CHANNEL, _on_decide, priority=100)

    async def _cleanup(_event: object) -> None:
        await server.stop()

    api.on(SessionShutdownEvent.CHANNEL, _cleanup)


__all__ = ("MANIFEST", "HumanInterruptConfig", "install")
