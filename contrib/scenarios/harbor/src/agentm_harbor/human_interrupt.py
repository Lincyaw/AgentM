"""Unix-socket bridge from Harbor operators to the public Session API."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agentm import AgentSession
from loguru import logger


def _default_inbox_root() -> Path:
    return Path.home() / ".agentm" / "inbox"


def socket_path(session_id: str, root: Path | None = None) -> Path:
    base = root if root is not None else _default_inbox_root()
    return base / f"{session_id}.sock"


class HumanInterruptServer:
    """Cancel the active turn and enqueue operator feedback as the next turn."""

    __slots__ = ("_path", "_server", "_session")

    def __init__(
        self,
        session: AgentSession,
        *,
        inbox_root: Path | None = None,
    ) -> None:
        self._session = session
        self._path = socket_path(session.session_id, inbox_root)
        self._server: asyncio.AbstractServer | None = None

    @property
    def path(self) -> Path:
        return self._path

    async def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.unlink(missing_ok=True)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                path=str(self._path),
            )
        except OSError as exc:
            if "path too long" in str(exc).lower():
                raise ValueError(f"human interrupt socket path is too long: {self._path}") from exc
            raise
        os.chmod(self._path, 0o600)
        logger.info("human interrupt: listening on {}", self._path)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        response = b"ok\n"
        try:
            data = await reader.read(64 * 1024)
            text = data.decode("utf-8").strip()
            if not text:
                response = b"error: empty message\n"
            else:
                await self._session.prompt(
                    text,
                    priority="now",
                    origin="human",
                    mode="interrupt",
                )
                logger.info(
                    "human interrupt: cancelled active turn and queued {} chars",
                    len(text),
                )
        except Exception as exc:
            logger.warning("human interrupt delivery failed: {}", exc)
            response = f"error: {exc}\n".encode("utf-8", errors="replace")
        finally:
            writer.write(response)
            try:
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._path.unlink(missing_ok=True)


__all__ = ("HumanInterruptServer", "socket_path")
