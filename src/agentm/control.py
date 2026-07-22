"""Host-side control transport for running AgentM sessions."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Protocol

from loguru import logger

from agentm.core.abi.trigger import TriggerPriority

_MAX_MESSAGE_BYTES = 64 * 1024
_CONTROL_PREFIX = b"agentm-control/1 "
_COMPACT_REQUEST = _CONTROL_PREFIX + b"compact\n"
_INTERRUPT_PREFIX = _CONTROL_PREFIX + b"interrupt\n"


class ControllableSession(Protocol):
    @property
    def session_id(self) -> str: ...

    async def prompt(
        self,
        text: str,
        *,
        priority: TriggerPriority,
        origin: str | None,
        mode: str,
    ) -> object: ...

    def compact(self) -> None: ...


class InterruptDeliveryError(RuntimeError):
    """The target session rejected an interrupt message."""


class CompactionDeliveryError(RuntimeError):
    """The target session rejected a compaction request."""


def control_socket_path(session_id: str, root: Path | None = None) -> Path:
    if not session_id or Path(session_id).name != session_id:
        raise ValueError("session id must be a non-empty path-safe name")
    base = root if root is not None else Path.home() / ".agentm" / "inbox"
    return base / f"{session_id}.sock"


async def _send_control_request(
    session_id: str,
    request: bytes,
    *,
    inbox_root: Path | None,
) -> str:
    path = control_socket_path(session_id, inbox_root)
    if not path.exists():
        raise FileNotFoundError(f"no active session {session_id} (socket not found)")
    reader, writer = await asyncio.open_unix_connection(str(path))
    try:
        writer.write(request)
        writer.write_eof()
        await writer.drain()
        return (await reader.readline()).decode("utf-8", errors="replace").strip()
    finally:
        writer.close()
        await writer.wait_closed()


async def send_interrupt(
    session_id: str,
    message: str,
    *,
    inbox_root: Path | None = None,
) -> None:
    if not message.strip():
        raise ValueError("interrupt message must be non-empty")
    response = await _send_control_request(
        session_id,
        _INTERRUPT_PREFIX + message.encode("utf-8"),
        inbox_root=inbox_root,
    )
    if response != "ok":
        raise InterruptDeliveryError(response or "session returned no acknowledgement")


async def send_compact(
    session_id: str,
    *,
    inbox_root: Path | None = None,
) -> None:
    response = await _send_control_request(
        session_id,
        _COMPACT_REQUEST,
        inbox_root=inbox_root,
    )
    if response != "ok":
        raise CompactionDeliveryError(response or "session returned no acknowledgement")


class SessionControlServer:
    """Expose live session controls over a local owner-only Unix socket."""

    __slots__ = ("_path", "_server", "_session")

    def __init__(
        self,
        session: ControllableSession,
        *,
        inbox_root: Path | None = None,
    ) -> None:
        self._session = session
        self._path = control_socket_path(session.session_id, inbox_root)
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
                raise ValueError(
                    f"session control socket path is too long: {self._path}"
                ) from exc
            raise
        os.chmod(self._path, 0o600)
        logger.info("session control: listening on {}", self._path)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        response = b"ok\n"
        try:
            data = await reader.read(_MAX_MESSAGE_BYTES + len(_INTERRUPT_PREFIX) + 1)
            if data == _COMPACT_REQUEST:
                self._session.compact()
                logger.info("session control: compaction scheduled")
                return
            if data.startswith(_INTERRUPT_PREFIX):
                data = data[len(_INTERRUPT_PREFIX) :]
            if len(data) > _MAX_MESSAGE_BYTES:
                response = (
                    f"error: message exceeds {_MAX_MESSAGE_BYTES} bytes\n".encode()
                )
                return
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
                    "session control: interrupted active turn and queued {} chars",
                    len(text),
                )
        except Exception as exc:
            logger.warning("session control delivery failed: {}", exc)
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


__all__ = [
    "CompactionDeliveryError",
    "InterruptDeliveryError",
    "SessionControlServer",
    "control_socket_path",
    "send_compact",
    "send_interrupt",
]
