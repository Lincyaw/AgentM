from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agentm.control import SessionControlServer, send_interrupt
from agentm.core.abi.trigger import TriggerPriority


class _FakeSession:
    session_id = "control-session"

    def __init__(self) -> None:
        self.prompts: list[tuple[str, TriggerPriority, str | None, str]] = []

    async def prompt(
        self,
        text: str,
        *,
        priority: TriggerPriority,
        origin: str | None,
        mode: str,
    ) -> object:
        self.prompts.append((text, priority, origin, mode))
        return object()


@pytest.mark.asyncio
async def test_session_control_delivers_immediate_interrupt() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        inbox = Path(directory)
        session = _FakeSession()
        server = SessionControlServer(session, inbox_root=inbox)
        await server.start()
        try:
            await send_interrupt(
                session.session_id,
                "reconsider the task",
                inbox_root=inbox,
            )
        finally:
            await server.stop()

        assert session.prompts == [("reconsider the task", "now", "human", "interrupt")]
        assert not server.path.exists()
