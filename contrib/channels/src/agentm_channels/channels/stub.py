"""In-memory channel for tests / local dry-runs.

Outbound messages are recorded in :attr:`outbox`; inbound is pushed via
:meth:`push`. Both ends are async-safe so a test can drive the whole
gateway loop without monkeypatching.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..base import BaseChannel
from ..bus import OutboundMessage


class StubChannel(BaseChannel):
    name = "stub"
    display_name = "Stub (in-memory)"

    def __init__(self, config: Any, bus: Any) -> None:
        super().__init__(config, bus)
        self.outbox: list[OutboundMessage] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._running = True
        # Block until stop() is called so the manager's task stays alive.
        await self._stopped.wait()

    async def stop(self) -> None:
        self._running = False
        self._stopped.set()

    async def send(self, msg: OutboundMessage) -> None:
        self.outbox.append(msg)

    async def push(
        self,
        *,
        sender_id: str,
        chat_id: str,
        content: str,
        button_value: str | None = None,
        session_key: str | None = None,
    ) -> None:
        """Test helper. Mirrors what a real channel does on inbound."""
        meta: dict[str, Any] = {}
        if button_value is not None:
            meta["button_value"] = button_value
        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            metadata=meta,
            session_key=session_key,
        )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"enabled": False, "allow_from": ["*"]}
