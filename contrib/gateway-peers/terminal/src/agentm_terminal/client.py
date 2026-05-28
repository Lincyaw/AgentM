"""Thin glue between the gateway ``WireClient`` and the frontends.

Exposes an async outbound *stream* (``outbound()``) and ``send_inbound`` so a
frontend never touches the wire envelope directly — it just consumes outbound
``body`` dicts and pushes inbound ones. Scenario is sent on the first inbound
only (§2.2); the session_key is ``terminal:<chat_id>``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from agentm.gateway.client import WireClient
from agentm.gateway.transport import ClientTransport
from agentm.gateway.wire import KIND_ERROR, KIND_OUTBOUND, Envelope


class TerminalClient:
    """Owns one ``WireClient`` connection for a single terminal conversation."""

    def __init__(
        self,
        *,
        transport: ClientTransport,
        peer_name: str,
        token: str | None,
        chat_id: str,
        scenario: str | None,
    ) -> None:
        self._chat_id = chat_id
        self._scenario = scenario
        self._session_key = f"terminal:{chat_id}"
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._first_sent = False
        self._client = WireClient(
            transport=transport,
            peer_name=peer_name,
            token=token,
            on_outbound=self._on_outbound,
        )

    async def _on_outbound(self, env: Envelope) -> None:
        if env.kind == KIND_OUTBOUND:
            body = env.body if isinstance(env.body, dict) else {}
            channel = str(body.get("channel") or "")
            # Defence in depth: drop a foreign-channel reply (the gateway
            # already routes by channel). Empty = degenerate, allowed.
            if channel and channel != "terminal":
                return
            await self._queue.put(body)
        elif env.kind == KIND_ERROR:
            body = env.body if isinstance(env.body, dict) else {}
            await self._queue.put(
                {
                    "channel": "terminal",
                    "content": f"gateway error: {body.get('message') or body.get('code')}",
                    "metadata": {"kind": "diagnostic_error"},
                }
            )
            await self._queue.put(None)  # fatal — end the stream
        # ping/pong are handled inside WireClient.

    async def connect(self) -> None:
        await self._client.connect()

    async def send_inbound(self, body: dict[str, Any]) -> None:
        scenario = None if self._first_sent else self._scenario
        self._first_sent = True
        await self._client.send_inbound(
            body,
            session_key=self._session_key,
            scenario=scenario,
            env_id=f"in-{int(time.time() * 1_000_000)}",
        )

    async def outbound(self) -> AsyncIterator[dict[str, Any]]:
        """Yield outbound bodies until the connection closes."""
        while True:
            body = await self._queue.get()
            if body is None:
                return
            yield body

    async def close(self) -> None:
        await self._client.close()
        await self._queue.put(None)


__all__ = ["TerminalClient"]
