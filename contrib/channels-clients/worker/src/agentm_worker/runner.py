"""Worker run loop.

Connects to an ``agentm-gateway --bind unix://…``, advertises a list of
scenarios as ``capabilities.scenarios`` at hello, then on every
forwarded inbound envelope drives the same code paths the gateway used
to run in-process. Internally we instantiate a private
:class:`MessageBus` + :class:`Gateway` so slash-command dispatch,
approval bridge, turn-complete signalling, and session resumption all
keep working unchanged — the worker is a *thin shim* on top of the
existing harness.

Outbound from the local bus is intercepted by a consumer task and
re-emitted as ``KIND_OUTBOUND`` envelopes on the wire. The gateway
routes those back to the chat client.

Concurrency: a single ``asyncio.Semaphore`` caps in-flight
``session.prompt`` calls (``--max-concurrency``). The cap is a guard
against runaway concurrent LLM traffic from one busy chat surface; the
per-session-key serialization that the gateway's per-route lock
already provides is what keeps a single conversation's turns ordered.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agentm_channels.bus import (
    InboundMessage,
    MessageBus,
    OutboundKind,
    OutboundMessage,
)
from agentm_channels.client import WireClient
from agentm_channels.gateway import Gateway, GatewayConfig, SessionFactory
from agentm_channels.wire import (
    KIND_INBOUND,
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
)

log = logging.getLogger("agentm_worker.runner")


class WorkerRunner:
    """End-to-end glue: wire ↔ local MessageBus ↔ in-process Gateway."""

    def __init__(
        self,
        *,
        client: WireClient,
        cwd: str,
        scenario: str | None,
        session_factory: SessionFactory,
        max_concurrency: int = 4,
    ) -> None:
        self._client = client
        self._cwd = cwd
        self._scenario = scenario
        self._session_factory = session_factory
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._bus = MessageBus()
        # The Gateway is happy with no command_registry; it discovers
        # one from cwd. That is the same behaviour the in-process path
        # had before the split.
        self._gateway = Gateway(
            bus=self._bus,
            config=GatewayConfig(cwd=cwd, scenario=scenario),
            session_factory=session_factory,
        )
        self._outbound_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    # -- lifecycle ---------------------------------------------------

    async def start(self) -> None:
        await self._gateway.start()
        self._outbound_task = asyncio.create_task(
            self._consume_outbound(), name="worker-outbound"
        )

    async def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        if self._outbound_task is not None:
            self._outbound_task.cancel()
            try:
                await self._outbound_task
            except (asyncio.CancelledError, Exception):
                pass
        # Gateway.stop() iterates routes and calls session.shutdown()
        # for each — that is also our per-chat session teardown.
        await self._gateway.stop()

    # -- inbound: wire → bus -----------------------------------------

    async def handle_inbound_envelope(self, env: Envelope) -> None:
        """Translate a ``KIND_INBOUND`` envelope to an :class:`InboundMessage`
        on the local bus.

        The gateway's worker-routing path forwards inbounds as
        ``KIND_INBOUND``; the worker treats them exactly like a fresh
        user turn. Concurrency is bounded by ``--max-concurrency``.
        """
        if env.kind != KIND_INBOUND:
            return
        body = env.body if isinstance(env.body, dict) else {}
        channel_name = str(body.get("channel") or "")
        chat_id = str(body.get("chat_id") or "")
        if not channel_name or not chat_id:
            log.warning(
                "worker dropped malformed inbound id=%s (missing channel/chat_id)",
                env.id,
            )
            return
        sender_id = str(body.get("sender_id") or "")
        content = str(body.get("content") or "")
        button_value_raw = body.get("button_value")
        button_value = (
            str(button_value_raw) if button_value_raw is not None else None
        )
        async with self._semaphore:
            await self._bus.publish_inbound(
                InboundMessage(
                    channel=channel_name,
                    sender_id=sender_id,
                    chat_id=chat_id,
                    content=content,
                    button_value=button_value,
                )
            )

    # -- outbound: bus → wire ----------------------------------------

    async def _consume_outbound(self) -> None:
        """Drain the local bus and ship each message as a wire envelope."""
        try:
            while not self._stopped.is_set():
                msg: OutboundMessage = await self._bus.consume_outbound()
                env = _outbound_to_envelope(msg)
                try:
                    await self._client.send(env)
                except Exception:
                    log.exception("worker failed to forward outbound to gateway")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker outbound consumer crashed")


def _outbound_to_envelope(msg: OutboundMessage) -> Envelope:
    body: dict[str, Any] = {
        "channel": msg.channel,
        "chat_id": msg.chat_id,
        "content": msg.content,
        "kind": msg.kind.value
        if isinstance(msg.kind, OutboundKind)
        else str(msg.kind),
    }
    if msg.buttons:
        body["buttons"] = [
            {"label": b.label, "value": b.value, "style": b.style}
            for b in msg.buttons
        ]
    if msg.metadata:
        body["metadata"] = dict(msg.metadata)
    return Envelope(
        v=WIRE_VERSION,
        id=f"wkr-out-{int(time.time() * 1_000_000)}",
        kind=KIND_OUTBOUND,
        ts=time.time(),
        body=body,
    )


__all__ = ["WorkerRunner"]
