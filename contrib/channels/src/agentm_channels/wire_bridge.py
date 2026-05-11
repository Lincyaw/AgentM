"""Bridge between :class:`WireServer` and the in-process v0 :class:`Gateway`.

The chosen extension point is **synthetic-channel registration**: each
connected wire peer (identified by the ``channel`` field on its first
inbound envelope) is injected into the running :class:`ChannelManager`
as a :class:`_WireChannel`, a :class:`BaseChannel` whose ``send()``
enqueues the outbound onto the peer's outbox. This reuses the existing
inbound/outbound MessageBus path unchanged: gateway dispatch logic, slash
commands, approvals, turn-complete signalling, channel allow-from — all
of it still works, with no new conditional code in :class:`Gateway`.

Rationale (vs. calling ``gateway._dispatch`` directly): going through
``MessageBus`` keeps a single dispatch contract for v0 and wire peers,
so observability / approval / retry semantics are identical. The cost
is one ``BaseChannel`` instance per wire peer — cheap and short-lived.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .base import BaseChannel
from .bus import MessageBus, OutboundKind, OutboundMessage
from .manager import ChannelManager
from .outbox import OutboxStore
from .peer import PeerSession
from .wire import KIND_INBOUND, KIND_OUTBOUND, WIRE_VERSION, Envelope

log = logging.getLogger("agentm_channels.wire_bridge")


class _WireChannel(BaseChannel):
    """Outbound sink for one wire peer.

    Not auto-discovered; instantiated and injected directly into
    :class:`ChannelManager.channels` by :class:`WireBridge` on hello.
    """

    name = "_wire"
    display_name = "Wire peer"
    # Synthetic bridge channel — not a v0 in-process platform adapter,
    # so the BaseChannel deprecation warning does not apply.
    _is_stub_fixture = True

    def __init__(
        self,
        config: dict[str, Any],
        bus: MessageBus,
        *,
        peer_id: str,
        channel_name: str,
        outbox: OutboxStore,
    ) -> None:
        super().__init__(config, bus)
        self._peer_id = peer_id
        self.name = channel_name  # per-instance: the manager keys on this
        self._outbox = outbox
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._running = True
        await self._stopped.wait()

    async def stop(self) -> None:
        self._running = False
        self._stopped.set()

    async def send(self, msg: OutboundMessage) -> None:
        # Turn-complete is internal control; chat clients want it as
        # an envelope too, but for v1 we ship it on the wire so peers
        # can mirror it (terminal channel uses it to print a marker).
        body: dict[str, Any] = {
            "channel": msg.channel,
            "chat_id": msg.chat_id,
            "content": msg.content,
            "kind": msg.kind.value
            if isinstance(msg.kind, OutboundKind)
            else str(msg.kind),
        }
        # Buttons round-trip the typed shape so terminal/Feishu clients
        # can render their native UI and the approval bridge's
        # button_value contract survives across the wire.
        if msg.buttons:
            body["buttons"] = [
                {"label": b.label, "value": b.value, "style": b.style}
                for b in msg.buttons
            ]
        # Pass through metadata when present — the approval bridge tags
        # approval_request/approval_resolved with correlation info there,
        # and the wire is the only path now that v0 channels are
        # deprecated.
        if msg.metadata:
            body["metadata"] = dict(msg.metadata)
        env = Envelope(
            v=WIRE_VERSION,
            id=f"out-{self._peer_id}-{int(time.time() * 1_000_000)}",
            kind=KIND_OUTBOUND,
            ts=time.time(),
            body=body,
        )
        await asyncio.to_thread(self._outbox.enqueue, self._peer_id, env)


class WireBridge:
    """Glue object: turns wire inbound envelopes into v0 inbound messages.

    Holds the :class:`ChannelManager` (so it can inject/remove synthetic
    :class:`_WireChannel` instances on the fly) and the
    :class:`OutboxStore` (so the synthetic channel can enqueue replies).
    """

    def __init__(
        self, *, bus: MessageBus, manager: ChannelManager, outbox: OutboxStore
    ) -> None:
        self._bus = bus
        self._manager = manager
        self._outbox = outbox
        # peer_id → channel_name registered for that peer (so we can
        # tear it down on disconnect).
        self._peer_channels: dict[str, str] = {}

    async def handle_inbound(self, peer_session: PeerSession, env: Envelope) -> None:
        """Invoked by :class:`WireServer` for each ``inbound`` envelope."""
        if env.kind != KIND_INBOUND:
            return  # ping/pong/bye are handled by the server already
        body = env.body if isinstance(env.body, dict) else {}
        channel_name = str(body.get("channel") or "")
        chat_id = str(body.get("chat_id") or "")
        sender_id = str(body.get("sender_id") or peer_session.peer_id)
        content = str(body.get("content") or "")
        button_value_raw = body.get("button_value")
        button_value = (
            str(button_value_raw) if button_value_raw is not None else None
        )
        if not channel_name or not chat_id:
            log.warning(
                "wire inbound rejected: missing channel/chat_id (peer=%s id=%s)",
                peer_session.peer_id,
                env.id,
            )
            return
        await self._ensure_channel(peer_session.peer_id, channel_name)
        from .bus import InboundMessage

        await self._bus.publish_inbound(
            InboundMessage(
                channel=channel_name,
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                button_value=button_value,
            )
        )

    async def _ensure_channel(self, peer_id: str, channel_name: str) -> None:
        existing = self._peer_channels.get(peer_id)
        if existing == channel_name:
            return
        if existing is not None:
            # Peer renamed; drop the old synthetic channel first.
            await self._drop_channel(existing)
        # Inject into ChannelManager so its dispatch loop routes
        # outbounds with this channel name through us. Bypass
        # allow_from enforcement — the wire peer authenticated via
        # peer-cred; per-sender ACL is deferred (§6.3).
        ch = _WireChannel(
            {"enabled": True, "allow_from": ["*"]},
            self._bus,
            peer_id=peer_id,
            channel_name=channel_name,
            outbox=self._outbox,
        )
        self._manager.inject_channel(channel_name, ch)
        self._peer_channels[peer_id] = channel_name
        log.info(
            "wire peer registered as channel %r (peer_id=%s)", channel_name, peer_id
        )

    async def _drop_channel(self, channel_name: str) -> None:
        ch = self._manager.channels.pop(channel_name, None)
        if ch is None:
            return
        try:
            await ch.stop()
        except Exception:  # noqa: BLE001
            log.exception("wire channel %r stop raised", channel_name)

    async def on_peer_disconnect(self, peer_id: str) -> None:
        channel = self._peer_channels.pop(peer_id, None)
        if channel is None:
            return
        await self._drop_channel(channel)
        log.info("wire peer disconnected: channel %r dropped", channel)


__all__ = ["WireBridge", "_WireChannel"]
