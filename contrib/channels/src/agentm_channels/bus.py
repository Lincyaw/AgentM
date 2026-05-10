"""Channel-agnostic message bus.

Two ``asyncio.Queue``s — one inbound (channel → agent), one outbound
(agent → channel). Channels never see ``AgentSession``; the gateway
never sees a Feishu / Slack object. Both ends only know
:class:`InboundMessage` / :class:`OutboundMessage`.

The shape mirrors HKUDS/nanobot's ``nanobot.bus`` so adding a channel
is a drop-in: implement :class:`agentm_channels.base.BaseChannel`,
register it via auto-discovery, and the rest of the system handles
session routing, approval, and outbound dispatch unchanged.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class InboundMessage:
    """A message a channel received from a user."""

    channel: str
    """Channel module name (``"feishu"`` / ``"slack"`` / …). The
    gateway uses this to route the agent's reply back to the same
    channel."""

    sender_id: str
    """Stable user identifier within ``channel`` (e.g. Feishu open_id)."""

    chat_id: str
    """Conversation identifier within ``channel`` (DM / group / channel)."""

    content: str
    """Plain-text message body. Channels are responsible for flattening
    rich content (markdown, posts, mentions) into text before publishing."""

    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    """Channel-specific extras (Feishu thread_id, Slack ts, etc.).
    Opaque to the gateway except for keys it explicitly documents."""

    session_key_override: str | None = None
    """Set when a channel scopes sessions tighter than ``chat_id``
    (e.g. one session per Feishu thread). Defaults to
    ``f"{channel}:{chat_id}"``."""

    @property
    def session_key(self) -> str:
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass(slots=True)
class OutboundMessage:
    """A message the agent wants delivered through a channel.

    Buttons are the abstraction for human-in-the-loop UI. Each button
    is ``[label, value]``; the channel chooses how to render them
    (Feishu interactive card, Slack action blocks, Telegram inline
    keyboard, plaintext numbered list as last resort). When the user
    interacts with a button, the channel publishes an
    :class:`InboundMessage` whose ``metadata`` carries
    ``{"button_value": <value>, "in_reply_to": <chat_id_of_card>}``
    — the approval bridge keys on those.
    """

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    buttons: list[list[str]] = field(default_factory=list)


class MessageBus:
    """Two-queue bus connecting channels and the gateway.

    The gateway owns the bus; channels and the gateway loop are
    consumers. Both queues are unbounded — backpressure is the channel
    implementation's responsibility (a slow Feishu user shouldn't be
    able to drop Slack traffic).

    Cross-loop publish is supported. ``lark_oapi.channel.FeishuChannel``
    dispatches inbound events on its own background loop (running in a
    separate thread); the queue, however, is created on the gateway's
    main loop. ``await queue.put()`` from a foreign loop *appears* to
    succeed but the consumer's waiter on the home loop never wakes —
    silent message loss / hang. ``publish_inbound`` and
    ``publish_outbound`` therefore detect a foreign caller loop and
    bridge through ``loop.call_soon_threadsafe`` onto the home loop.
    """

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        try:
            self._home_loop: asyncio.AbstractEventLoop | None = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            # Allow construction outside an async context (test setup);
            # the first publish/consume from the home loop will retry.
            self._home_loop = None

    def _ensure_home_loop(self) -> asyncio.AbstractEventLoop:
        if self._home_loop is None:
            self._home_loop = asyncio.get_running_loop()
        return self._home_loop

    async def _publish(self, queue: asyncio.Queue[Any], msg: Any) -> None:
        home = self._ensure_home_loop()
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is None or current is home:
            await queue.put(msg)
            return
        # Foreign loop: schedule the put on the home loop. Unbounded
        # queue → ``put_nowait`` never blocks or raises; we don't need
        # to await its completion (the queue's consumer will pick it up
        # whenever the home loop next runs).
        home.call_soon_threadsafe(queue.put_nowait, msg)

    async def publish_inbound(self, msg: InboundMessage) -> None:
        await self._publish(self.inbound, msg)

    async def consume_inbound(self) -> InboundMessage:
        self._ensure_home_loop()
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        await self._publish(self.outbound, msg)

    async def consume_outbound(self) -> OutboundMessage:
        self._ensure_home_loop()
        return await self.outbound.get()
