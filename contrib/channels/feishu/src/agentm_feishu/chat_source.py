"""ChatSource — the abstract transport seam.

A :class:`ChatSource` is everything the gateway needs from "the place
users talk to the agent". The production implementation
(:class:`~agentm_feishu.feishu_source.FeishuChatSource`) bridges
``lark_oapi.channel.FeishuChannel``; the test implementation
(:class:`StubChatSource`) is a pure in-memory queue. The gateway only
sees this interface.

Design notes:

- The seam is **transport-shaped, not Feishu-shaped**: card payloads
  are typed dicts, identities are opaque strings. A future Slack /
  Discord adapter would implement the same protocol.
- Inbound events are delivered through ``async for msg in source``.
  The source owns the connection lifecycle (``connect`` / ``close``);
  the gateway only pushes messages and drains events.
- ``send_card`` returns a stable ``card_id`` the caller can pass to
  ``update_card`` for streaming updates and approval-resolution flows.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """A single user-facing message from the chat surface.

    ``chat_id`` is the routing key the gateway uses to look up an
    :class:`agentm.harness.AgentSession`. ``thread_id`` (if set) carries
    the Feishu thread / Slack thread_ts so the gateway can scope a
    session to a thread instead of a chat — this is how multi-topic
    group chats avoid stomping each other's session state.

    ``user_id`` identifies the human who sent the message; the gateway
    uses it to attribute approvals (only the original requester can
    approve their own tool call) and to inject identity into the
    session config (e.g. ``--as user`` for downstream lark-cli calls).
    """

    chat_id: str
    user_id: str
    text: str
    thread_id: str | None = None
    message_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CardActionEvent:
    """A button click / form submit on a card the gateway sent earlier.

    ``card_id`` matches the value returned from
    :meth:`ChatSource.send_card`. ``action`` is the button name
    (e.g. ``"approve"`` / ``"deny"``). ``user_id`` is who clicked.
    """

    card_id: str
    user_id: str
    action: str
    value: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SendResult:
    """Identifier the chat surface returns after a successful send."""

    card_id: str | None = None
    message_id: str | None = None


class ChatSource(Protocol):
    """Abstract transport seam.

    Implementations MUST be safe to drive from a single asyncio task —
    inbound events are funnelled through one queue; outbound calls may
    be concurrent (the gateway fans them out).
    """

    async def connect(self) -> None:
        """Establish the connection. Idempotent. Must be called before iteration."""
        ...

    async def close(self) -> None:
        """Tear down the connection. Idempotent."""
        ...

    def messages(self) -> AsyncIterator[InboundMessage]:
        """Async iterator over inbound user messages.

        The gateway consumes this once. Implementations end the iterator
        when the underlying connection is permanently closed.
        """
        ...

    def card_actions(self) -> AsyncIterator[CardActionEvent]:
        """Async iterator over button / form submissions on cards we sent."""
        ...

    async def send_text(
        self,
        chat_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        reply_to: str | None = None,
    ) -> SendResult:
        """Post a plain-text message into a chat / thread."""
        ...

    async def send_card(
        self,
        chat_id: str,
        card: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> SendResult:
        """Post an interactive card; returns a handle suitable for ``update_card``."""
        ...

    async def update_card(self, card_id: str, card: dict[str, Any]) -> SendResult:
        """Replace the body of a previously-sent card.

        Used for two flows: streaming assistant text into a single
        card (instead of spamming N messages), and resolving an
        approval card after the user clicks a button.
        """
        ...


class StubChatSource:
    """In-memory ChatSource for tests and local dry-runs.

    Inbound messages are pushed via :meth:`push_message` /
    :meth:`push_card_action`. Outbound sends are recorded in
    :attr:`outbox` for assertions; ``card_id`` is monotonically
    increasing so tests can match a ``send_card`` to a later
    ``update_card``.

    Conforms structurally to :class:`ChatSource`.
    """

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[InboundMessage | None] = asyncio.Queue()
        self._actions: asyncio.Queue[CardActionEvent | None] = asyncio.Queue()
        self.outbox: list[dict[str, Any]] = []
        self._next_card_id = 0
        self._connected = False
        self._closed = False

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._inbox.put(None)
        await self._actions.put(None)

    def push_message(self, msg: InboundMessage) -> None:
        """Test helper — enqueue an inbound user message."""
        self._inbox.put_nowait(msg)

    def push_card_action(self, action: CardActionEvent) -> None:
        """Test helper — simulate a button click on a card we sent."""
        self._actions.put_nowait(action)

    async def messages(self) -> AsyncIterator[InboundMessage]:
        while True:
            item = await self._inbox.get()
            if item is None:
                return
            yield item

    async def card_actions(self) -> AsyncIterator[CardActionEvent]:
        while True:
            item = await self._actions.get()
            if item is None:
                return
            yield item

    async def send_text(
        self,
        chat_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        reply_to: str | None = None,
    ) -> SendResult:
        message_id = f"msg-{len(self.outbox)}"
        self.outbox.append(
            {
                "kind": "text",
                "chat_id": chat_id,
                "thread_id": thread_id,
                "reply_to": reply_to,
                "text": text,
                "message_id": message_id,
            }
        )
        return SendResult(message_id=message_id)

    async def send_card(
        self,
        chat_id: str,
        card: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> SendResult:
        card_id = f"card-{self._next_card_id}"
        self._next_card_id += 1
        self.outbox.append(
            {
                "kind": "card",
                "chat_id": chat_id,
                "thread_id": thread_id,
                "card": card,
                "card_id": card_id,
            }
        )
        return SendResult(card_id=card_id)

    async def update_card(self, card_id: str, card: dict[str, Any]) -> SendResult:
        self.outbox.append({"kind": "update_card", "card_id": card_id, "card": card})
        return SendResult(card_id=card_id)
