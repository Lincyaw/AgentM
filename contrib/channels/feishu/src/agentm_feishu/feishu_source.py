"""Production :class:`ChatSource` backed by ``lark_oapi.channel.FeishuChannel``.

Imports from ``lark_oapi`` are deferred to ``connect`` time so the rest
of this package (gateway core, stub source, tests) loads without the
optional ``[feishu]`` dependency present.

Wire mapping
============

- ``FeishuChannel.on("message")`` → :class:`InboundMessage` queue.
- ``FeishuChannel.on("cardAction")`` → :class:`CardActionEvent` queue.
- :meth:`send_text` → ``FeishuChannel.send(chat_id, {"text": ...})``.
- :meth:`send_card` → ``FeishuChannel.send(chat_id, {"card": {...}})``;
  the returned ``message_id`` doubles as our ``card_id`` so subsequent
  :meth:`update_card` calls map directly to ``FeishuChannel.update_card``.
- :meth:`update_card` → ``FeishuChannel.update_card(message_id, card)``.

The card-button → ``card_id`` round-trip relies on each button's
``value.card_id`` being set when the card is constructed (see
:mod:`agentm_feishu.cards`); ``cardAction`` callbacks read the same
field back so the mapping is symmetric and stateless from Feishu's
side.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .chat_source import CardActionEvent, ChatSource, InboundMessage, SendResult


if TYPE_CHECKING:  # pragma: no cover — typing only.
    from lark_oapi.channel import FeishuChannel as _FeishuChannel  # noqa: F401


logger = logging.getLogger(__name__)


class FeishuChatSource(ChatSource):
    """Adapter from :mod:`lark_oapi.channel` into the gateway's seam."""

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        bot_open_id: str | None = None,
        ignore_self_messages: bool = True,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._bot_open_id = bot_open_id
        self._ignore_self_messages = ignore_self_messages
        self._channel: Any = None
        self._messages: asyncio.Queue[InboundMessage | None] = asyncio.Queue()
        self._actions: asyncio.Queue[CardActionEvent | None] = asyncio.Queue()
        self._connect_task: asyncio.Task[Any] | None = None
        self._closed = False

    async def connect(self) -> None:
        if self._channel is not None:
            return
        try:
            from lark_oapi.channel import FeishuChannel
        except ImportError as exc:  # pragma: no cover — surfaced as a friendly error.
            raise RuntimeError(
                "lark_oapi.channel is not installed. Install with the "
                "[feishu] extra: `pip install agentm-feishu-channel[feishu]`."
            ) from exc

        channel = FeishuChannel(app_id=self._app_id, app_secret=self._app_secret)
        self._channel = channel

        @channel.on("message")
        async def _on_message(msg: Any) -> None:
            if self._ignore_self_messages and self._is_self(msg):
                return
            inbound = InboundMessage(
                chat_id=getattr(msg, "chat_id", "") or "",
                user_id=getattr(msg, "sender_id", "") or "",
                text=getattr(msg, "content_text", "") or "",
                thread_id=self._extract_thread_id(msg),
                message_id=getattr(msg, "message_id", None),
                raw={},  # Don't capture full raw — too large; rely on log on error.
            )
            await self._messages.put(inbound)

        @channel.on("cardAction")
        async def _on_card_action(event: Any) -> None:
            value = getattr(event.action, "value", None) or {}
            card_id = self._resolve_card_id(value, event)
            if not card_id:
                return
            action = self._resolve_action_name(value, event)
            user = getattr(event.operator, "open_id", None) or ""
            await self._actions.put(
                CardActionEvent(
                    card_id=card_id,
                    user_id=user,
                    action=action,
                    value=value if isinstance(value, dict) else {"raw": value},
                )
            )

        # FeishuChannel.connect() blocks for the lifetime of the connection;
        # run it as a background task and signal close by closing the queue.
        self._connect_task = asyncio.create_task(channel.connect(), name="feishu-channel")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._connect_task is not None:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._messages.put(None)
        await self._actions.put(None)

    async def messages(self) -> AsyncIterator[InboundMessage]:
        while True:
            item = await self._messages.get()
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
        opts = self._build_opts(thread_id=thread_id, reply_to=reply_to)
        result = await self._require_channel().send(
            chat_id, {"text": text}, opts=opts
        )
        return SendResult(message_id=getattr(result, "message_id", None))

    async def send_card(
        self,
        chat_id: str,
        card: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> SendResult:
        opts = self._build_opts(thread_id=thread_id)
        result = await self._require_channel().send(
            chat_id, {"card": card}, opts=opts
        )
        message_id = getattr(result, "message_id", None)
        return SendResult(card_id=message_id, message_id=message_id)

    async def update_card(self, card_id: str, card: dict[str, Any]) -> SendResult:
        result = await self._require_channel().update_card(card_id, card)
        return SendResult(card_id=card_id, message_id=getattr(result, "message_id", None))

    # -------- helpers ----------------------------------------------------

    def _require_channel(self) -> Any:
        if self._channel is None:
            raise RuntimeError("FeishuChatSource.connect() has not been called")
        return self._channel

    def _build_opts(
        self, *, thread_id: str | None = None, reply_to: str | None = None
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        if thread_id:
            # Reply-in-thread carries the parent message_id; the channel's
            # coerce_send_opts maps this onto the wire-level fields.
            opts["reply_in_thread"] = True
            if not reply_to:
                opts["reply_to"] = thread_id
        if reply_to:
            opts["reply_to"] = reply_to
        return opts

    def _is_self(self, msg: Any) -> bool:
        if not self._bot_open_id:
            return False
        sender = getattr(msg, "sender_id", None)
        return sender == self._bot_open_id

    @staticmethod
    def _extract_thread_id(msg: Any) -> str | None:
        # lark_oapi.channel surfaces thread metadata via reply / thread_id
        # depending on the wire shape — try both.
        thread_id = getattr(msg, "thread_id", None)
        if thread_id:
            return str(thread_id)
        reply = getattr(msg, "reply", None)
        if reply is not None:
            tid = getattr(reply, "thread_id", None) or getattr(reply, "message_id", None)
            return str(tid) if tid else None
        return None

    @staticmethod
    def _resolve_card_id(value: Any, event: Any) -> str | None:
        if isinstance(value, dict) and "card_id" in value:
            return str(value["card_id"])
        # Fall back to the source message id — still usable as a key, just
        # only matches if the bridge keys on it.
        return getattr(event, "message_id", None)

    @staticmethod
    def _resolve_action_name(value: Any, event: Any) -> str:
        if isinstance(value, dict) and "action" in value:
            return str(value["action"])
        name = getattr(event.action, "name", None)
        return str(name) if name else "unknown"
