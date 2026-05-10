"""Feishu / Lark channel.

Wraps :class:`lark_oapi.channel.FeishuChannel` behind the
:class:`agentm_channels.base.BaseChannel` interface. Everything Feishu-
specific (lark_oapi import, WSS quirks, card schema, button-value
encoding) lives in this one file — the gateway, manager, and other
channels stay oblivious.

Inbound mapping:

- ``message`` event → :meth:`_handle_message` with the flattened text.
  ``thread_id`` (if present) becomes the ``session_key`` so a
  multi-topic group chat doesn't collapse all threads into one
  AgentSession.
- ``cardAction`` event → an :class:`InboundMessage` whose
  ``metadata`` carries ``{"button_value": "<approval_id>:approve|deny"}``.
  The gateway routes those to the approval bridge.

Outbound mapping:

- :class:`OutboundMessage` with no ``buttons`` → plain Feishu text.
- :class:`OutboundMessage` with ``buttons`` → Feishu interactive card
  whose buttons carry ``value={"button_value": <pair[1]>}``. The
  ``cardAction`` callback echoes that back, closing the loop.

Config (under ``channels.feishu`` in the gateway YAML)::

    channels:
      feishu:
        enabled: true
        app_id: cli_xxxx
        app_secret: xxxxxxxx
        allow_from: ['*']            # or specific open_ids
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..base import BaseChannel
from ..bus import OutboundMessage


logger = logging.getLogger(__name__)


def _patch_ws_loop_once() -> None:
    """Workaround for an upstream lark-oapi bug (>= 1.6.2).

    ``lark_oapi.ws.client`` captures ``asyncio.get_event_loop()`` at
    module-import time. When imported from inside an already-running
    event loop (which is exactly our case — the gateway runs under
    ``asyncio.run``), that captured reference *is* the main loop;
    ``WSClient.start`` then calls ``loop.run_until_complete(...)`` from
    a background thread against the still-running main loop and raises
    ``RuntimeError: This event loop is already running``. Replacing the
    module-level reference with a fresh loop dedicated to the WS client
    makes ``run_until_complete`` legal again — the WS client is the
    only consumer of that module global.
    """
    import lark_oapi.ws.client as _ws_client

    _ws_client.loop = asyncio.new_event_loop()


class FeishuChannel(BaseChannel):
    name = "feishu"
    display_name = "Feishu / Lark"

    def __init__(self, config: Any, bus: Any) -> None:
        super().__init__(config, bus)
        self._channel: Any = None
        self._connect_task: asyncio.Task[Any] | None = None

    # -- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        try:
            from lark_oapi.channel import FeishuChannel as _Lark
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "lark_oapi is not installed. Add it as a dep on the "
                "agentm-channels package or your gateway image."
            ) from exc

        cfg = self.config
        app_id = cfg.get("app_id") if isinstance(cfg, dict) else getattr(cfg, "app_id", None)
        app_secret = (
            cfg.get("app_secret") if isinstance(cfg, dict)
            else getattr(cfg, "app_secret", None)
        )
        if not app_id or not app_secret:
            raise RuntimeError(
                "feishu channel: app_id / app_secret missing in config"
            )

        _patch_ws_loop_once()
        channel = _Lark(app_id=app_id, app_secret=app_secret)
        self._channel = channel
        # ``FeishuChannel.on`` in lark-oapi 1.6.x requires a
        # ``(name, handler)`` call — the decorator form documented in
        # older READMEs raises at runtime.
        channel.on("message", self._on_message)
        channel.on("cardAction", self._on_card_action)
        self._connect_task = asyncio.create_task(
            channel.connect(), name="feishu-channel"
        )
        try:
            await channel.connect_until_ready(timeout=20.0)
        except Exception:
            logger.exception("feishu connect_until_ready failed")
            raise
        self._running = True
        # Stay alive until stop().
        await self._connect_task

    async def stop(self) -> None:
        if not self._running and self._channel is None:
            return
        self._running = False
        if self._channel is not None:
            try:
                await self._channel.disconnect()
            except Exception:
                logger.exception("FeishuChannel.disconnect raised")
            try:
                await self._channel.stop_background()
            except Exception:
                logger.exception("FeishuChannel.stop_background raised")
        if self._connect_task is not None and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass

    # -- inbound -------------------------------------------------------

    async def _on_message(self, msg: Any) -> None:
        sender = getattr(msg, "sender_id", "") or ""
        message_id = getattr(msg, "message_id", None)
        chat_id = getattr(msg, "chat_id", "") or ""
        content = getattr(msg, "content_text", "") or ""
        logger.info(
            "[feishu] rx chat=%s sender=%s msg_id=%s text=%r",
            chat_id,
            sender,
            message_id,
            content[:120],
        )
        if self._is_self(sender):
            return
        # Quick visual ACK so the user knows the bot got it before the
        # LLM has had a chance to reply. Best-effort — failure is logged
        # but does not block the inbound dispatch.
        if message_id:
            try:
                await self._channel.add_reaction(message_id, self._ack_emoji())
            except Exception:
                logger.exception("[feishu] add_reaction failed")
        await self._handle_message(
            sender_id=sender,
            chat_id=chat_id,
            content=content,
            session_key=self._session_key_for(msg),
            metadata={"feishu_message_id": message_id},
        )

    def _ack_emoji(self) -> str:
        cfg = self.config
        if isinstance(cfg, dict):
            return str(cfg.get("ack_emoji", "OK"))
        return getattr(cfg, "ack_emoji", "OK")

    async def _on_card_action(self, event: Any) -> None:
        value = getattr(event.action, "value", None) or {}
        button_value: str | None = None
        if isinstance(value, dict):
            button_value = value.get("button_value")
        if not button_value:
            return
        operator = getattr(event, "operator", None)
        sender = getattr(operator, "open_id", None) or getattr(operator, "user_id", None) or ""
        chat_id = getattr(event, "chat_id", "") or ""
        await self._handle_message(
            sender_id=str(sender),
            chat_id=chat_id,
            # The gateway only inspects ``metadata.button_value`` for
            # approval routing; ``content`` is informational.
            content=f"[card click: {button_value}]",
            metadata={
                "button_value": str(button_value),
                "feishu_card_message_id": getattr(event, "message_id", None),
            },
        )

    # -- outbound ------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        if self._channel is None:
            raise RuntimeError("FeishuChannel.start() has not completed")
        if msg.buttons:
            await self._channel.send(
                msg.chat_id,
                {"card": _buttons_card(msg.content, msg.buttons)},
            )
            return
        await self._channel.send(msg.chat_id, {"text": msg.content})

    # -- helpers -------------------------------------------------------

    def _is_self(self, sender_id: str) -> bool:
        bot = getattr(self._channel, "_bot", None) if self._channel else None
        bot_oid = getattr(bot, "open_id", None) if bot else None
        return bool(bot_oid and sender_id == bot_oid)

    @staticmethod
    def _session_key_for(msg: Any) -> str | None:
        thread_id = getattr(msg, "thread_id", None)
        if thread_id:
            return f"feishu:{getattr(msg, 'chat_id', '')}::{thread_id}"
        reply = getattr(msg, "reply", None)
        if reply is not None:
            tid = getattr(reply, "thread_id", None)
            if tid:
                return f"feishu:{getattr(msg, 'chat_id', '')}::{tid}"
        return None  # falls back to channel:chat_id

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "enabled": False,
            "app_id": "",
            "app_secret": "",
            "allow_from": [],
        }


def _buttons_card(text: str, buttons: list[list[str]]) -> dict[str, Any]:
    """Construct a minimal Feishu interactive card with action buttons.

    Each button's ``value.button_value`` carries the second element of
    ``[label, value]`` so the ``cardAction`` callback can round-trip
    cleanly. The first button styles as ``primary``, the rest as
    ``default`` (or ``danger`` if the label looks negative); this keeps
    the rendering predictable without forcing callers to think in
    Feishu schema.
    """
    return {
        "schema": "2.0",
        "header": {
            "title": {"tag": "plain_text", "content": "Approval"}
        },
        "body": {
            "elements": [
                {"tag": "markdown", "content": text or "*(no body)*"},
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": label},
                            "type": _button_style(label, idx),
                            "name": label.lower(),
                            "value": {"button_value": value},
                        }
                        for idx, (label, value) in enumerate(buttons)
                    ],
                },
            ]
        },
    }


def _button_style(label: str, idx: int) -> str:
    low = label.strip().lower()
    if low in {"deny", "reject", "no", "cancel"}:
        return "danger"
    if idx == 0:
        return "primary"
    return "default"
