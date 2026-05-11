"""Feishu / Lark channel.

Wraps :class:`lark_oapi.channel.FeishuChannel` behind the
:class:`agentm_channels.base.BaseChannel` interface. Everything Feishu-
specific (lark_oapi import, WSS quirks, card schema 2.0, button value
round-trip) lives in this one file — the gateway, manager, and other
channels stay oblivious.

Inbound mapping:

- ``message`` event → :meth:`_handle_message` with the flattened text.
  ``thread_id`` (if present) becomes the ``session_key`` so a
  multi-topic group chat doesn't collapse all threads into one
  AgentSession.
- ``cardAction`` event → an :class:`InboundMessage` whose
  ``button_value`` carries the typed value the original
  :class:`agentm_channels.bus.Button` was sent with. The gateway hands
  every such inbound to the approval bridge.

Outbound mapping:

- :attr:`OutboundKind.MESSAGE` → Feishu interactive card with one
  schema-2.0 markdown element (so Chinese, code fences, lists, etc.
  all render). When ``buttons`` is non-empty an action block is
  appended; each :class:`Button.style` maps to a Feishu button type.
- :attr:`OutboundKind.TURN_COMPLETE` → tear down ACK reactions; nothing
  goes on the wire.

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
import warnings
from typing import Any

from ..base import BaseChannel
from ..bus import Button, ButtonStyle, OutboundKind, OutboundMessage


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

    TODO: drop this once an upstream fix lands. File the issue on
    https://github.com/larksuite/oapi-sdk-python with a reproducer and
    delete this helper plus its caller in :meth:`FeishuChannel.start`.
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
        # Pending ACK reactions keyed by chat_id. Each entry is the task
        # that's currently adding (or has added) a reaction to a recent
        # inbound message; when the gateway signals turn_complete for
        # the chat, we await the task and remove the reaction so the
        # visual "received" indicator only sticks around while the
        # agent is actually thinking.
        self._pending_acks: dict[str, list[asyncio.Task[Any]]] = {}

    # -- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        warnings.warn(
            "agentm_channels.channels.feishu.FeishuChannel is deprecated as "
            "of Phase 3 of the gateway/client split. Run the Feishu adapter "
            "as a separate process: "
            "`agentm-feishu --connect unix:///path/to/gateway.sock --app-id ... "
            "--app-secret /path/to/secret-file`. The in-process driver "
            "remains functional but will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        # LLM has had a chance to reply. Best-effort, fire-and-forget —
        # we MUST NOT await it here: if the SDK's reaction RPC hangs or
        # holds a lock, awaiting blocks the inbound handler and the
        # message never reaches the agent.
        if message_id and self._ack_emoji():
            task = asyncio.create_task(
                self._safe_reaction(message_id, self._ack_emoji()),
                name="feishu-ack",
            )
            self._pending_acks.setdefault(chat_id, []).append(task)
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

    async def _safe_reaction(
        self, message_id: str, emoji: str
    ) -> tuple[str, str] | None:
        """Add an ACK reaction; return ``(message_id, reaction_id)`` on
        success so the turn-complete cleanup can remove it.
        """
        try:
            result = await self._channel.add_reaction(message_id, emoji)
        except Exception:
            logger.exception("[feishu] add_reaction failed for %s", message_id)
            return None
        raw = getattr(result, "raw", None) or {}
        data = raw.get("data") if isinstance(raw, dict) else None
        reaction_id: str | None = None
        if isinstance(data, dict):
            # Feishu's response shape: {"data": {"reaction_id": "..."}}.
            inner = data.get("reaction_id")
            if isinstance(inner, str):
                reaction_id = inner
            else:
                # Some SDK versions wrap one level deeper under "reaction".
                reaction = data.get("reaction")
                if isinstance(reaction, dict):
                    rid = reaction.get("reaction_id")
                    if isinstance(rid, str):
                        reaction_id = rid
        if not reaction_id:
            logger.warning(
                "[feishu] add_reaction succeeded but reaction_id missing in %r",
                raw,
            )
            return None
        return message_id, reaction_id

    async def _clear_pending_acks(self, chat_id: str) -> None:
        """Remove every ACK reaction we attached for ``chat_id``.

        Called when the gateway signals turn_complete: agent has
        finished replying so the "received" emoji should disappear.
        Awaits each in-flight add task before deleting — otherwise the
        delete races the add and the reaction lingers.
        """
        tasks = self._pending_acks.pop(chat_id, None)
        if not tasks:
            return
        for task in tasks:
            try:
                pair = await task
            except Exception:
                logger.exception("[feishu] ack-add task raised")
                continue
            if not pair:
                continue
            message_id, reaction_id = pair
            try:
                await self._channel.remove_reaction(message_id, reaction_id)
            except Exception:
                logger.exception(
                    "[feishu] remove_reaction failed (msg=%s reaction=%s)",
                    message_id,
                    reaction_id,
                )

    async def _on_card_action(self, event: Any) -> None:
        value = getattr(event.action, "value", None) or {}
        button_value: str | None = None
        if isinstance(value, dict):
            raw = value.get("button_value")
            if raw is not None:
                button_value = str(raw)
        if not button_value:
            return
        operator = getattr(event, "operator", None)
        sender = getattr(operator, "open_id", None) or getattr(operator, "user_id", None) or ""
        chat_id = getattr(event, "chat_id", "") or ""
        await self._handle_message(
            sender_id=str(sender),
            chat_id=chat_id,
            # The gateway routes on ``button_value``; ``content`` is
            # informational only (shows up in observability traces).
            content=f"[card click: {button_value}]",
            button_value=button_value,
            metadata={
                "feishu_card_message_id": getattr(event, "message_id", None),
            },
        )

    # -- outbound ------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        if self._channel is None:
            raise RuntimeError("FeishuChannel.start() has not completed")
        if msg.kind is OutboundKind.TURN_COMPLETE:
            # Control signal — clear ACK reactions, nothing on the wire.
            await self._clear_pending_acks(msg.chat_id)
            return
        # Render every chat message as a schema-2.0 interactive card so
        # Chinese, markdown (code fences, lists, headings) and action
        # buttons all use the same path. Lark's plain ``text`` message
        # type does not render markdown.
        await self._channel.send(
            msg.chat_id,
            {"card": _markdown_card(msg.content, buttons=msg.buttons)},
        )

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


_BUTTON_STYLE_MAP: dict[ButtonStyle, str] = {
    "primary": "primary",
    "danger": "danger",
    "default": "default",
}


def _markdown_card(text: str, *, buttons: list[Button]) -> dict[str, Any]:
    """Construct a Lark schema-2.0 interactive card.

    The body is one ``markdown`` element so Chinese, code fences, lists,
    and inline formatting all render. When ``buttons`` is non-empty an
    ``action`` block is appended; each :class:`Button.value` round-trips
    through Lark's ``cardAction`` callback as the typed inbound
    ``button_value``. Style mapping is mechanical — no label-string
    heuristics — so the caller fully controls visual emphasis via
    :attr:`Button.style`.
    """

    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": text or "*(empty)*"}
    ]
    if buttons:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": btn.label},
                        "type": _BUTTON_STYLE_MAP.get(btn.style, "default"),
                        "name": btn.label.lower(),
                        "value": {"button_value": btn.value},
                    }
                    for btn in buttons
                ],
            }
        )
    return {"schema": "2.0", "body": {"elements": elements}}
