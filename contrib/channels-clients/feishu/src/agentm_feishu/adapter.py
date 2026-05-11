"""Feishu / Lark adapter for the ``agentm-feishu`` client process.

Lifted from the legacy in-process ``FeishuChannel`` driver in
``agentm_channels.channels.feishu``. Interface differences:

* No ``BaseChannel`` parent. The adapter is a plain class that owns a
  :class:`WireClient` and pushes inbound traffic out over the wire
  instead of an in-process :class:`MessageBus`.
* Outbound delivery is no longer dispatched by a manager; the CLI's
  wire read loop calls :meth:`handle_outbound` for every
  ``KIND_OUTBOUND`` envelope from the gateway.

Card schema 2.0, approval buttons, the ACK-emoji reaction handling,
``cardAction`` → ``button_value`` round-trip, and ``thread_id`` →
``session_key`` mapping are preserved byte-for-byte.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agentm_channels.client import WireClient
from agentm_channels.wire import KIND_INBOUND, WIRE_VERSION, Envelope

log = logging.getLogger("agentm_feishu.adapter")


# Mirrors ``agentm_channels.bus.ButtonStyle`` without depending on the
# enum module — the adapter sees raw outbound bodies off the wire.
_BUTTON_STYLE_MAP: dict[str, str] = {
    "primary": "primary",
    "danger": "danger",
    "default": "default",
}


@dataclass(slots=True)
class FeishuConfig:
    """Adapter config — derived from CLI flags / env in ``cli.py``."""

    app_id: str
    app_secret: str
    allow_from: list[str] = field(default_factory=lambda: ["*"])
    chat_id_prefix: str = "feishu"
    ack_emoji: str = "OK"
    channel_name: str = "feishu"
    """Channel name announced on inbound envelopes. The gateway uses it
    as the synthetic-channel key registered through ``inject_channel``.
    """


class FeishuAdapter:
    """Bridges a ``lark_oapi`` Feishu channel to a :class:`WireClient`."""

    def __init__(self, client: WireClient, config: FeishuConfig) -> None:
        self._client = client
        self._config = config
        self._channel: Any = None
        self._connect_task: asyncio.Task[Any] | None = None
        # Pending ACK reactions keyed by chat_id — same lifetime contract
        # as the legacy implementation. Cleared on turn_complete.
        self._pending_acks: dict[str, list[asyncio.Task[Any]]] = {}
        self._running = False

    # -- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        """Connect to Feishu and stay alive until :meth:`stop`."""
        if self._running:
            return
        try:
            from lark_oapi.channel import FeishuChannel as _Lark
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "lark_oapi is not installed. Reinstall agentm-feishu with "
                "its declared dependencies (`uv sync`)."
            ) from exc

        cfg = self._config
        if not cfg.app_id or not cfg.app_secret:
            raise RuntimeError(
                "feishu adapter: app_id / app_secret missing"
            )

        channel = _Lark(app_id=cfg.app_id, app_secret=cfg.app_secret)
        self._channel = channel
        # ``FeishuChannel.on`` in lark-oapi 1.6.x requires the
        # ``(name, handler)`` call form — the decorator form raises.
        channel.on("message", self._on_message)
        channel.on("cardAction", self._on_card_action)
        self._connect_task = asyncio.create_task(
            channel.connect(), name="feishu-channel"
        )
        try:
            await channel.connect_until_ready(timeout=20.0)
        except Exception:
            log.exception("feishu connect_until_ready failed")
            raise
        self._running = True
        # Stay alive until stop() is invoked from outside.
        await self._connect_task

    async def stop(self) -> None:
        if not self._running and self._channel is None:
            return
        self._running = False
        if self._channel is not None:
            try:
                await self._channel.disconnect()
            except Exception:
                log.exception("FeishuAdapter.disconnect raised")
            try:
                await self._channel.stop_background()
            except Exception:
                log.exception("FeishuAdapter.stop_background raised")
        if self._connect_task is not None and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass

    # -- inbound (Feishu → gateway) -----------------------------------

    async def _on_message(self, msg: Any) -> None:
        sender = getattr(msg, "sender_id", "") or ""
        message_id = getattr(msg, "message_id", None)
        chat_id = getattr(msg, "chat_id", "") or ""
        content = getattr(msg, "content_text", "") or ""
        log.info(
            "[feishu] rx chat=%s sender=%s msg_id=%s text=%r",
            chat_id,
            sender,
            message_id,
            content[:120],
        )
        if self._is_self(sender):
            return
        if not self._is_allowed(sender):
            return
        # Quick visual ACK so the user knows the bot got it before the
        # LLM has had a chance to reply. Best-effort, fire-and-forget —
        # we MUST NOT await it here: if the SDK's reaction RPC hangs or
        # holds a lock, awaiting blocks the inbound handler and the
        # message never reaches the agent.
        if message_id and self._config.ack_emoji:
            task = asyncio.create_task(
                self._safe_reaction(message_id, self._config.ack_emoji),
                name="feishu-ack",
            )
            self._pending_acks.setdefault(chat_id, []).append(task)
        await self._forward_inbound(
            sender_id=str(sender),
            chat_id=str(chat_id),
            content=content,
            button_value=None,
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
        sender = (
            getattr(operator, "open_id", None)
            or getattr(operator, "user_id", None)
            or ""
        )
        chat_id = getattr(event, "chat_id", "") or ""
        if not self._is_allowed(str(sender)):
            return
        await self._forward_inbound(
            sender_id=str(sender),
            chat_id=str(chat_id),
            # The gateway routes on ``button_value``; ``content`` is
            # informational only (shows up in observability traces).
            content=f"[card click: {button_value}]",
            button_value=button_value,
        )

    async def _forward_inbound(
        self,
        *,
        sender_id: str,
        chat_id: str,
        content: str,
        button_value: str | None,
    ) -> None:
        body: dict[str, Any] = {
            "channel": self._config.channel_name,
            "sender_id": sender_id,
            "chat_id": chat_id,
            "content": content,
        }
        if button_value is not None:
            body["button_value"] = button_value
        env = Envelope(
            v=WIRE_VERSION,
            id=f"in-feishu-{int(time.time() * 1_000_000)}",
            kind=KIND_INBOUND,
            ts=time.time(),
            body=body,
        )
        try:
            await self._client.send(env)
        except Exception:  # noqa: BLE001
            log.exception("forward_inbound failed; dropping envelope")

    # -- outbound (gateway → Feishu) ----------------------------------

    async def handle_outbound(self, env: Envelope) -> None:
        """Render and deliver one outbound envelope to Feishu."""
        if self._channel is None:
            raise RuntimeError("FeishuAdapter.start() has not completed")
        body = env.body if isinstance(env.body, dict) else {}
        out_kind = str(body.get("kind") or "message")
        chat_id = str(body.get("chat_id") or "")
        if not chat_id:
            log.warning("outbound dropped: empty chat_id (env id=%s)", env.id)
            return
        if out_kind == "turn_complete":
            # Control signal — clear ACK reactions, nothing on the wire.
            await self._clear_pending_acks(chat_id)
            return
        # Buttons travel in the body under "buttons" — list of dicts
        # ``{"label": str, "value": str, "style": str}`` mirroring the
        # in-process Button dataclass. The wire bridge does not yet
        # serialize buttons (Phase 1 outbound body has no field for
        # them); when it gains one, this code path is already prepared.
        buttons_raw = body.get("buttons") or []
        buttons: list[_ButtonLike] = []
        if isinstance(buttons_raw, list):
            for b in buttons_raw:
                if not isinstance(b, dict):
                    continue
                label = str(b.get("label", ""))
                value = str(b.get("value", ""))
                style = str(b.get("style", "default"))
                if label and value:
                    buttons.append(_ButtonLike(label=label, value=value, style=style))
        content = str(body.get("content") or "")
        await self._send_card(chat_id, content, buttons)

    async def _send_card(
        self, chat_id: str, content: str, buttons: list[_ButtonLike]
    ) -> None:
        # Render every chat message as a schema-2.0 interactive card so
        # Chinese, markdown (code fences, lists, headings) and action
        # buttons all use the same path. Lark's plain ``text`` message
        # type does not render markdown.
        assert self._channel is not None
        await self._channel.send(
            chat_id,
            {"card": _markdown_card(content, buttons=buttons)},
        )

    # -- ACK reactions ------------------------------------------------

    async def _safe_reaction(
        self, message_id: str, emoji: str
    ) -> tuple[str, str] | None:
        """Add an ACK reaction; return ``(message_id, reaction_id)`` on
        success so the turn-complete cleanup can remove it.
        """
        assert self._channel is not None
        try:
            result = await self._channel.add_reaction(message_id, emoji)
        except Exception:
            log.exception("[feishu] add_reaction failed for %s", message_id)
            return None
        raw = getattr(result, "raw", None) or {}
        data = raw.get("data") if isinstance(raw, dict) else None
        reaction_id: str | None = None
        if isinstance(data, dict):
            inner = data.get("reaction_id")
            if isinstance(inner, str):
                reaction_id = inner
            else:
                reaction = data.get("reaction")
                if isinstance(reaction, dict):
                    rid = reaction.get("reaction_id")
                    if isinstance(rid, str):
                        reaction_id = rid
        if not reaction_id:
            log.warning(
                "[feishu] add_reaction succeeded but reaction_id missing in %r",
                raw,
            )
            return None
        return message_id, reaction_id

    async def _clear_pending_acks(self, chat_id: str) -> None:
        """Remove every ACK reaction we attached for ``chat_id``."""
        tasks = self._pending_acks.pop(chat_id, None)
        if not tasks:
            return
        assert self._channel is not None
        for task in tasks:
            try:
                pair = await task
            except Exception:
                log.exception("[feishu] ack-add task raised")
                continue
            if not pair:
                continue
            message_id, reaction_id = pair
            try:
                await self._channel.remove_reaction(message_id, reaction_id)
            except Exception:
                log.exception(
                    "[feishu] remove_reaction failed (msg=%s reaction=%s)",
                    message_id,
                    reaction_id,
                )

    # -- helpers -------------------------------------------------------

    def _is_self(self, sender_id: str) -> bool:
        bot = getattr(self._channel, "_bot", None) if self._channel else None
        bot_oid = getattr(bot, "open_id", None) if bot else None
        return bool(bot_oid and sender_id == bot_oid)

    def _is_allowed(self, sender_id: str) -> bool:
        allow = self._config.allow_from
        if not allow:
            log.warning(
                "[feishu] allow_from is empty — denying %s; "
                "pass --allow-from '*' to permit everyone",
                sender_id,
            )
            return False
        if "*" in allow:
            return True
        return str(sender_id) in {str(x) for x in allow}


@dataclass(slots=True)
class _ButtonLike:
    """Local mirror of ``agentm_channels.bus.Button``.

    Kept local so the adapter doesn't need to import bus types — the
    wire transports buttons as plain dicts (when the bridge gains
    button serialization).
    """

    label: str
    value: str
    style: str = "default"


def _markdown_card(text: str, *, buttons: list[_ButtonLike]) -> dict[str, Any]:
    """Construct a Lark schema-2.0 interactive card.

    Body is one ``markdown`` element so Chinese, code fences, lists,
    and inline formatting all render. When ``buttons`` is non-empty an
    ``action`` block is appended; each button's ``value`` round-trips
    through Lark's ``cardAction`` callback as the typed inbound
    ``button_value``. Style mapping is mechanical — no label-string
    heuristics — so the caller controls visual emphasis via
    ``Button.style``.
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


__all__ = ["FeishuAdapter", "FeishuConfig"]
