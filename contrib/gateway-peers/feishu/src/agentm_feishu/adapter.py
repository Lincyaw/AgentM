"""Feishu / Lark adapter for the ``agentm-feishu`` chat-client peer (v2).

Owns a :class:`WireClient`, maps Feishu message / card-action events to
v2 ``inbound`` envelopes (computing ``session_key`` per §3.4 and sending
``scenario`` on the first message for a chat), and renders ``outbound``
envelopes as schema-2.0 interactive cards.

The v1 in-place card-streaming machinery is gone: v2 emits one
assistant-text outbound per turn (§2.5), so each outbound is a fresh
card. (Card streaming is a Phase-2 polish item.) ``metadata.kind`` picks
the render style; approval cards carry ``buttons`` whose ``value``
round-trips back as ``button_value``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agentm.gateway.client import WireClient
from agentm.gateway.wire import KIND_INBOUND, WIRE_VERSION, Envelope

log = logging.getLogger("agentm_feishu.adapter")


# Maps the typed v2 button style to Lark's button type.
_BUTTON_STYLE_MAP: dict[str, str] = {
    "primary": "primary",
    "danger": "danger",
    "default": "default",
}


async def _cancel_lark_tasks() -> None:
    """Cancel pending tasks on lark's private event loop (best-effort).

    lark-oapi's WS client schedules ``_ping_loop`` / ``_receive_message_loop``
    / ``ExpiringCache._start_clear_cron`` via ``loop.create_task`` on a
    separate loop run in a daemon thread. lark's teardown stops that loop
    but leaves the tasks pending, so the asyncio GC logs "Task was
    destroyed but it is pending" at exit. We flag them cancelled for
    hygiene; the cli's log filter silences the residual warning.
    """
    try:
        import lark_oapi.ws.client as _ws_client  # noqa: PLC0415
    except ImportError:
        return
    lark_loop = getattr(_ws_client, "loop", None)
    if lark_loop is None or lark_loop.is_closed():
        return
    try:
        pending = [t for t in asyncio.all_tasks(loop=lark_loop) if not t.done()]
    except RuntimeError:
        return
    for task in pending:
        task.cancel()


@dataclass(slots=True)
class FeishuConfig:
    """Adapter config — derived from CLI flags / env in ``cli.py``."""

    app_id: str
    app_secret: str
    allow_from: list[str] = field(default_factory=lambda: ["*"])
    ack_emoji: str = "OK"
    channel_name: str = "feishu"
    """Channel name announced on inbound envelopes (§2.4) and used as the
    ``<channel>`` half of the computed session_key (§3.4)."""
    scenario: str | None = None
    """Scenario the gateway builds new sessions in; sent on the first
    inbound for a chat the gateway has not seen (§2.2)."""
    session_scope: str = "chat"
    """How session_key is composed (§3.4): ``chat`` -> ``feishu:<chat_id>``;
    ``user`` -> ``feishu:<chat_id>:<sender_id>``."""


class FeishuAdapter:
    """Bridges a ``lark_oapi`` Feishu channel to a :class:`WireClient`."""

    def __init__(self, client: WireClient, config: FeishuConfig) -> None:
        self._client = client
        self._config = config
        self._channel: Any = None
        # Pending ACK reactions keyed by chat_id. Cleared when the agent's
        # reply lands (v2 has no turn_complete signal; the assistant_text
        # outbound is the cue).
        self._pending_acks: dict[str, list[asyncio.Task[Any]]] = {}
        # session_keys we have already sent ``scenario`` for, so later
        # inbounds for the same chat omit it (§2.2).
        self._scenario_sent: set[str] = set()
        self._running = False
        self._stop_event: asyncio.Event = asyncio.Event()

    # -- session_key (§3.4) -------------------------------------------

    def _session_key(self, chat_id: str, sender_id: str) -> str:
        base = f"{self._config.channel_name}:{chat_id}"
        if self._config.session_scope == "user":
            return f"{base}:{sender_id}"
        return base

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
            raise RuntimeError("feishu adapter: app_id / app_secret missing")

        channel = _Lark(app_id=cfg.app_id, app_secret=cfg.app_secret)
        self._channel = channel
        # ``FeishuChannel.on`` in lark-oapi 1.6.x requires the
        # ``(name, handler)`` call form — the decorator form raises.
        channel.on("message", self._on_message)
        channel.on("cardAction", self._on_card_action)
        try:
            await channel.connect_until_ready(timeout=20.0)
        except Exception:
            log.exception("feishu connect_until_ready failed")
            raise
        self._running = True
        await self._stop_event.wait()

    async def stop(self) -> None:
        if not self._running and self._channel is None:
            self._stop_event.set()
            return
        self._running = False
        self._stop_event.set()
        if self._channel is not None:
            try:
                await self._channel.disconnect()
            except Exception:
                log.exception("FeishuAdapter.disconnect raised")
            try:
                await self._channel.stop_background()
            except Exception:
                log.exception("FeishuAdapter.stop_background raised")
        await _cancel_lark_tasks()

    # -- inbound (Feishu -> gateway) ----------------------------------

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
        # Quick visual ACK so the user knows the bot got it. Fire-and-forget
        # — awaiting here would block the inbound handler if the RPC hangs.
        if message_id and self._config.ack_emoji:
            task = asyncio.create_task(
                self._safe_reaction(message_id, self._config.ack_emoji),
                name="feishu-ack",
            )
            self._pending_acks.setdefault(str(chat_id), []).append(task)
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
        session_key = self._session_key(chat_id, sender_id)
        body: dict[str, Any] = {
            "channel": self._config.channel_name,
            "sender_id": sender_id,
            "chat_id": chat_id,
            "content": content,
        }
        if button_value is not None:
            body["button_value"] = button_value
        scenario = None
        if session_key not in self._scenario_sent:
            scenario = self._config.scenario
            self._scenario_sent.add(session_key)
        env = Envelope(
            v=WIRE_VERSION,
            id=f"in-feishu-{int(time.time() * 1_000_000)}",
            kind=KIND_INBOUND,
            ts=time.time(),
            session_key=session_key,
            scenario=scenario,
            body=body,
        )
        try:
            await self._client.send(env)
        except Exception:  # noqa: BLE001
            log.exception("forward_inbound failed; dropping envelope")

    # -- outbound (gateway -> Feishu) ---------------------------------

    async def handle_outbound(self, env: Envelope) -> None:
        """Render and deliver one v2 ``outbound`` envelope to Feishu (§2.5)."""
        if self._channel is None:
            raise RuntimeError("FeishuAdapter.start() has not completed")
        body = env.body if isinstance(env.body, dict) else {}
        chat_id = str(body.get("chat_id") or "")
        if not chat_id:
            log.warning("outbound dropped: empty chat_id (env id=%s)", env.id)
            return
        meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        meta_kind = str(meta.get("kind") or "assistant_text")
        # The agent has produced output — clear the ACK reaction(s) for
        # this chat (v2 has no turn_complete; assistant_text is the cue).
        if meta_kind == "assistant_text":
            await self._clear_pending_acks(chat_id)
        buttons: list[_ButtonLike] = []
        for b in body.get("buttons") or []:
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
        # Every chat message is a schema-2.0 interactive card so Chinese,
        # markdown, and action buttons share one path.
        assert self._channel is not None
        await self._channel.send(
            chat_id,
            {"card": _markdown_card(content, buttons=buttons)},
        )

    # -- ACK reactions ------------------------------------------------

    async def _safe_reaction(
        self, message_id: str, emoji: str
    ) -> tuple[str, str] | None:
        """Add an ACK reaction; return ``(message_id, reaction_id)`` so the
        cleanup can remove it."""
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
        """Remove every ACK reaction attached for ``chat_id``."""
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
    """Local mirror of the wire ``Button`` shape (kept local so the
    adapter need not import gateway types)."""

    label: str
    value: str
    style: str = "default"


def _markdown_card(text: str, *, buttons: list[_ButtonLike]) -> dict[str, Any]:
    """Construct a Lark schema-2.0 interactive card.

    Body is one ``markdown`` element so Chinese, code fences, lists, and
    inline formatting render. A non-empty ``buttons`` list appends an
    ``action`` block; each button's ``value`` round-trips through Lark's
    ``cardAction`` callback as the typed ``button_value``.
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
