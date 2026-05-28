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


async def _cancel_lark_tasks() -> None:
    """Cancel pending tasks on lark's private event loop (best-effort).

    lark-oapi's WS client schedules ``_ping_loop`` / ``_receive_message_loop``
    / ``ExpiringCache._start_clear_cron`` via ``loop.create_task`` on a
    separate loop installed by :func:`apply_ws_patch` and run in a
    daemon thread. lark's own teardown stops that loop after closing
    the WebSocket — but the tasks scheduled on it are left pending,
    and the stopped loop can't process ``task.cancel()`` callbacks
    from our main thread. The asyncio garbage collector then logs
    "Task was destroyed but it is pending" at process exit.

    Trying to restart that loop from our main thread fights both
    ``asyncio.run`` (only one loop per thread) and the bg thread that
    used to drive it. The pragmatic move is to (a) flag the tasks
    cancelled for hygiene, then (b) silence the asyncio warning for
    coroutines that live in ``lark_oapi.*`` — those tasks are owned by
    a daemon thread that dies with the process anyway, so the warning
    has no remediation value. See :func:`_install_lark_warning_filter`.
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
    chat_id_prefix: str = "feishu"
    ack_emoji: str = "OK"
    channel_name: str = "feishu"
    """Channel name announced on inbound envelopes. The gateway uses it
    as the synthetic-channel key registered through ``inject_channel``.
    """
    stream_debounce_s: float = 0.3
    """Trailing-edge debounce window for streamed message updates.
    Feishu's ``update_card`` is rate-limited to roughly 5 QPS per
    message; coalescing rapid updates into one patch every ~300 ms
    keeps us comfortably under that ceiling while still feeling live.
    """


@dataclass(slots=True)
class _StreamState:
    """Per-(chat_id, stream_id) state for in-place card updates.

    The adapter sends the first frame as a fresh card and records
    ``message_id`` from the SendResult; every subsequent frame for the
    same stream re-renders the card and calls ``channel.update_card``.
    A debounced flush task (see :attr:`flush_task`) coalesces rapid
    updates so we patch at most once per ``stream_debounce_s``.
    """

    message_id: str | None = None
    """``None`` until the first SendResult comes back. Frames that
    arrive before the initial send completes are queued by simply
    overwriting :attr:`pending` — they'll be flushed once the
    initial send registers ``message_id``."""

    pending: tuple[str, list["_ButtonLike"]] | None = None
    """Latest queued (content, buttons) waiting for the next patch
    window. Always the most recent update — older intermediate frames
    are discarded by design."""

    flush_task: asyncio.Task[Any] | None = None
    """Sleeper task that waits ``stream_debounce_s`` then flushes
    :attr:`pending`. None when no flush is scheduled."""

    flush_now: asyncio.Event = field(default_factory=asyncio.Event)
    """Set to short-circuit the debounce sleep — the flush task wakes
    on either the timer or this event. Used by ``final=True`` frames
    and ``turn_complete`` to land the last update immediately."""

    final_seen: bool = False
    """Set when an outbound for this stream arrived with ``final=True``.
    The flush path tears the state down once ``pending`` drains."""


class FeishuAdapter:
    """Bridges a ``lark_oapi`` Feishu channel to a :class:`WireClient`."""

    def __init__(self, client: WireClient, config: FeishuConfig) -> None:
        self._client = client
        self._config = config
        self._channel: Any = None
        # Pending ACK reactions keyed by chat_id — same lifetime contract
        # as the legacy implementation. Cleared on turn_complete.
        self._pending_acks: dict[str, list[asyncio.Task[Any]]] = {}
        # Per-(chat_id, stream_id) state for streamed updates. Keyed on
        # the tuple so two streams to the same chat (e.g. tool-call
        # progress + final reply) don't clobber each other.
        self._streams: dict[tuple[str, str], _StreamState] = {}
        self._streams_lock = asyncio.Lock()
        self._running = False
        # Signalled by stop() to wake start() out of its idle wait. We
        # used to ``await asyncio.create_task(channel.connect())`` here,
        # but lark-oapi 1.6.x made ``connect()`` a thin alias for
        # ``start_background()`` — it returns once the WebSocket handshake
        # completes, instead of blocking for the lifetime of the
        # connection. That made adapter.start() exit ~0.4 s after handshake,
        # the cli's asyncio.wait fall through, and the finally block
        # ``adapter.stop()`` tear the WS down with code 1000 (mis-logged
        # as ERROR by lark). An explicit stop event makes the intended
        # "block until somebody tells me to stop" lifetime obvious.
        self._stop_event: asyncio.Event = asyncio.Event()

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
        try:
            # start_background spawns the WS receive loop, returns once
            # the handshake is ready.
            await channel.connect_until_ready(timeout=20.0)
        except Exception:
            log.exception("feishu connect_until_ready failed")
            raise
        self._running = True
        # Block until stop() is invoked. The lark client keeps its own
        # background reader / ping tasks alive in the meantime.
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
        # lark's ws client spawns its own ``_ping_loop`` /
        # ``_receive_message_loop`` / ``ExpiringCache._start_clear_cron``
        # tasks via ``loop.create_task`` and never tracks them. After our
        # disconnect closes the WS, ``auto_reconnect=True`` (lark's
        # default) keeps the recv loop alive waiting to reconnect. Find
        # any task whose coroutine lives under lark_oapi and cancel it
        # so the process can exit clean without "Task was destroyed but
        # it is pending" warnings.
        await _cancel_lark_tasks()

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
            # Control signal — clear ACK reactions and flush any active
            # streams in this chat (a turn ending without an explicit
            # final=True frame still needs the in-place card to settle).
            await self._clear_pending_acks(chat_id)
            await self._finalize_streams_for_chat(chat_id)
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
        stream_id = body.get("stream_id")
        final = bool(body.get("final"))
        if isinstance(stream_id, str) and stream_id:
            await self._handle_stream_frame(
                chat_id=chat_id,
                stream_id=stream_id,
                content=content,
                buttons=buttons,
                final=final,
            )
            return
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

    # -- streaming ----------------------------------------------------

    async def _handle_stream_frame(
        self,
        *,
        chat_id: str,
        stream_id: str,
        content: str,
        buttons: list[_ButtonLike],
        final: bool,
    ) -> None:
        """Apply one frame of a streamed update.

        First frame for a ``(chat_id, stream_id)`` does a real ``send``
        and captures the resulting ``message_id``. Subsequent frames
        queue into the state's ``pending`` slot and schedule a
        trailing-edge debounce flush. The ``final`` flag forces an
        immediate flush and tears the entry down once delivered.
        """
        assert self._channel is not None
        key = (chat_id, stream_id)
        async with self._streams_lock:
            state = self._streams.get(key)
            if state is None:
                state = _StreamState()
                self._streams[key] = state
                first_frame = True
            else:
                first_frame = False
            if final:
                state.final_seen = True

        if first_frame:
            # Initial send happens synchronously so we capture
            # message_id before any subsequent frame tries to patch.
            try:
                result = await self._channel.send(
                    chat_id,
                    {"card": _markdown_card(content, buttons=buttons)},
                )
            except Exception:
                log.exception(
                    "[feishu] stream %s: initial send failed; dropping stream",
                    stream_id,
                )
                async with self._streams_lock:
                    self._streams.pop(key, None)
                return
            message_id = getattr(result, "message_id", None)
            async with self._streams_lock:
                state.message_id = (
                    str(message_id) if isinstance(message_id, str) else None
                )
            if final:
                async with self._streams_lock:
                    self._streams.pop(key, None)
            return

        # Subsequent frame: stash latest content + schedule flush.
        async with self._streams_lock:
            state.pending = (content, buttons)
            if final:
                # Either wake the running flush task so it skips the
                # rest of the debounce window, or start one with no
                # debounce if none was scheduled yet.
                state.flush_now.set()
                if state.flush_task is None:
                    state.flush_task = asyncio.create_task(
                        self._flush_stream(key, debounce=False),
                        name=f"feishu-stream-flush-{stream_id}",
                    )
            elif state.flush_task is None:
                state.flush_task = asyncio.create_task(
                    self._flush_stream(key, debounce=True),
                    name=f"feishu-stream-flush-{stream_id}",
                )

    async def _flush_stream(
        self, key: tuple[str, str], *, debounce: bool
    ) -> None:
        """Trailing-edge flush loop for one stream's pending updates.

        Loops:
          1. wait the debounce window (skipped on the first pass if
             ``debounce=False`` — used by ``final=True`` and turn_complete
             paths so the last frame lands immediately);
          2. pull the latest pending content (older frames already
             dropped by the producer side);
          3. patch the card (or fall back to a fresh send if the initial
             send never yielded a ``message_id``);
          4. if another frame arrived during the patch RPC, debounce
             again and patch the new latest; otherwise release the
             flush slot (and drop the state if ``final_seen``).
        """
        chat_id, stream_id = key
        first = True
        while True:
            async with self._streams_lock:
                state = self._streams.get(key)
                if state is None:
                    return
                event = state.flush_now
            if not (first and not debounce):
                try:
                    await asyncio.wait_for(
                        event.wait(), timeout=self._config.stream_debounce_s
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    pass
                except asyncio.CancelledError:
                    return
            first = False

            async with self._streams_lock:
                state = self._streams.get(key)
                if state is None:
                    return
                pending = state.pending
                state.pending = None
                state.flush_now.clear()
                if pending is None:
                    state.flush_task = None
                    if state.final_seen:
                        self._streams.pop(key, None)
                    return
                message_id = state.message_id

            content, buttons = pending
            if message_id is None:
                try:
                    await self._send_card(chat_id, content, buttons)
                except Exception:
                    log.exception(
                        "[feishu] stream %s: fallback send failed", stream_id
                    )
            else:
                try:
                    assert self._channel is not None
                    await self._channel.update_card(
                        message_id,
                        _markdown_card(content, buttons=buttons),
                    )
                except Exception:
                    log.exception(
                        "[feishu] stream %s: update_card failed (msg=%s)",
                        stream_id,
                        message_id,
                    )
            # If this was the final frame and no new frame arrived
            # during the patch RPC, the stream is done — release the
            # slot now instead of waiting another debounce window.
            async with self._streams_lock:
                state = self._streams.get(key)
                if state is None:
                    return
                if state.final_seen and state.pending is None:
                    state.flush_task = None
                    self._streams.pop(key, None)
                    return

    async def _finalize_streams_for_chat(self, chat_id: str) -> None:
        """Flush + drop every stream belonging to ``chat_id``.

        Called on ``turn_complete``: any stream that ended without an
        explicit ``final=True`` frame still needs its last update to
        land so the in-place card matches the agent's actual final
        output. We force ``final_seen`` and trigger a no-debounce
        flush for each affected key.
        """
        start_keys: list[tuple[str, str]] = []
        async with self._streams_lock:
            for key, state in self._streams.items():
                if key[0] != chat_id:
                    continue
                state.final_seen = True
                # Wake any in-flight flush task so its debounce wait
                # returns immediately and it patches with the latest
                # pending content.
                state.flush_now.set()
                if state.pending is not None and state.flush_task is None:
                    start_keys.append(key)
        for key in start_keys:
            async with self._streams_lock:
                pending_state = self._streams.get(key)
                if pending_state is None or pending_state.flush_task is not None:
                    continue
                pending_state.flush_task = asyncio.create_task(
                    self._flush_stream(key, debounce=False),
                    name=f"feishu-stream-finalize-{key[1]}",
                )
        # Garbage-collect entries with no pending work.
        async with self._streams_lock:
            stale = [
                k
                for k, s in self._streams.items()
                if k[0] == chat_id and s.pending is None and s.flush_task is None
            ]
            for k in stale:
                self._streams.pop(k, None)

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
