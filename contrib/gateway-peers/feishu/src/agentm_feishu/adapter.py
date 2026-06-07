"""Feishu / Lark adapter for the ``agentm-feishu`` chat-client peer (v2).

Owns a :class:`WireClient`, maps Feishu message / card-action events to
v2 ``inbound`` envelopes (computing ``session_key`` per §3.4 and sending
``scenario`` on the first message for a chat), and renders ``outbound``
envelopes as schema-2.0 interactive cards.

**Live-turn rendering.** Since #187 the gateway fans out the full session
event surface (``OutboundMetaKind`` — 25 kinds: ``turn_start``,
``stream_text``, ``tool_call``, ``usage`` …), not just the five durable
chat kinds. A naive "one card per outbound" adapter would flood the chat
with a card per token / event. Instead we collapse a whole agent run into
**one card per turn, updated in place**: the first consumed event sends
the card (``channel.send`` returns its message_id) and every later state
transition patches it with ``channel.update_card``. A status line shows
what the agent is doing right now (思考中 → 调用工具 X → 完成); tool
steps accumulate in a collapsible panel so detail is available without
overwhelming; the final ``assistant_text`` fills the answer body.

Runtime/observability kinds that don't belong in a chat (``usage``,
``child_*``, ``extension_*``, ``api_*``, ``session_ready`` …) are
consumed and dropped. ``approval_request`` and ``diagnostic_error`` /
``diagnostic_warning`` get their own standalone cards — approvals carry
``buttons`` whose ``value`` round-trips back as ``button_value`` and must
persist, so they are never folded into the (overwritten) live card.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Future
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

# Activity-line labels for the live turn card. The activity line shows the
# latest operation (see ``_activity_markdown``) so the chat perceives live
# progress; there is deliberately no generic "正在回答" state.
_STATUS_THINKING = "🤔 思考中…"
_STATUS_DONE = "✅ 完成"
_STATUS_INTERRUPTED = "⏹ 已中断"

# Label / action value for the in-place "stop" button shown on a live,
# in-progress turn card. The button round-trips ``{"control": "interrupt"}``
# (NOT ``button_value``), so ``_on_card_action`` can distinguish it from an
# approval button and forward it as an interrupt control verb. Clicking it
# aborts the in-flight turn server-side (kernel abort signal) while preserving
# context; the user's next message is then acted on immediately.
_STOP_LABEL = "⏹ 停止"
_INTERRUPT_CONTROL = "interrupt"

# How many recent tool steps the activity line shows at once.
_ACTIVITY_TAIL = 2

# Outbound kinds the live-turn renderer consumes to drive one card. Every
# other kind is either rendered as a standalone card (approval / diagnostic)
# or dropped as chat-irrelevant runtime/observability noise (usage, child_*,
# extension_*, api_*, resource_write, plan_submitted, after_compact,
# cost_budget_exceeded, session_ready, command_dispatched).
_LIVE_KINDS: frozenset[str] = frozenset(
    {
        "turn_start",
        "stream_thinking",
        "tool_call",
        "tool_result",
        "stream_text",
        "assistant_text",
        "agent_end",
    }
)

# Minimum wall-clock gap between two in-place updates of the same card.
# Feishu throttles card patches; bursts (a flurry of fast tool calls) are
# coalesced into a single trailing update instead of one patch per event.
_MIN_UPDATE_INTERVAL = 0.35

# Grace window opened by ``agent_end`` before a turn is finalized. Since the
# gateway unified delivery onto one ordered channel (single-process-gateway
# §2.6), ``agent_end`` no longer overtakes the run's final ``assistant_text``
# — they arrive in event order. This window is now a **defensive backstop**
# (e.g. against any future transport that reorders, or a text-less run): each
# trailing assistant_text restarts it (multi-turn runs emit one per turn) and
# its expiry is the finalize cue; a pure-tool run with no reply just waits it
# out. With ordering fixed at the source it could be shortened, but a small
# window is cheap insurance.
_AGENT_END_GRACE = 1.5


def _result_message_id(result: Any) -> str | None:
    """Extract the Feishu message_id from a lark ``SendResult``.

    ``channel.send`` returns a structured ``SendResult`` (``.message_id`` /
    ``.ok`` / ``.raw``), not a bare string — stringifying it yields the
    repr, which ``update_card`` then rejects as an invalid open_message_id.
    Falls back to treating a plain string as the id (defensive).
    """
    mid = getattr(result, "message_id", None)
    if mid:
        return str(mid)
    if isinstance(result, str) and result:
        return result
    return None


def _pretty_tool(name: str) -> str:
    """Trim an MCP-style ``a__b__tool`` name to its final segment."""
    return name.rsplit("__", 1)[-1] if "__" in name else name


def _arg_summary(args: Any) -> str:
    """One short human hint from a tool's args (first non-empty scalar)."""
    if not isinstance(args, dict):
        return ""
    for value in args.values():
        if isinstance(value, str) and value.strip():
            text = value.strip().replace("\n", " ")
            return text if len(text) <= 40 else text[:39] + "…"
        if isinstance(value, (int, float, bool)):
            return str(value)
    return ""


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
class _Step:
    """One tool invocation shown in the live card's collapsible panel."""

    call_id: str
    name: str
    summary: str = ""
    state: str = "running"  # running | ok | fail


@dataclass(slots=True)
class _LiveTurn:
    """In-place card backing one agent run for a chat (see module docstring).

    A run starts on the first consumed event and ends on ``agent_end``;
    every state transition patches the same Feishu card via its
    ``message_id``.
    """

    chat_id: str
    body: str = ""
    steps: list[_Step] = field(default_factory=list)
    message_id: str | None = None
    # True once any agent-lifecycle event (turn_start / stream_* / tool_*)
    # has been seen. A turn that reaches assistant_text WITHOUT this is a
    # one-shot reply (a slash command's ctx.reply emits no turn_start /
    # agent_end), so it must finalize immediately instead of waiting for an
    # agent_end that never comes — otherwise it becomes a zombie turn that
    # later turns reuse, freezing every reply onto one stale card.
    saw_lifecycle: bool = False
    agent_ended: bool = False
    finalized: bool = False
    interrupted: bool = False
    last_render: float = 0.0
    pending: bool = False
    flush_task: asyncio.Task[Any] | None = None
    finalize_task: asyncio.Task[Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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
        # The asyncio loop that owns the WireClient + card rendering. lark's
        # WS callbacks (_on_message / _on_card_action) fire on lark's OWN loop
        # in a daemon thread, so any reaction work scheduled from there must be
        # bounced onto this loop — otherwise an ack task created on lark's loop
        # is awaited from this one and asyncio raises "Future attached to a
        # different loop". Captured in :meth:`start`.
        self._main_loop: asyncio.AbstractEventLoop | None = None
        # Pending ACK reactions keyed by chat_id, as concurrent futures from
        # run_coroutine_threadsafe (resolved on the main loop). Cleared when the
        # agent's reply lands (v2 has no turn_complete; assistant_text is cue).
        self._pending_acks: dict[str, list[Future[Any]]] = {}
        # session_keys we have already sent ``scenario`` for, so later
        # inbounds for the same chat omit it (§2.2).
        self._scenario_sent: set[str] = set()
        # Live turn card per chat_id (one in-flight agent run -> one card).
        self._live: dict[str, _LiveTurn] = {}
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
        # Capture the loop that owns the WireClient / card rendering, so lark's
        # daemon-thread callbacks can bounce reaction work back onto it.
        self._main_loop = asyncio.get_running_loop()
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
        for turn in self._live.values():
            for task in (turn.flush_task, turn.finalize_task):
                if task is not None and not task.done():
                    task.cancel()
        self._live.clear()
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
        # Quick visual ACK so the user knows the bot got it. Scheduled onto the
        # main loop (this callback runs on lark's daemon-thread loop) so the
        # reaction task — and its later removal in _clear_pending_acks — live on
        # the same loop. Fire-and-forget: awaiting here would block the inbound
        # handler if the RPC hangs.
        if message_id and self._config.ack_emoji and self._main_loop is not None:
            cf = asyncio.run_coroutine_threadsafe(
                self._safe_reaction(message_id, self._config.ack_emoji),
                self._main_loop,
            )
            self._pending_acks.setdefault(str(chat_id), []).append(cf)
        await self._forward_inbound(
            sender_id=str(sender),
            chat_id=str(chat_id),
            content=content,
            button_value=None,
        )

    async def _on_card_action(self, event: Any) -> None:
        value = getattr(event.action, "value", None) or {}
        button_value: str | None = None
        control: str | None = None
        if isinstance(value, dict):
            # Two disjoint card-button vocabularies share this callback:
            #   - approval cards carry ``button_value`` (a conversational
            #     answer the agent is waiting on);
            #   - the live-turn stop button carries ``control`` (an out-of-band
            #     verb, currently only ``interrupt``).
            # ``control`` wins if both are somehow present so an interrupt is
            # never misrouted as an approval reply.
            raw_control = value.get("control")
            if raw_control is not None:
                control = str(raw_control)
            else:
                raw = value.get("button_value")
                if raw is not None:
                    button_value = str(raw)
        if not button_value and not control:
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
        if control == _INTERRUPT_CONTROL:
            # Reflect the interrupt on the live card right away so the user
            # sees it took effect, then forward the control verb so the
            # gateway aborts the in-flight turn (context preserved).
            await self._mark_interrupted(str(chat_id))
            await self._forward_inbound(
                sender_id=str(sender),
                chat_id=str(chat_id),
                content="[interrupt]",
                control=_INTERRUPT_CONTROL,
            )
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
        button_value: str | None = None,
        control: str | None = None,
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
        if control is not None:
            # Out-of-band control verb (e.g. ``interrupt``) parsed by
            # ``InboundBody.control`` server-side; not a conversational turn.
            body["control"] = control
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
        """Dispatch one v2 ``outbound`` envelope to its render path.

        Live-transcript kinds drive a single in-place turn card; approvals
        and diagnostics get standalone cards; everything else is dropped as
        chat-irrelevant noise (see module docstring).
        """
        if self._channel is None:
            raise RuntimeError("FeishuAdapter.start() has not completed")
        body = env.body if isinstance(env.body, dict) else {}
        # Defence-in-depth: the gateway routes outbound by channel (§3.2),
        # but a dumb adapter must also refuse to post anything not addressed
        # to its own channel — otherwise a foreign-channel reply would land
        # in Feishu. An empty channel is allowed through (degenerate case).
        out_channel = str(body.get("channel") or "")
        if out_channel and out_channel != self._config.channel_name:
            return
        chat_id = str(body.get("chat_id") or "")
        if not chat_id:
            log.warning("outbound dropped: empty chat_id (env id=%s)", env.id)
            return
        raw_meta = body.get("metadata")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        meta_kind = str(meta.get("kind") or "assistant_text")

        if meta_kind == "approval_request":
            await self._send_approval(chat_id, body)
            return
        if meta_kind in ("diagnostic_error", "diagnostic_warning"):
            await self._send_alert(chat_id, body, meta_kind)
            return
        if meta_kind not in _LIVE_KINDS:
            return  # chat-irrelevant runtime/observability frame
        await self._apply_live(chat_id, meta_kind, body, meta)

    # -- live turn card -----------------------------------------------

    async def _apply_live(
        self, chat_id: str, kind: str, body: dict[str, Any], meta: dict[str, Any]
    ) -> None:
        """Fold one live-transcript event into the chat's turn card.

        The get-or-create below has no lock: it is safe because WireClient
        dispatches outbound frames one at a time (``await on_outbound`` per
        frame in its read loop), so calls into here are serialised per
        connection and never interleave at the create. The deferred-flush /
        finalize tasks only ever call :meth:`_render` (guarded by
        ``turn.lock``), never this method.
        """
        turn = self._live.get(chat_id)
        if turn is None or turn.finalized:
            turn = _LiveTurn(chat_id=chat_id)
            self._live[chat_id] = turn

        # Any of these marks an actual agent run (a slash-command reply emits
        # only a bare assistant_text), so the turn must wait for agent_end.
        if kind in ("turn_start", "stream_thinking", "tool_call", "tool_result",
                    "stream_text"):
            turn.saw_lifecycle = True

        if kind == "turn_start":
            # One per LLM turn. Render once to show 思考中 before any tool;
            # later turns within a run change nothing visible (the activity
            # line tracks tools), so don't re-render.
            if turn.message_id is not None:
                return
        elif kind == "stream_thinking":
            # Per-token thinking delta; only worth a card if none exists yet.
            if turn.message_id is not None:
                return
        elif kind == "tool_call":
            turn.steps.append(
                _Step(
                    call_id=str(meta.get("tool_call_id") or ""),
                    name=str(meta.get("name") or "tool"),
                    summary=_arg_summary(meta.get("args")),
                )
            )
        elif kind == "tool_result":
            call_id = str(meta.get("tool_call_id") or "")
            ok = bool(meta.get("ok", True))
            for step in reversed(turn.steps):
                if step.call_id == call_id:
                    step.state = "ok" if ok else "fail"
                    break
        elif kind == "stream_text":
            # No generic "正在回答" status — the activity line shows the
            # latest operation instead. The body is filled authoritatively
            # by assistant_text, so a text delta adds nothing to render.
            return
        elif kind == "assistant_text":
            # Each agent turn emits one assistant_text; only the last (after
            # agent_end) is the final answer. We can't tell which is last from
            # the frame, so we always render the newest into the body. If the
            # agent has already ended, restart the grace window rather than
            # finalizing — a further trailing assistant_text would otherwise
            # reopen a second card.
            turn.body = str(body.get("content") or "")
            await self._clear_pending_acks(chat_id)
            if not turn.saw_lifecycle:
                # One-shot reply (slash command) — no agent_end will come.
                # Finalize now so it lands as its own card and the next reply
                # starts fresh instead of reusing this turn.
                await self._finalize(turn)
                return
            if turn.agent_ended:
                self._schedule_finalize_timeout(turn, restart=True)
        elif kind == "agent_end":
            # Never finalize on agent_end itself: the run's final
            # assistant_text (durable) reliably trails it. Open the grace
            # window; the trailing assistant_text restarts it, and its
            # expiry is what finalizes the card.
            turn.agent_ended = True
            self._schedule_finalize_timeout(turn)
            return

        await self._maybe_render(turn)

    def _schedule_finalize_timeout(
        self, turn: _LiveTurn, *, restart: bool = False
    ) -> None:
        """Arm (or restart) the grace-period finalize after ``agent_end``."""
        if restart and turn.finalize_task is not None and not turn.finalize_task.done():
            turn.finalize_task.cancel()
            turn.finalize_task = None
        if turn.finalize_task is not None and not turn.finalize_task.done():
            return
        turn.finalize_task = asyncio.create_task(
            self._finalize_after_grace(turn), name="feishu-finalize-grace"
        )

    async def _finalize_after_grace(self, turn: _LiveTurn) -> None:
        try:
            await asyncio.sleep(_AGENT_END_GRACE)
        except asyncio.CancelledError:
            return
        # Detach self before finalizing so _finalize's cancel-the-timer step
        # does not cancel this very coroutine mid-flight.
        turn.finalize_task = None
        if not turn.finalized:
            await self._finalize(turn)

    async def _finalize(self, turn: _LiveTurn) -> None:
        """Render the terminal state of a turn card and retire it (idempotent)."""
        if turn.finalized:
            return
        turn.finalized = True
        if turn.finalize_task is not None and not turn.finalize_task.done():
            turn.finalize_task.cancel()
        # A run that produced nothing chat-relevant (no card, no steps, no
        # text) should not leave a bare "完成" card behind.
        if turn.message_id is None and not turn.steps and not turn.body:
            self._live.pop(turn.chat_id, None)
            return
        if not turn.body:
            turn.body = "_(已完成,无文本回复)_"
        await self._clear_pending_acks(turn.chat_id)
        await self._maybe_render(turn, force=True)
        self._live.pop(turn.chat_id, None)

    async def _mark_interrupted(self, chat_id: str) -> None:
        """Flag the chat's live turn as interrupted and repaint its card.

        Best-effort UX feedback for a 停止 click: if a live (non-finalized)
        turn exists, set its activity line to "已中断" so the user sees the
        interrupt registered. The server-side abort then ends the turn (the
        trailing assistant_text / agent_end finalizes the card as usual). A
        no-op if nothing is running, mirroring ``sess.interrupt()``.
        """
        turn = self._live.get(chat_id)
        if turn is None or turn.finalized:
            return
        turn.interrupted = True
        await self._maybe_render(turn, force=True)

    async def _maybe_render(self, turn: _LiveTurn, *, force: bool = False) -> None:
        """Render now, or schedule a trailing flush, respecting the throttle."""
        if force:
            if turn.flush_task is not None and not turn.flush_task.done():
                turn.flush_task.cancel()
            turn.pending = False
            await self._render(turn)
            return
        if time.monotonic() - turn.last_render >= _MIN_UPDATE_INTERVAL:
            await self._render(turn)
            return
        # Too soon — coalesce into a single trailing update.
        turn.pending = True
        if turn.flush_task is None or turn.flush_task.done():
            turn.flush_task = asyncio.create_task(
                self._deferred_flush(turn), name="feishu-card-flush"
            )

    async def _deferred_flush(self, turn: _LiveTurn) -> None:
        try:
            await asyncio.sleep(_MIN_UPDATE_INTERVAL)
        except asyncio.CancelledError:
            return
        if turn.pending and not turn.finalized:
            turn.pending = False
            await self._render(turn)

    async def _render(self, turn: _LiveTurn) -> None:
        """Send (first frame) or patch (later frames) the live card."""
        assert self._channel is not None
        card = _live_card(turn)
        async with turn.lock:
            try:
                if turn.message_id is None:
                    result = await self._channel.send(turn.chat_id, {"card": card})
                    turn.message_id = _result_message_id(result)
                    if turn.message_id is None:
                        log.warning("[feishu] live card send returned no message_id")
                else:
                    await self._channel.update_card(turn.message_id, card)
            except Exception:
                log.exception("[feishu] live card render failed (chat=%s)", turn.chat_id)
                return
            turn.last_render = time.monotonic()

    # -- standalone cards (approval / diagnostics) --------------------

    async def _send_approval(self, chat_id: str, body: dict[str, Any]) -> None:
        """Approvals need a persistent, interactive card of their own."""
        assert self._channel is not None
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
        await self._channel.send(
            chat_id, {"card": _markdown_card(content, buttons=buttons)}
        )

    async def _send_alert(
        self, chat_id: str, body: dict[str, Any], kind: str
    ) -> None:
        assert self._channel is not None
        icon = "⚠️" if kind == "diagnostic_warning" else "🛑"
        content = str(body.get("content") or "")
        source = str(body.get("source") or "") or str(
            (body.get("metadata") or {}).get("source") or ""
        )
        text = f"{icon} **{content}**"
        if source:
            text += f"\n\n_来源: {source}_"
        await self._channel.send(chat_id, {"card": _markdown_card(text, buttons=[])})

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
        futures = self._pending_acks.pop(chat_id, None)
        if not futures:
            return
        assert self._channel is not None
        for cf in futures:
            try:
                # The reaction coro runs on the main loop (scheduled via
                # run_coroutine_threadsafe); wrap_future lets us await its
                # concurrent.futures.Future from this same loop.
                pair = await asyncio.wrap_future(cf)
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
    inline formatting render. Each button in ``buttons`` is appended as a
    standalone ``button`` element — schema 2.0 dropped the 1.0 ``action``/
    ``actions`` container (it rejects ``tag: action`` with ErrCode 200861).
    Each button's ``value`` round-trips through Lark's ``cardAction``
    callback as the typed ``button_value``.
    """
    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": text or "*(empty)*"}
    ]
    for btn in buttons:
        elements.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": btn.label},
                "type": _BUTTON_STYLE_MAP.get(btn.style, "default"),
                "value": {"button_value": btn.value},
            }
        )
    return {"schema": "2.0", "body": {"elements": elements}}


_STEP_ICON: dict[str, str] = {"running": "⏳", "ok": "✅", "fail": "❌"}


def _steps_markdown(steps: list[_Step]) -> str:
    lines: list[str] = []
    for step in steps:
        line = f"{_STEP_ICON.get(step.state, '•')} {_pretty_tool(step.name)}"
        if step.summary:
            line += f" · {step.summary}"
        lines.append(line)
    return "\n".join(lines)


def _activity_markdown(turn: _LiveTurn) -> str:
    """The live activity line: the latest 1-2 operations (or 思考中 / 完成)."""
    if turn.finalized:
        return _STATUS_INTERRUPTED if turn.interrupted else _STATUS_DONE
    if turn.interrupted:
        return _STATUS_INTERRUPTED
    if turn.steps:
        return _steps_markdown(turn.steps[-_ACTIVITY_TAIL:])
    return _STATUS_THINKING


def _live_card(turn: _LiveTurn) -> dict[str, Any]:
    """Render the current state of a live turn as a schema-2.0 card.

    Layout (see module docstring): a live **activity line** showing the
    latest 1-2 operations, the answer body once it exists, and a
    collapsible panel with the full tool-step history. The panel
    auto-expands while the agent is still working (no body yet) so progress
    is visible, and collapses once the answer lands so detail does not
    crowd the reply.
    """
    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "element_id": "status", "content": _activity_markdown(turn)}
    ]
    # While the run is in flight (not finalized, not already interrupted),
    # carry a stop button so the user can abort the turn like Ctrl-C. Its
    # click round-trips ``{"control": "interrupt"}`` (distinct from approval
    # buttons' ``button_value``). Once the turn finalizes or is interrupted
    # the button is dropped — a retired card must not invite another abort.
    if not turn.finalized and not turn.interrupted:
        # Schema 2.0: a standalone ``button`` element (the 1.0 ``action``
        # container is rejected with ErrCode 200861). ``element_id`` rides on
        # the button itself so partial card updates can target it.
        elements.append(
            {
                "tag": "button",
                "element_id": "stop",
                "text": {"tag": "plain_text", "content": _STOP_LABEL},
                "type": "danger",
                "value": {"control": _INTERRUPT_CONTROL},
            }
        )
    if turn.body:
        elements.append({"tag": "hr", "element_id": "sep"})
        elements.append(
            {"tag": "markdown", "element_id": "body", "content": turn.body}
        )
    if turn.steps:
        elements.append(
            {
                "tag": "collapsible_panel",
                "element_id": "steps",
                "expanded": not (turn.finalized or bool(turn.body)),
                "header": {
                    "title": {
                        "tag": "markdown",
                        "content": f"执行步骤 ({len(turn.steps)})",
                    }
                },
                "elements": [
                    {"tag": "markdown", "content": _steps_markdown(turn.steps)}
                ],
            }
        )
    return {
        "schema": "2.0",
        "config": {"update_multi": True},
        "body": {"elements": elements},
    }


__all__ = ["FeishuAdapter", "FeishuConfig"]
