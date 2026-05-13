"""Gateway — the agent-side half of the channel system.

A long-running consumer of
``MessageBus.inbound`` that, for each message, looks up (or creates) an
:class:`agentm.harness.AgentSession` keyed on
:attr:`InboundMessage.session_key`, sets per-turn approval context,
then calls ``session.prompt(content)``. The session's ``EventBus``
emits assistant text / tool results, the gateway translates those into
:class:`OutboundMessage` and publishes back onto the bus —
:class:`agentm_channels.manager.ChannelManager` then dispatches to
whichever channel ``msg.channel`` names.

Approval clicks come back as inbound messages whose ``metadata``
carries ``button_value``; the gateway intercepts those and feeds them
to :class:`ApprovalBridge.resolve` instead of the agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.abi import (
    AssistantMessage,
    EventBus,
    TextContent,
    ToolCallEvent,
    ToolResultEvent,
)
from agentm.core.abi.events import (
    DiagnosticEvent,
    StreamDeltaEvent,
    TurnEndEvent,
)
from agentm.core.abi.stream import TextDelta

from .approval import ApprovalBridge, ApprovalContext, ApprovalPolicy
from .bus import InboundMessage, MessageBus, OutboundKind, OutboundMessage
from .chat_session_map import ChatSessionMap
from .commands import (
    CommandContext,
    CommandRegistry,
    CommandRouter,
    discover_commands,
)


_APPROVAL_VALUE_SEP = ":"  # Matches the bridge's internal encoding so we can do a hint-only id lookup.


logger = logging.getLogger(__name__)


SessionFactory = Callable[
    [str, EventBus, str | None],
    # (cwd, bus, resume_session_id)
    Awaitable[Any],  # → object with .prompt(text) + .shutdown(), and
                    # an optional .session_manager.get_session_id()
]


@dataclass
class GatewayConfig:
    cwd: str
    scenario: str | None = None
    state_dir: Path | None = None
    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    command_registry: CommandRegistry | None = None
    """Pre-built command registry. ``None`` means the gateway scans
    ``cwd`` for commands at start. Tests/embedders can pre-build a
    minimal registry to keep their environment hermetic."""

    skill_paths: list[str] = field(default_factory=list)
    """Extra skill directories beyond the defaults
    (``<cwd>/.claude/skills``, ``~/.claude/skills``)."""

    atom_commands_enabled: bool = False
    """Surface ``/atom:install`` / ``/atom:uninstall`` / ``/atom:list``
    as slash commands. Off by default — exposing kernel-level
    mutations to chat users requires explicit deployment intent."""

    atom_allow: list[str] = field(default_factory=list)
    """Atoms that may be surfaced by the ``/atom:*`` commands. Two
    gates total for any atom to appear: this list AND
    ``MANIFEST.mountable_via_command = True`` on the atom itself.
    Use ``["*"]`` in development to surface every mountable atom."""


@dataclass
class _TurnStreamState:
    """Per-turn streaming state for the "one reflowing card per turn" UX.

    Tool calls and (debounced) assistant token deltas are merged into a
    single :class:`OutboundMessage` stream keyed by ``stream_id``. Channels
    that understand ``stream_id`` (Feishu) patch a single card; channels
    that don't (terminal) see a series of MESSAGE outbounds, throttled to
    ~6/s by ``TEXT_DEBOUNCE_S``.

    The state is **not** an atom — it lives on the gateway because the
    composition decision ("tool call + token stream → one logical
    message") is a channel-presentation policy, not a kernel-level
    primitive.
    """

    stream_id: str
    channel: str
    chat_id: str
    assistant_buf: str = ""
    tool_started_at: dict[str, float] = field(default_factory=dict)
    """Per tool_call_id start timestamp, set when the ToolCallEvent
    arrives and read again on ToolResultEvent so the terminal frame
    can carry the full lifecycle (``started_at`` + ``ended_at``)."""
    tool_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per tool_call_id args snapshot. ToolResultEvent does not carry
    args, but the terminal frame must echo them so renderers can show
    one self-contained card."""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Token-delta coalescer: TextDelta handlers set ``dirty`` instead of
    # publishing directly; a background flusher publishes at most every
    # ``TEXT_DEBOUNCE_S`` so terminal renderers don't drown in updates.
    dirty: asyncio.Event = field(default_factory=asyncio.Event)
    flusher: asyncio.Task[None] | None = None
    closed: bool = False


TEXT_DEBOUNCE_S: float = 0.15


@dataclass
class _Route:
    session_key: str
    channel: str
    chat_id: str
    sender_id: str
    session: Any
    bus: EventBus  # the agent session's bus, NOT the channel MessageBus
    bridge: ApprovalBridge
    approval_ctx: ApprovalContext | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn_count: int = 0
    stream: _TurnStreamState | None = None


class Gateway:
    """MessageBus ↔ AgentSession bridge."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        config: GatewayConfig,
        session_factory: SessionFactory,
    ) -> None:
        self._bus = bus
        self._config = config
        self._session_factory = session_factory
        self._routes: dict[str, _Route] = {}
        self._routes_lock = asyncio.Lock()
        state_dir = config.state_dir or (Path(config.cwd) / ".agentm" / "channels")
        self._chat_map = ChatSessionMap(state_dir / "session_map.json")
        self._consume_task: asyncio.Task[Any] | None = None
        # Approval id → owning bridge. Each :class:`ApprovalBridge`
        # mutates this dict on request publish / resolve so button
        # clicks route in O(1). The broadcast fallback in
        # :meth:`_dispatch` survives a stale or missed entry.
        self._approval_index: dict[str, ApprovalBridge] = {}
        registry = config.command_registry or discover_commands(
            config.cwd,
            skill_paths=config.skill_paths,
            atom_commands_enabled=config.atom_commands_enabled,
            atom_allow=config.atom_allow,
        )
        self._command_router = CommandRouter(registry=registry)
        self._registry = registry

    async def start(self) -> None:
        self._consume_task = asyncio.create_task(self._consume(), name="gw-consume")

    async def stop(self) -> None:
        if self._consume_task is not None and not self._consume_task.done():
            self._consume_task.cancel()
            try:
                await self._consume_task
            except (asyncio.CancelledError, Exception):
                pass
        async with self._routes_lock:
            routes = list(self._routes.values())
            self._routes.clear()
        for route in routes:
            try:
                await route.session.shutdown()
            except Exception:
                logger.exception("session shutdown failed for %s", route.session_key)

    async def _consume(self) -> None:
        # Per-message task spawn — a session.prompt() that's awaiting an
        # approval future MUST NOT block the inbound stream, otherwise
        # the button click that would resolve the future never gets
        # routed (it sits behind the in-flight prompt). The per-route
        # lock inside ``_dispatch_user_turn`` still serializes new
        # turns within a single chat.
        while True:
            msg = await self._bus.consume_inbound()
            asyncio.create_task(
                self._dispatch_safely(msg), name=f"gw-disp-{msg.session_key}"
            )

    async def _dispatch_safely(self, msg: InboundMessage) -> None:
        try:
            await self._dispatch(msg)
        except Exception as exc:
            logger.exception("error handling inbound from %s", msg.session_key)
            err_type = type(exc).__name__
            err_msg = str(exc).strip() or "(no message)"
            # Cap to keep one envelope reasonable; full trace is in the log.
            if len(err_msg) > 800:
                err_msg = err_msg[:800] + "… (truncated; see gateway log)"
            try:
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Gateway error — {err_type}: {err_msg}",
                        metadata={
                            "error": {
                                "type": err_type,
                                "message": err_msg,
                            }
                        },
                    )
                )
            except Exception:
                logger.exception("failed to publish error reply")

    async def _dispatch(self, msg: InboundMessage) -> None:
        logger.info(
            "gateway dispatch: channel=%s session_key=%s sender=%s len=%d",
            msg.channel,
            msg.session_key,
            msg.sender_id,
            len(msg.content),
        )
        # Button click? Route O(1) through the approval index when the
        # value carries a recognizable id prefix; fall back to a full
        # broadcast on miss so a stale-index bug doesn't drop a real
        # click. The bridge owns the encoding — gateway only peeks at
        # the id prefix as a routing hint.
        if msg.button_value:
            hinted_id = msg.button_value.split(_APPROVAL_VALUE_SEP, 1)[0]
            bridge = self._approval_index.get(hinted_id)
            if bridge is not None and await bridge.try_resolve_inbound(msg):
                return
            async with self._routes_lock:
                routes = list(self._routes.values())
            for route in routes:
                if await route.bridge.try_resolve_inbound(msg):
                    return
            logger.info(
                "button click %r matched no pending request", msg.button_value
            )
            return

        # Slash command? Intercept before the LLM sees the message.
        # Control commands return early; prompt commands rewrite
        # ``msg.content`` and fall through to the normal route path so
        # the agent sees the expanded prompt as the user turn.
        if msg.content.startswith("/") and not msg.content.startswith("//"):
            cmd_msg = await self._dispatch_command(msg)
            if cmd_msg is None:
                return
            msg = cmd_msg

        logger.info("gateway: getting route for %s", msg.session_key)
        route = await self._get_or_create_route(msg)
        logger.info("gateway: prompting session=%s", msg.session_key)
        async with route.lock:
            route.approval_ctx = ApprovalContext(
                channel=msg.channel,
                chat_id=msg.chat_id,
                sender_id=msg.sender_id,
            )
            route.turn_count += 1
            route.stream = _TurnStreamState(
                stream_id=f"{route.session_key}#{route.turn_count}",
                channel=msg.channel,
                chat_id=msg.chat_id,
            )
            route.stream.flusher = asyncio.create_task(
                self._stream_flusher(route.stream),
                name=f"gw-stream-{route.session_key}-{route.turn_count}",
            )
            try:
                await route.session.prompt(msg.content)
            finally:
                route.approval_ctx = None
                # Final flush: stop the debouncer, push one last frame with
                # final=True so stream-aware channels release the card.
                # ``_finalize_stream`` is a no-op if a TurnEnd handler
                # already sent the final frame.
                await self._finalize_stream(route)
                # Typed control signal: the agent finished a turn.
                # Channels use it to tear down "thinking" affordances
                # (e.g. the Feishu ACK emoji). Channels that don't care
                # ignore it.
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        kind=OutboundKind.TURN_COMPLETE,
                    )
                )

    async def _dispatch_command(
        self, msg: InboundMessage
    ) -> InboundMessage | None:
        """Run the inbound through the command router.

        Returns ``None`` when the command was handled in full (control
        kind, or unknown/empty command — in which case a user-visible
        reply has already been published). Returns a rewritten
        :class:`InboundMessage` when a prompt command expanded into a
        new user-turn text; the caller falls through to the normal
        agent path with that message.
        """
        ctx = self._command_context_for(msg)
        result = await self._command_router.try_dispatch(msg, ctx)
        if result is None:
            return msg  # not actually a command (parse rejected); fall through unchanged
        for out in result.outbound:
            await self._bus.publish_outbound(out)
        if result.side_effect is not None:
            try:
                await result.side_effect(self)
            except Exception:
                logger.exception(
                    "command side_effect raised for %r", msg.content
                )
        if result.expanded_prompt is None:
            return None
        from dataclasses import replace

        return replace(msg, content=result.expanded_prompt)

    def _command_context_for(self, msg: InboundMessage) -> CommandContext:
        session_key = msg.session_key
        route = self._routes.get(session_key)

        async def drop_route() -> None:
            await self._drop_route(session_key)

        async def forget_chat_mapping() -> None:
            self._chat_map.drop(session_key)

        def get_stats() -> dict[str, Any]:
            current = self._routes.get(session_key)
            if current is None:
                return {
                    "session_id": None,
                    "turn_count": 0,
                    "pending_approvals": 0,
                    "_supports_forget": True,
                    "_forget_chat_mapping": forget_chat_mapping,
                }
            return {
                "session_id": self._extract_session_id(current.session),
                "turn_count": current.turn_count,
                "pending_approvals": len(current.bridge._pending),  # type: ignore[attr-defined]
                "_supports_forget": True,
                "_forget_chat_mapping": forget_chat_mapping,
            }

        def get_extension_api() -> Any | None:
            current = self._routes.get(session_key)
            if current is None:
                return None
            # AgentSession keeps the live ExtensionAPI on a private
            # attribute; no public accessor exists yet. Reach through
            # ``getattr`` so a future public ``session.extension_api``
            # property is picked up automatically if the SDK lands one.
            api = getattr(current.session, "extension_api", None)
            if api is not None:
                return api
            return getattr(current.session, "_extension_api", None)

        return CommandContext(
            route_key=session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            drop_route=drop_route,
            get_route_stats=get_stats,
            list_commands=self._registry.all,
            approval_bridge=route.bridge if route is not None else None,
            get_extension_api=get_extension_api,
        )

    async def _drop_route(self, session_key: str) -> None:
        async with self._routes_lock:
            route = self._routes.pop(session_key, None)
        if route is None:
            return
        try:
            await route.session.shutdown()
        except Exception:
            logger.exception("session shutdown failed for %s", session_key)

    async def _get_or_create_route(self, msg: InboundMessage) -> _Route:
        key = msg.session_key
        async with self._routes_lock:
            existing = self._routes.get(key)
            if existing is not None:
                return existing

        bus = EventBus()
        resume_id = self._chat_map.get(key)
        session = await self._session_factory(self._config.cwd, bus, resume_id)
        new_id = self._extract_session_id(session)
        if new_id and new_id != resume_id:
            self._chat_map.set(key, new_id)

        route = _Route(
            session_key=key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            session=session,
            bus=bus,
            bridge=ApprovalBridge(
                self._bus,
                self._config.approval_policy,
                get_context=lambda k=key: self._routes_get_ctx(k),  # type: ignore[misc]
                index=self._approval_index,
            ),
        )

        # Order matters: emit the structured tool-call envelope BEFORE
        # the approval bridge so the user sees the intent immediately;
        # otherwise the approval bridge would ``await`` on the button
        # future and block emit-loop progress for several seconds. The
        # tool-call envelopes are independent of the assistant text
        # stream (their own ``stream_id``), so renderers display them
        # as standalone cards rather than as lines inside the text bubble.
        bus.on(ToolCallEvent.CHANNEL, self._make_tool_call_renderer(route))
        bus.on(ToolCallEvent.CHANNEL, route.bridge.handle_tool_call)
        bus.on(ToolResultEvent.CHANNEL, self._make_tool_result_renderer(route))
        bus.on(StreamDeltaEvent.CHANNEL, self._make_stream_delta_handler(route))
        bus.on(TurnEndEvent.CHANNEL, self._make_turn_end_handler(route))
        bus.on(DiagnosticEvent.CHANNEL, self._make_diagnostic_handler(route))

        async with self._routes_lock:
            existing = self._routes.get(key)
            if existing is not None:
                await session.shutdown()
                return existing
            self._routes[key] = route
        return route

    def _routes_get_ctx(self, session_key: str) -> ApprovalContext | None:
        route = self._routes.get(session_key)
        return route.approval_ctx if route else None

    @staticmethod
    def _extract_session_id(session: Any) -> str | None:
        manager = getattr(session, "session_manager", None)
        if manager is None:
            return None
        getter = getattr(manager, "get_session_id", None)
        if getter is None:
            return None
        try:
            sid = getter()
        except Exception:
            return None
        return str(sid) if sid else None

    # --- per-session EventBus → MessageBus.outbound -------------------

    def _make_tool_call_renderer(
        self, route: _Route
    ) -> Callable[[ToolCallEvent], Awaitable[None]]:
        async def _h(event: ToolCallEvent) -> None:
            state = route.stream
            if state is None:
                return
            started_at = time.time()
            args = dict(event.args)
            async with state.lock:
                state.tool_started_at[event.tool_call_id] = started_at
                state.tool_args[event.tool_call_id] = args
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=state.channel,
                    chat_id=state.chat_id,
                    content="",
                    kind=OutboundKind.TOOL_CALL,
                    stream_id=f"tc-{event.tool_call_id}",
                    metadata={
                        "tool_call_id": event.tool_call_id,
                        "tool_name": event.tool_name,
                        "args": args,
                        "status": "running",
                        "started_at": started_at,
                    },
                    final=False,
                )
            )

        return _h

    def _make_tool_result_renderer(
        self, route: _Route
    ) -> Callable[[ToolResultEvent], Awaitable[None]]:
        async def _h(event: ToolResultEvent) -> None:
            state = route.stream
            if state is None:
                return
            ended_at = time.time()
            async with state.lock:
                started_at = state.tool_started_at.pop(
                    event.tool_call_id, ended_at
                )
                args = state.tool_args.pop(event.tool_call_id, {})
            result_text = _result_text(event.result)
            status = "error" if event.result.is_error else "ok"
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=state.channel,
                    chat_id=state.chat_id,
                    content="",
                    kind=OutboundKind.TOOL_CALL,
                    stream_id=f"tc-{event.tool_call_id}",
                    metadata={
                        "tool_call_id": event.tool_call_id,
                        "tool_name": event.tool_name,
                        "args": args,
                        "status": status,
                        "result_text": result_text,
                        "is_error": bool(event.result.is_error),
                        "started_at": started_at,
                        "ended_at": ended_at,
                    },
                    final=True,
                )
            )

        return _h

    def _make_stream_delta_handler(
        self, route: _Route
    ) -> Callable[[StreamDeltaEvent], Awaitable[None]]:
        async def _h(event: StreamDeltaEvent) -> None:
            state = route.stream
            if state is None:
                return
            delta = event.delta
            # Only TextDelta contributes to the visible assistant buffer.
            # ThinkingDelta is suppressed on purpose — chat surfaces
            # should not expose internal reasoning by default. Tool-call
            # frames are emitted via the dedicated ToolCallEvent handler.
            if not isinstance(delta, TextDelta):
                return
            if not delta.text:
                return
            async with state.lock:
                state.assistant_buf += delta.text
            # Mark dirty; the flusher coalesces to ~6 frames/sec.
            state.dirty.set()

        return _h

    def _make_turn_end_handler(
        self, route: _Route
    ) -> Callable[[TurnEndEvent], Awaitable[None]]:
        async def _h(event: TurnEndEvent) -> None:
            state = route.stream
            text = _assistant_text(event.message)
            if state is None:
                # Stream lifecycle missed (test fixtures driving the
                # session bus directly outside a _dispatch). Preserve
                # the legacy one-shot behaviour so test_gateway_e2e
                # still sees its "echo: …" outbound.
                if not text.strip():
                    return
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=route.channel,
                        chat_id=route.chat_id,
                        content=text,
                    )
                )
                return
            async with state.lock:
                # The fully-assembled assistant text from TurnEndEvent
                # is authoritative; replace any partial we accumulated
                # from StreamDelta. Empty assistant text is normal for
                # tool-only turns — render whatever tool lines we have.
                state.assistant_buf = text
                state.closed = True
                content = _render(state)
            await self._publish_stream_frame(state, content, final=True)

        return _h

    def _make_diagnostic_handler(
        self, route: _Route
    ) -> Callable[[DiagnosticEvent], Awaitable[None]]:
        async def _h(event: DiagnosticEvent) -> None:
            severity = getattr(event, "severity", "info")
            if severity not in {"warning", "error"}:
                return
            message = getattr(event, "message", "")
            if not message:
                return
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=route.channel,
                    chat_id=route.chat_id,
                    content=f"⚠ {severity}: {message}",
                )
            )

        return _h

    # --- streaming card lifecycle ------------------------------------

    async def _publish_stream_frame(
        self, state: _TurnStreamState, content: str, *, final: bool
    ) -> None:
        if not content and not final:
            return
        await self._bus.publish_outbound(
            OutboundMessage(
                channel=state.channel,
                chat_id=state.chat_id,
                content=content,
                stream_id=state.stream_id,
                final=final,
            )
        )

    async def _stream_flusher(self, state: _TurnStreamState) -> None:
        """Coalesce TextDelta-driven dirty marks into ≤1 frame per
        ``TEXT_DEBOUNCE_S`` window.

        Tool-call/result handlers publish synchronously and **do not**
        go through the flusher — token streams are the only producer
        dense enough to need throttling. The flusher exits as soon as
        the turn closes; TurnEnd is responsible for the final frame.
        """
        try:
            while not state.closed:
                await state.dirty.wait()
                state.dirty.clear()
                if state.closed:
                    return
                await asyncio.sleep(TEXT_DEBOUNCE_S)
                async with state.lock:
                    if state.closed:
                        return
                    content = _render(state)
                await self._publish_stream_frame(state, content, final=False)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception(
                "stream flusher crashed (stream_id=%s)", state.stream_id
            )

    async def _finalize_stream(self, route: _Route) -> None:
        """Tear down the per-turn stream state.

        TurnEnd is the normal "final frame" producer; we only get here
        from the ``_dispatch`` finally, so by the time this runs the
        TurnEnd handler has either fired (and already pushed final=True)
        or the turn aborted before it could. In the abort path we still
        push one explicit final frame so stream-aware channels release
        the card lock.
        """
        state = route.stream
        route.stream = None
        if state is None:
            return
        async with state.lock:
            had_content = bool(state.assistant_buf)
            already_finalized = state.closed
            state.closed = True
        state.dirty.set()  # unblock the flusher if parked
        if state.flusher is not None and not state.flusher.done():
            state.flusher.cancel()
            try:
                await state.flusher
            except (asyncio.CancelledError, Exception):
                pass
        if already_finalized or not had_content:
            return
        async with state.lock:
            content = _render(state)
        await self._publish_stream_frame(state, content, final=True)


def _assistant_text(message: AssistantMessage) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent) and block.text
    )


def _render(state: _TurnStreamState) -> str:
    """Assemble the streamed-card body from per-turn state.

    Only the assistant text stream lives here; structured tool-call
    envelopes are emitted on their own per-call ``stream_id`` and never
    fold into this body.
    """
    return state.assistant_buf


_RESULT_TEXT_CAP = 32 * 1024


def _result_text(result: Any, *, limit: int = _RESULT_TEXT_CAP) -> str:
    """Concatenate the full text of a ToolResult.

    Joins every non-empty ``TextContent`` block with a newline; emits
    ``<image>`` for image blocks. Caps the total at ``limit`` bytes to
    bound memory in pathological cases — full multi-line stdout still
    survives; only multi-MB blobs get trimmed.
    """
    blocks = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
            continue
        if getattr(block, "type", None) == "image":
            parts.append("<image>")
    joined = "\n".join(parts)
    if len(joined) > limit:
        joined = joined[:limit] + f"\n…(truncated at {limit} bytes)"
    return joined


