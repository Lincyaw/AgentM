"""Gateway — the agent-side half of the channel system.

Mirrors HKUDS/nanobot's flow: a long-running consumer of
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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.abi import (
    AssistantMessage,
    EventBus,
    TextContent,
    ToolCallEvent,
)
from agentm.core.abi.events import DiagnosticEvent, TurnEndEvent

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
            config.cwd, skill_paths=config.skill_paths
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
        except Exception:
            logger.exception("error handling inbound from %s", msg.session_key)
            try:
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=(
                            "Sorry — your message could not be processed. "
                            "The error has been logged; please retry or contact "
                            "the operator if it keeps happening."
                        ),
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
            try:
                await route.session.prompt(msg.content)
            finally:
                route.approval_ctx = None
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

        return CommandContext(
            route_key=session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            drop_route=drop_route,
            get_route_stats=get_stats,
            list_commands=self._registry.all,
            approval_bridge=route.bridge if route is not None else None,
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

        # ``handle_tool_call`` is the ONLY tool_call subscriber: a
        # separate "calling X" render would leak the intent before the
        # bridge has resolved a block. Tool *results* are intentionally
        # not relayed to the chat — they land in the trajectory for
        # operators, but pushing them back into the chat clutters the
        # conversation and exposes raw payloads. The agent's next
        # assistant turn summarizes whatever the user needs to know.
        bus.on(ToolCallEvent.CHANNEL, route.bridge.handle_tool_call)
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

    def _make_turn_end_handler(
        self, route: _Route
    ) -> Callable[[TurnEndEvent], Awaitable[None]]:
        async def _h(event: TurnEndEvent) -> None:
            text = _assistant_text(event.message)
            if not text.strip():
                return
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=route.channel,
                    chat_id=route.chat_id,
                    content=text,
                )
            )

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


def _assistant_text(message: AssistantMessage) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent) and block.text
    )
