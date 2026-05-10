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
    ToolResultEvent,
)
from agentm.core.abi.events import DiagnosticEvent, TurnEndEvent

from .approval import ApprovalBridge, ApprovalContext, ApprovalPolicy
from .bus import InboundMessage, MessageBus, OutboundMessage
from .chat_session_map import ChatSessionMap


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
                        content="Internal error handling your message; the operator has been notified.",
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
        # Approval click? Route directly to any pending bridge.
        button_value = str(msg.metadata.get("button_value") or "")
        if button_value and ":" in button_value:
            approval_id = button_value.split(":", 1)[0]
            async with self._routes_lock:
                routes = list(self._routes.values())
            for route in routes:
                if await route.bridge.resolve(
                    approval_id, value=button_value, sender_id=msg.sender_id
                ):
                    return
            logger.info(
                "approval reply %r matched no pending request", button_value
            )
            return

        logger.info("gateway: getting route for %s", msg.session_key)
        route = await self._get_or_create_route(msg)
        logger.info("gateway: prompting session=%s", msg.session_key)
        async with route.lock:
            route.approval_ctx = ApprovalContext(
                channel=msg.channel,
                chat_id=msg.chat_id,
                sender_id=msg.sender_id,
            )
            try:
                await route.session.prompt(msg.content)
            finally:
                route.approval_ctx = None

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
            ),
        )

        # ``handle_tool_call`` is the ONLY tool_call subscriber the gateway
        # registers — a separate "calling X" render would leak the intent
        # before the bridge has resolved a block. Tool outcomes surface
        # through tool_result (or the synthetic block-error rendered by
        # the tool_error_messages builtin atom).
        bus.on(ToolCallEvent.CHANNEL, route.bridge.handle_tool_call)
        bus.on(TurnEndEvent.CHANNEL, self._make_turn_end_handler(route))
        bus.on(ToolResultEvent.CHANNEL, self._make_tool_result_handler(route))
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

    def _make_tool_result_handler(
        self, route: _Route
    ) -> Callable[[ToolResultEvent], Awaitable[None]]:
        async def _h(event: ToolResultEvent) -> None:
            text = _tool_result_text(event.result)
            preview = text if len(text) <= 800 else text[:800] + "\n…(truncated)"
            icon = "❗" if event.result.is_error else "↩"
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=route.channel,
                    chat_id=route.chat_id,
                    content=f"{icon} `{event.tool_name}` →\n```\n{preview}\n```",
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


def _tool_result_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", ()) or ():
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)
