"""FeishuGateway — composes a :class:`ChatSource` with an
:class:`agentm.harness.AgentSession` per chat / thread.

Lifecycle::

    gateway = FeishuGateway(source=stub_or_feishu, scenario="feishu_chat", ...)
    await gateway.run()  # blocks until source closes / cancellation

Internally the gateway runs three concurrent tasks:

- ``_consume_messages``   pulls inbound user messages → ``session.prompt``
- ``_consume_card_actions`` routes button clicks → ``ApprovalBridge``
- ``_render_*`` handlers (registered on the per-session ``EventBus``)
  translate kernel events into ``ChatSource`` calls.

One :class:`AgentSession` per ``(chat_id, thread_id)`` route. The map is
persisted to disk so gateway restarts can resume the same agent for a
chat — see :class:`ChatSessionMap`.
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
from .chat_session_map import ChatSessionMap
from .chat_source import ChatSource, InboundMessage


logger = logging.getLogger(__name__)


# Type for the session factory the gateway uses. Indirected so tests can
# inject a fake without needing AgentSession.
SessionFactory = Callable[
    [str, EventBus, str | None],  # (cwd, bus, resume_session_id)
    Awaitable[Any],  # returns an object with ``.prompt(text)`` + ``.shutdown()``
]


@dataclass
class GatewayConfig:
    cwd: str
    """Working directory passed to each AgentSession."""

    scenario: str | None = None
    """Scenario name to mount on each session (e.g. ``feishu_chat``)."""

    extra_extensions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """Atoms mounted on top of the scenario."""

    state_dir: Path | None = None
    """Where to persist the chat→session map. Defaults to ``<cwd>/.agentm/feishu/``."""

    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)

    response_in_card: bool = False
    """When True, stream assistant text into a single updating card.
    When False (default), post each assistant turn as a plain text reply —
    simpler and more robust against card-template drift."""


@dataclass
class _ChatRoute:
    """Per-chat state held by the gateway."""

    chat_id: str
    thread_id: str | None
    session: Any
    bus: EventBus
    bridge: ApprovalBridge
    approval_ctx: ApprovalContext | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """One prompt-at-a-time per route — Feishu users may double-tap; we
    serialize so the second message queues behind the first turn."""


class FeishuGateway:
    """Long-running daemon mediating between Feishu chats and AgentM."""

    def __init__(
        self,
        *,
        source: ChatSource,
        config: GatewayConfig,
        session_factory: SessionFactory,
    ) -> None:
        self._source = source
        self._config = config
        self._session_factory = session_factory
        self._routes: dict[tuple[str, str | None], _ChatRoute] = {}
        self._routes_lock = asyncio.Lock()
        state_dir = config.state_dir or (Path(config.cwd) / ".agentm" / "feishu")
        self._chat_map = ChatSessionMap(state_dir / "chat_sessions.json")
        self._stopping = False
        self._tasks: list[asyncio.Task[Any]] = []

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        await self._source.connect()
        try:
            self._tasks = [
                asyncio.create_task(self._consume_messages(), name="feishu-msgs"),
                asyncio.create_task(self._consume_card_actions(), name="feishu-cards"),
            ]
            done, pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in pending:
                task.cancel()
            for task in done:
                # Surface the first exception so the caller sees the failure.
                exc = task.exception()
                if exc is not None:
                    raise exc
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        async with self._routes_lock:
            routes = list(self._routes.values())
            self._routes.clear()
        for route in routes:
            try:
                await route.session.shutdown()
            except Exception:
                logger.exception("session shutdown failed for chat %s", route.chat_id)
        try:
            await self._source.close()
        except Exception:
            logger.exception("source close failed")

    # ------------------------------------------------------------------- consumers

    async def _consume_messages(self) -> None:
        async for msg in self._source.messages():
            try:
                await self._handle_message(msg)
            except Exception:
                logger.exception("error handling message in chat %s", msg.chat_id)
                try:
                    await self._source.send_text(
                        msg.chat_id,
                        "Internal error handling your message; the operator has been notified.",
                        thread_id=msg.thread_id,
                    )
                except Exception:
                    logger.exception("failed to send error reply")

    async def _consume_card_actions(self) -> None:
        async for action in self._source.card_actions():
            handled = False
            async with self._routes_lock:
                routes = list(self._routes.values())
            for route in routes:
                try:
                    if await route.bridge.resolve(action):
                        handled = True
                        break
                except Exception:
                    logger.exception("approval resolve failed for card %s", action.card_id)
            if not handled:
                logger.info(
                    "card action %s on %s did not match any pending approval",
                    action.action,
                    action.card_id,
                )

    # ------------------------------------------------------------------- per-message

    async def _handle_message(self, msg: InboundMessage) -> None:
        route = await self._get_or_create_route(msg)
        async with route.lock:
            route.approval_ctx = ApprovalContext(
                chat_id=msg.chat_id,
                user_id=msg.user_id,
                thread_id=msg.thread_id,
            )
            try:
                await route.session.prompt(msg.text)
            finally:
                route.approval_ctx = None

    async def _get_or_create_route(self, msg: InboundMessage) -> _ChatRoute:
        key = (msg.chat_id, msg.thread_id)
        async with self._routes_lock:
            existing = self._routes.get(key)
            if existing is not None:
                return existing

        bus = EventBus()
        resume_id = self._chat_map.get(msg.chat_id, msg.thread_id)
        session = await self._session_factory(self._config.cwd, bus, resume_id)
        new_id = self._extract_session_id(session)
        if new_id and new_id != resume_id:
            self._chat_map.set(msg.chat_id, new_id, msg.thread_id)

        route = _ChatRoute(
            chat_id=msg.chat_id,
            thread_id=msg.thread_id,
            session=session,
            bus=bus,
            bridge=ApprovalBridge(
                self._source,
                self._config.approval_policy,
                get_context=lambda r=None: self._routes_get_ctx(msg.chat_id, msg.thread_id),  # type: ignore[misc]
            ),
        )

        # ``handle_tool_call`` MUST be the only ``tool_call`` subscriber the
        # gateway adds: a downstream "calling bash" render would also fire
        # when the bridge ends up blocking the call, leaking the intent
        # before the user has seen the approval card resolve. The tool's
        # outcome surfaces through ``tool_result`` (or the synthetic block
        # error rendered by the ``tool_error_messages`` builtin atom).
        bus.on(ToolCallEvent.CHANNEL, route.bridge.handle_tool_call)
        bus.on(TurnEndEvent.CHANNEL, self._make_turn_end_handler(route))
        bus.on(ToolResultEvent.CHANNEL, self._make_tool_result_renderer(route))
        bus.on(DiagnosticEvent.CHANNEL, self._make_diagnostic_renderer(route))

        async with self._routes_lock:
            existing = self._routes.get(key)
            if existing is not None:
                # Lost the race; throw away the one we just made.
                await session.shutdown()
                return existing
            self._routes[key] = route
        return route

    def _routes_get_ctx(
        self, chat_id: str, thread_id: str | None
    ) -> ApprovalContext | None:
        route = self._routes.get((chat_id, thread_id))
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

    # ------------------------------------------------------------------- renderers

    def _make_turn_end_handler(
        self, route: _ChatRoute
    ) -> Callable[[TurnEndEvent], Awaitable[None]]:
        async def _handler(event: TurnEndEvent) -> None:
            text = _assistant_text(event.message)
            if not text.strip():
                return
            try:
                await self._source.send_text(
                    route.chat_id, text, thread_id=route.thread_id
                )
            except Exception:
                logger.exception("send assistant text failed")

        return _handler

    def _make_tool_result_renderer(
        self, route: _ChatRoute
    ) -> Callable[[ToolResultEvent], Awaitable[None]]:
        async def _handler(event: ToolResultEvent) -> None:
            text = _tool_result_text(event.result)
            preview = text if len(text) <= 800 else text[:800] + "\n…(truncated)"
            icon = "❗" if event.result.is_error else "↩"
            try:
                await self._source.send_text(
                    route.chat_id,
                    f"{icon} `{event.tool_name}` →\n```\n{preview}\n```",
                    thread_id=route.thread_id,
                )
            except Exception:
                logger.exception("tool_result render failed")

        return _handler

    def _make_diagnostic_renderer(
        self, route: _ChatRoute
    ) -> Callable[[DiagnosticEvent], Awaitable[None]]:
        async def _handler(event: DiagnosticEvent) -> None:
            severity = getattr(event, "severity", "info")
            if severity not in {"warning", "error"}:
                return
            message = getattr(event, "message", "")
            if not message:
                return
            try:
                await self._source.send_text(
                    route.chat_id,
                    f"⚠ {severity}: {message}",
                    thread_id=route.thread_id,
                )
            except Exception:
                logger.exception("diagnostic render failed")

        return _handler


def _assistant_text(message: AssistantMessage) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "\n".join(p for p in parts if p)


def _tool_result_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", ()) or ():
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)
