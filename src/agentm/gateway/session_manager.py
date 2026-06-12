"""SessionManager — holds ``sessions: dict[session_key, AgentSession]`` (§3.3).

This is the heart of the single-process gateway: the daemon IS the SDK
process, so every chat conversation lives as an in-memory
:class:`AgentSession` keyed by its (chat-client-computed) ``session_key``.

``get_or_create`` is the only public method that matters. On the first
inbound for a ``session_key`` in this process lifetime it either recovers
(if the persistent :class:`ChatSessionMap` has a prior session_id — a
daemon-restart recovery) or creates fresh. Either way it stamps the
``wire_driver`` atom so the session's events fan out as outbound
envelopes scoped to this ``session_key``.

The wire_driver atom (§4) reaches these services by name (it cannot
import gateway modules — import allow-list), so the contract is:

* ``wire_outbound``  -> ``Callable[[dict], Awaitable[None]]`` (outbound sink)
* ``session_key``    -> ``str``
* ``approval_manager`` -> :class:`ApprovalManager` (or ``None``)
* ``turn_context``   -> a mutable dict the gateway updates per turn with
  ``channel`` / ``chat_id`` / ``thread_id`` / ``sender_id`` so approval
  cards route back to the originating chat.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from .approval import ApprovalManager
from .chat_session_map import ChatSessionMap
from .wire import InboundBody

# (cwd, session_key, scenario, resume_session_id, wire_services) -> AgentSession
#
# ``wire_services`` is seeded into the session's service registry BEFORE any
# atom installs (via ``AgentSessionConfig.initial_services``) and carries the
# ``wire_outbound`` / ``session_key`` / ``turn_context`` / ``approval_manager``
# the ``wire_driver`` atom reads at install time. The factory is responsible
# for listing ``wire_driver`` in the config so it installs DURING ``create()``
# and is subscribed in time to forward the creation-time ``SessionReadyEvent``
# (the chat client's slash-command catalog) — stamping it after ``create()``
# returns drops that first frame.
SessionFactory = Callable[
    [str, str, str | None, str | None, dict[str, Any]], Awaitable[Any]
]

OutboundSink = Callable[[dict[str, Any]], Awaitable[None]]


class SessionManager:
    """In-memory ``session_key -> AgentSession`` registry (§3.3)."""

    def __init__(
        self,
        *,
        cwd: str,
        chat_map: ChatSessionMap,
        session_factory: SessionFactory,
        outbound_sink: OutboundSink,
        approval_manager: ApprovalManager | None = None,
    ) -> None:
        self._cwd = cwd
        self._chat_map = chat_map
        self._factory = session_factory
        self._outbound_sink = outbound_sink
        self._approval = approval_manager
        self._sessions: dict[str, Any] = {}
        # Per-session mutable turn-context dict the wire_driver reads when
        # building approval cards. Updated on each prompt.
        self._turn_ctx: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    # -- public -------------------------------------------------------

    async def get_or_create(
        self, session_key: str, scenario: str | None, inbound: InboundBody
    ) -> Any:
        async with self._lock:
            sess = self._sessions.get(session_key)
            if sess is not None:
                return sess

            prior_session_id = self._chat_map.get(session_key)

            turn_ctx: dict[str, Any] = {
                "channel": inbound.channel,
                "chat_id": inbound.chat_id,
                "thread_id": inbound.thread_id,
                "sender_id": inbound.sender_id,
            }
            self._turn_ctx[session_key] = turn_ctx

            # Hand the wire_driver's services to the factory so they are seeded
            # into the service registry BEFORE atoms install. The factory mounts
            # wire_driver during create(), so it is subscribed in time to
            # forward the creation-time SessionReadyEvent (the chat client's
            # slash-command catalog). Stamping after create() returns dropped
            # that first frame (§3.3).
            wire_services: dict[str, Any] = {
                "wire_outbound": self._outbound_sink,
                "session_key": session_key,
                "turn_context": turn_ctx,
            }
            if self._approval is not None:
                wire_services["approval_manager"] = self._approval

            sess = await self._factory(
                self._cwd, session_key, scenario, prior_session_id, wire_services
            )
            new_id = _extract_session_id(sess)
            if new_id and new_id != prior_session_id:
                self._chat_map.set(session_key, new_id)

            self._sessions[session_key] = sess
            return sess

    def set_turn_context(self, session_key: str, inbound: InboundBody) -> None:
        """Update the per-turn routing context (mutated in place so the
        wire_driver's captured reference sees the new turn)."""
        ctx = self._turn_ctx.get(session_key)
        if ctx is None:
            return
        ctx["channel"] = inbound.channel
        ctx["chat_id"] = inbound.chat_id
        ctx["thread_id"] = inbound.thread_id
        ctx["sender_id"] = inbound.sender_id

    def get(self, session_key: str) -> Any | None:
        return self._sessions.get(session_key)

    def set_factory(self, factory: SessionFactory) -> None:
        """Swap the session factory (e.g. after ``/model``). Live sessions keep
        the model they were built with until torn down; the next
        ``get_or_create`` uses the new factory. Pair with
        :meth:`shutdown_session` to make a model switch take effect on the
        current chat's next message (transcript resumes)."""
        self._factory = factory

    def session_id(self, session_key: str) -> str | None:
        """Resolve the session id for ``session_key``.

        Prefer a live in-memory session's id; fall back to the persisted
        ChatSessionMap entry when none is live. The fallback matters after a
        gateway restart: ``self._sessions`` is empty and a slash COMMAND
        (unlike a normal message) never calls :meth:`get_or_create`, so
        ``/context`` would otherwise report "no active session" even though
        the conversation has persisted history. The chat->session mapping
        survives restarts (``get_or_create`` reads it as ``prior_session_id``),
        so it can resolve the trace file without first sending a message.
        Returns ``None`` only when neither a live session nor a map entry
        exists."""
        sess = self._sessions.get(session_key)
        if sess is not None:
            return _extract_session_id(sess)
        return self._chat_map.get(session_key)

    async def shutdown_session(self, session_key: str) -> None:
        """Tear down the in-memory session. Call :meth:`forget` afterwards
        to also clear the persistent ``ChatSessionMap`` entry."""
        async with self._lock:
            sess = self._sessions.pop(session_key, None)
            self._turn_ctx.pop(session_key, None)
        if sess is not None:
            try:
                await sess.shutdown()
            except Exception:
                logger.exception(f"session shutdown failed for {session_key}")

    def forget(self, session_key: str) -> None:
        """Clear the persistent ChatSessionMap entry."""
        self._chat_map.drop(session_key)

    def set_chat_mapping(self, session_key: str, session_id: str) -> None:
        """Point the persistent ChatSessionMap entry to ``session_id``."""
        self._chat_map.set(session_key, session_id)

    async def shutdown_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._turn_ctx.clear()
        for sess in sessions:
            try:
                await sess.shutdown()
            except Exception:
                logger.exception("session shutdown failed during shutdown_all")


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


__all__ = ["OutboundSink", "SessionFactory", "SessionManager"]
