"""SessionManager — holds ``sessions: dict[session_key, AgentSession]`` (§3.3).

This is the heart of the single-process gateway: the daemon IS the SDK
process, so every chat conversation lives as an in-memory
:class:`AgentSession` keyed by its (chat-client-computed) ``session_key``.

``get_or_create`` is the only public method that matters. On the first
inbound for a ``session_key`` in this process lifetime it either resumes
(if the persistent :class:`ChatSessionMap` has a prior session_id — a
daemon-restart recovery) or creates fresh. Either way it stamps the
``wire_driver`` atom so the session's events fan out as outbound
envelopes scoped to this ``session_key``.

The wire_driver atom (§4) reaches these services by name (it cannot
import gateway modules — §11 import allow-list), so the contract is:

* ``wire_outbound``  -> ``Callable[[dict], Awaitable[None]]`` (outbound sink)
* ``session_key``    -> ``str``
* ``approval_manager`` -> :class:`ApprovalManager` (or ``None``)
* ``turn_context``   -> a mutable dict the gateway updates per turn with
  ``channel`` / ``chat_id`` / ``thread_id`` / ``sender_id`` so approval
  cards route back to the originating chat.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .approval import ApprovalManager
from .chat_session_map import ChatSessionMap
from .wire import InboundBody

logger = logging.getLogger("agentm.gateway.session_manager")

# (cwd, session_key, scenario, resume_session_id) -> AgentSession
SessionFactory = Callable[[str, str, str | None, str | None], Awaitable[Any]]

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
            sess = await self._factory(
                self._cwd, session_key, scenario, prior_session_id
            )
            new_id = _extract_session_id(sess)
            if new_id and new_id != prior_session_id:
                self._chat_map.set(session_key, new_id)

            turn_ctx: dict[str, Any] = {
                "channel": inbound.channel,
                "chat_id": inbound.chat_id,
                "thread_id": inbound.thread_id,
                "sender_id": inbound.sender_id,
            }
            self._turn_ctx[session_key] = turn_ctx

            # Stamp the wire_driver atom so this session's events fan out
            # via outbound_sink, scoped to this session_key (§3.3).
            sess.set_service("wire_outbound", self._outbound_sink)
            sess.set_service("session_key", session_key)
            sess.set_service("turn_context", turn_ctx)
            if self._approval is not None:
                sess.set_service("approval_manager", self._approval)
            sess.install_atom("wire_driver")

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

    def session_id(self, session_key: str) -> str | None:
        sess = self._sessions.get(session_key)
        return _extract_session_id(sess) if sess is not None else None

    async def shutdown_session(self, session_key: str) -> None:
        """Tear down the in-memory session. The ChatSessionMap entry STAYS
        (so /new-then-message resumes the transcript); only ``/end`` clears
        it via :meth:`forget`."""
        async with self._lock:
            sess = self._sessions.pop(session_key, None)
            self._turn_ctx.pop(session_key, None)
        if sess is not None:
            try:
                await sess.shutdown()
            except Exception:
                logger.exception("session shutdown failed for %s", session_key)

    def forget(self, session_key: str) -> None:
        """Clear the persistent ChatSessionMap entry (``/end``)."""
        self._chat_map.drop(session_key)

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
