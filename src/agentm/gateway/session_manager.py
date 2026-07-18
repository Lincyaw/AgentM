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
from .child_registry import CHILD_SESSION_REGISTRY_SERVICE, ChildSessionRegistry
from .wire import InboundBody


def _available_model_names() -> list[str]:
    """Configured model-profile names (``[models.<name>]`` keys), seeded into
    the wire services so the wire_driver's session_ready frame can advertise a
    model switcher. Best-effort and gateway-side (the gateway may read user
    config; the wire_driver atom may not — §11.4.6)."""
    try:
        from agentm.core.lib.user_config import load_user_config

        return list(load_user_config().models.keys())
    except Exception:  # noqa: BLE001
        logger.debug("failed to load model profiles for wire service seeding")
        return []

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
        child_registry: ChildSessionRegistry | None = None,
    ) -> None:
        self._cwd = cwd
        self._chat_map = chat_map
        self._factory = session_factory
        self._outbound_sink = outbound_sink
        self._approval = approval_manager
        # Shared with the gateway runtime: seeded into every session's service
        # map so the sub_agent atom can register spawned children, making them
        # interactively addressable by id (see interactive-subagent design).
        self._child_registry = child_registry
        self._sessions: dict[str, Any] = {}
        # Per-session mutable turn-context dict the wire_driver reads when
        # building approval cards. Updated on each prompt.
        self._turn_ctx: dict[str, dict[str, Any]] = {}
        # Pending fork requests: session_key -> (source_session_id, up_to). Set by
        # /fork; consumed by the next get_or_create, which seeds a NEW session
        # from the source transcript instead of resuming the chat's prior id.
        self._pending_fork: dict[str, tuple[str, int | None]] = {}
        self._lock = asyncio.Lock()

    # -- public -------------------------------------------------------

    async def get_or_create(
        self, session_key: str, scenario: str | None, inbound: InboundBody,
        *, cwd: str | None = None,
    ) -> Any:
        async with self._lock:
            sess = self._sessions.get(session_key)
            if sess is not None:
                return sess

            # A pending /fork overrides the resume path: start a NEW session
            # seeded from the source transcript rather than resuming the chat's
            # prior id. resume is forced None so resolve_session_state takes the
            # fork branch; the fork params ride wire_services to the factory.
            fork = self._pending_fork.pop(session_key, None)
            prior_session_id = None if fork else self._chat_map.get(session_key)

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
                # Model-profile names the wire_driver's session_ready frame
                # advertises so a chat client can populate a model switcher.
                # Seeded here (gateway may read user config) so the atom needn't
                # import agentm.core.lib sub-modules (§11.4.6).
                "model_names": _available_model_names(),
            }
            if fork is not None:
                wire_services["fork_source"] = fork[0]
                wire_services["fork_up_to"] = fork[1]
            if self._approval is not None:
                wire_services["approval_manager"] = self._approval
            # Hand the child-session registry to the session so its sub_agent
            # atom can register spawned children for interactive addressing.
            # Absent outside the interactive gateway (sub_agent then tears
            # children down on finalize as before).
            if self._child_registry is not None:
                wire_services[CHILD_SESSION_REGISTRY_SERVICE] = self._child_registry

            effective_cwd = cwd or self._cwd
            try:
                sess = await self._factory(
                    effective_cwd, session_key, scenario, prior_session_id, wire_services
                )
            except FileNotFoundError as exc:
                missing_id = str(exc) if str(exc) else None
                if prior_session_id is None or missing_id != prior_session_id:
                    raise
                logger.warning(
                    "chat_session_map: dropping stale mapping {} -> {}",
                    session_key,
                    prior_session_id,
                )
                self._chat_map.drop(session_key)
                prior_session_id = None
                sess = await self._factory(
                    effective_cwd, session_key, scenario, None, wire_services
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

    def set_pending_fork(
        self, session_key: str, source_session_id: str, up_to: int | None
    ) -> None:
        """Arrange for the next :meth:`get_or_create` for ``session_key`` to fork
        a new session from ``source_session_id`` (up to ``up_to`` messages, or
        all when ``None``) instead of resuming the chat's prior session."""
        self._pending_fork[session_key] = (source_session_id, up_to)

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
    except Exception:  # noqa: BLE001
        logger.debug("get_session_id() raised for session object")
        return None
    return str(sid) if sid else None


__all__ = ["OutboundSink", "SessionFactory", "SessionManager"]
