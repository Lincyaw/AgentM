"""ChildSessionRegistry — addressable live sub-agent sessions (§interactive-subagent).

A spawned sub-agent *is* a complete :class:`AgentSession` with its own bus,
driver and :class:`SessionInbox`. The single-process gateway already forwards a
child's trajectory onto the parent wire (``child_wire_forwarder``); this registry
is the symmetric *inbound* path: it lets the gateway address a live child by its
``session_id`` so a human's message can be delivered to that child's inbox — the
same caller-agnostic seam the main agent uses when it ``inject_instruction``s.

Design: ``.claude/designs/interactive-subagent.md``. The registry is plumbing,
not policy — it holds in-memory child sessions keyed by id, populated by the
``sub_agent`` atom (which reaches it by name as the ``child_session_registry``
service, never by import — §11). The gateway runtime reads it to decide whether
an inbound ``session_key`` names a child (route to its inbox) or an ordinary
chat session (the usual ``get_or_create`` path).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Service name under which the gateway seeds this registry into a session's
# service map. Child-spawning atoms reach it via ``api.get_service`` (§11: no
# atom-to-atom import) and register each freshly spawned child.
CHILD_SESSION_REGISTRY_SERVICE = "child_session_registry"


class ChildSessionRegistry:
    """In-memory ``session_id -> child AgentSession`` map (live children only).

    A child is registered the moment ``sub_agent`` spawns it and stays
    registered — and its session alive — until the owning parent session is
    torn down, so a human can keep chatting with it after its dispatched task
    finishes (the "continue after death" case in the design). Membership is the
    discriminator the gateway uses to route an inbound: a ``session_key`` in
    this registry is a child to deliver to; anything else is an ordinary chat.
    """

    def __init__(self) -> None:
        self._children: dict[str, Any] = {}

    def register(self, session: Any) -> None:
        """Register a live child session, keyed by its ``session_id``.

        Silently skips a child that does not expose a truthy ``session_id`` —
        the same defensiveness the wire forwarder applies, so a malformed child
        object degrades to "not interactively addressable" rather than raising
        inside the spawn hot path."""
        sid = getattr(session, "session_id", None)
        if not sid:
            logger.debug("child registry: skipping child with no session_id")
            return
        self._children[str(sid)] = session

    def deregister(self, session_id: str) -> None:
        self._children.pop(session_id, None)

    def get(self, session_id: str) -> Any | None:
        return self._children.get(session_id)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._children

    def ids(self) -> list[str]:
        return list(self._children)

    def values(self) -> list[Any]:
        return list(self._children.values())


__all__ = ["CHILD_SESSION_REGISTRY_SERVICE", "ChildSessionRegistry"]
