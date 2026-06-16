"""Fail-stop: a child-addressed inbound reaches the child's inbox, not a chat.

Interactive sub-agents (``.claude/designs/interactive-subagent.md``) rely on the
gateway routing an inbound whose ``session_key`` names a *registered child* to
that child's :class:`SessionInbox` as ``source="user"`` — never through the
``get_or_create`` chat-map path (which would mint a *new* chat session keyed by
the child's id and silently drop the human's message). This locks that fork down,
plus the interrupt twin.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from agentm.core.runtime.session_inbox import SessionInbox
from agentm.gateway.child_registry import ChildSessionRegistry
from agentm.gateway.peer import PeerSession
from agentm.gateway.runtime import GatewayRuntime
from agentm.gateway.wire import WIRE_VERSION, Envelope


class _FakeChild:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.inbox = SessionInbox()
        self.interrupts = 0

    def interrupt(self) -> None:
        self.interrupts += 1


class _ExplodingSessions:
    """Any chat-session access here means the child branch was NOT taken."""

    def get(self, _session_key: str) -> Any:  # pragma: no cover - guard
        raise AssertionError("child inbound must not touch the chat SessionManager")

    async def get_or_create(self, *_a: Any, **_k: Any) -> Any:  # pragma: no cover
        raise AssertionError("child inbound must not create a chat session")


def _child_inbound(child_id: str, *, content: str, control: str = "") -> Envelope:
    body: dict[str, Any] = {
        "channel": "terminal",
        "chat_id": "t1",
        "sender_id": "u1",
        "content": content,
    }
    if control:
        body["control"] = control
    return Envelope(
        v=WIRE_VERSION,
        id="c1",
        kind="inbound",
        ts=1.0,
        session_key=child_id,
        body=body,
    )


def _runtime(registry: ChildSessionRegistry) -> GatewayRuntime:
    rt = object.__new__(GatewayRuntime)
    rt._sessions = cast(Any, _ExplodingSessions())
    rt._peer_channels = {}
    rt._inflight = set()
    rt._child_registry = registry
    return rt


@pytest.mark.asyncio
async def test_child_inbound_lands_in_child_inbox() -> None:
    child = _FakeChild("child-abc")
    registry = ChildSessionRegistry()
    registry.register(child)
    rt = _runtime(registry)

    peer = PeerSession(peer_id="p1", transport_writer=cast(Any, None))
    await rt.handle_inbound(peer, _child_inbound("child-abc", content="hello sub"))

    items = child.inbox.drain()
    assert len(items) == 1
    assert items[0].source == "user"
    assert items[0].payload == "hello sub"
    assert rt._inflight == set()  # never dispatched as a chat prompt task


@pytest.mark.asyncio
async def test_child_interrupt_routes_to_child() -> None:
    child = _FakeChild("child-abc")
    registry = ChildSessionRegistry()
    registry.register(child)
    rt = _runtime(registry)

    peer = PeerSession(peer_id="p1", transport_writer=cast(Any, None))
    await rt.handle_inbound(
        peer, _child_inbound("child-abc", content="", control="interrupt")
    )

    assert child.interrupts == 1
    assert child.inbox.is_empty()


@pytest.mark.asyncio
async def test_blank_child_input_is_dropped() -> None:
    child = _FakeChild("child-abc")
    registry = ChildSessionRegistry()
    registry.register(child)
    rt = _runtime(registry)

    peer = PeerSession(peer_id="p1", transport_writer=cast(Any, None))
    await rt.handle_inbound(peer, _child_inbound("child-abc", content="   "))

    assert child.inbox.is_empty()  # whitespace-only never enqueues a turn
