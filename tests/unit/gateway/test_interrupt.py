"""Fail-stop: an interrupt inbound preempts the in-flight prompt INLINE.

If the gateway dispatched ``control="interrupt"`` as a normal detached prompt
task (or routed it through the LLM), Esc-to-interrupt would queue behind the
very turn it is meant to cancel — the interrupt would never land. The gateway
must call ``AgentSession.interrupt()`` directly, off the task path.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from agentm.gateway.child_registry import ChildSessionRegistry
from agentm.gateway.runtime import GatewayRuntime
from agentm.gateway.peer import PeerSession
from agentm.gateway.wire import WIRE_VERSION, Envelope


class _FakeSession:
    def __init__(self) -> None:
        self.interrupts = 0

    def interrupt(self) -> None:
        self.interrupts += 1


class _FakeSessions:
    def __init__(self, sess: _FakeSession) -> None:
        self._sess = sess

    def get(self, _session_key: str) -> _FakeSession:
        return self._sess


def _interrupt_inbound() -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id="i1",
        kind="inbound",
        ts=1.0,
        session_key="terminal:t1",
        body={
            "channel": "terminal",
            "chat_id": "t1",
            "sender_id": "u1",
            "content": "",
            "control": "interrupt",
        },
    )


@pytest.mark.asyncio
async def test_interrupt_inbound_calls_session_interrupt_and_spawns_no_task() -> None:
    sess = _FakeSession()
    rt = object.__new__(GatewayRuntime)
    rt._sessions = cast(Any, _FakeSessions(sess))
    rt._peer_channels = {}
    rt._inflight = set()
    rt._child_registry = ChildSessionRegistry()

    peer = PeerSession(peer_id="p1", transport_writer=cast(Any, None))
    await rt.handle_inbound(peer, _interrupt_inbound())

    assert sess.interrupts == 1  # preempted via AgentSession.interrupt()
    assert rt._inflight == set()  # NOT dispatched as a detached prompt task
