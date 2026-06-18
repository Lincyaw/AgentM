"""Fail-stop: runtime request-id semantics and submit policy.

The gateway should treat request IDs as explicit idempotency keys for mutable
inbound actions and support ``policy=="interrupt_first"`` so a new user
submission can preempt an active model run before being enqueued.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, cast

import pytest

from agentm.gateway.child_registry import ChildSessionRegistry
from agentm.gateway.approval import ApprovalManager
from agentm.gateway.runtime import GatewayRuntime
from agentm.gateway.wire import WIRE_VERSION, Envelope, InboundBody
from agentm.gateway.peer import PeerSession


class FakeInbox:
    def __init__(self, owner: "FakeSession") -> None:
        self.owner = owner
        self.items: list[Any] = []

    def push(self, item: Any) -> None:
        self.owner.events.append("push")
        self.items.append(item)


class FakeSession:
    def __init__(self) -> None:
        self.interrupts = 0
        self.events: list[str] = []
        self.inbox = FakeInbox(self)

    def interrupt(self) -> None:
        self.interrupts += 1
        self.events.append("interrupt")


class FakeSessions:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.get_or_create_calls = 0
        self.set_turn_context_calls = 0

    async def get_or_create(self, *_a: Any, **_k: Any) -> FakeSession:
        self.get_or_create_calls += 1
        return self.session

    def get(self, _session_key: str) -> FakeSession:
        return self.session

    def set_turn_context(self, _session_key: str, _inbound: InboundBody) -> None:
        self.set_turn_context_calls += 1

    def session_id(self, _session_key: str) -> str | None:
        return "sid-debug"


def _runtime() -> tuple[GatewayRuntime, FakeSessions, list[dict[str, Any]], FakeSession]:
    rt = object.__new__(GatewayRuntime)
    rt._scenario = None
    rt._child_registry = ChildSessionRegistry()
    rt._peer_channels: dict[str, str] = {}
    rt._session_commands: dict[str, set[str]] = {}
    rt._session_routes: dict[str, tuple[str, str, str | None]] = {}
    rt._snapshots: dict[str, Any] = {}
    rt._turn_counts: dict[str, int] = {}
    rt._inflight: set[Any] = set()
    rt._request_id_cache = OrderedDict[str, None]()
    rt._max_request_cache_size = 64
    outbound: list[dict[str, Any]] = []

    async def emit(body: dict[str, Any]) -> None:
        outbound.append(body)

    rt._approval = ApprovalManager(emit)
    rt._emit_outbound = emit

    session = FakeSession()
    rt._sessions = FakeSessions(session)
    return rt, cast(FakeSessions, rt._sessions), outbound, session


def _inbound(
    *,
    request_id: str | None = None,
    action: str | None = None,
    policy: str | None = None,
    content: str = "hello",
) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id="r1",
        kind="inbound",
        ts=1.0,
        session_key="terminal:t1",
        body={
            "channel": "terminal",
            "chat_id": "c1",
            "sender_id": "u1",
            "content": content,
            **({"request_id": request_id} if request_id is not None else {}),
            **({"action": action} if action is not None else {}),
            **({"policy": policy} if policy is not None else {}),
        },
    )


async def _drain_inflight(rt: GatewayRuntime) -> None:
    if not rt._inflight:
        return
    await asyncio.gather(*list(rt._inflight), return_exceptions=True)


@pytest.mark.asyncio
async def test_request_id_prevents_duplicate_submit_side_effects() -> None:
    rt, sessions, outbound, session = _runtime()

    peer = PeerSession(peer_id="p1", transport_writer=cast(Any, None))
    await rt.handle_inbound(peer, _inbound(request_id="x-1"))
    await rt.handle_inbound(peer, _inbound(request_id="x-1"))

    # Duplicate second request_id is a protocol-level duplicate; no new task is
    # scheduled, and no duplicate inbox push is delivered to the session.
    await _drain_inflight(rt)
    assert sessions.get_or_create_calls == 1
    assert len(session.inbox.items) == 1
    # Both attempts still get ack so the client can distinguish accepted vs replayed.
    assert any(body["metadata"]["request_id"] == "x-1" for body in outbound)
    assert any(body["metadata"]["status"] == "accepted" for body in outbound)
    assert any(body["metadata"]["status"] == "duplicate" for body in outbound)


@pytest.mark.asyncio
async def test_submit_interrupt_first_interrupts_before_enqueue() -> None:
    rt, _sessions, _outbound, session = _runtime()

    peer = PeerSession(peer_id="p1", transport_writer=cast(Any, None))
    await rt.handle_inbound(
        peer, _inbound(action="submit", policy="interrupt_first", request_id="x-2")
    )
    await _drain_inflight(rt)

    # interrupt must happen before the user message enters the session inbox.
    assert session.events == ["interrupt", "push"]
    assert session.interrupts == 1


@pytest.mark.asyncio
async def test_interaction_response_with_request_id_is_acknowledged() -> None:
    rt, _unused_sessions, outbound, _unused_session = _runtime()

    peer = PeerSession(peer_id="p1", transport_writer=cast(Any, None))
    await rt.handle_inbound(
        peer,
        _inbound(
            action="interaction_response",
            request_id="ix-1",
            content="",
        ),
    )
    await _drain_inflight(rt)

    assert any(
        body["metadata"]["kind"] == "request_ack"
        and body["metadata"]["request_id"] == "ix-1"
        and body["metadata"]["action"] == "interaction_response"
        for body in outbound
    )


@pytest.mark.asyncio
async def test_session_snapshot_tracks_turn_and_approval_state() -> None:
    rt, _sessions, outbound, _session = _runtime()

    changed = rt._update_session_snapshot(
        "terminal:t1",
        "session_ready",
        {"tool_names": ["git", "bash"], "command_names": ["/help", "/status"]},
    )
    assert changed is True
    snap = rt._snapshot_for("terminal:t1")
    assert snap.phase == "idle"
    assert snap.tool_names == ["bash", "git"]
    assert snap.command_names == ["/help", "/status"]

    changed = rt._update_session_snapshot(
        "terminal:t1", "turn_start", {"turn_id": "turn-1"}
    )
    snap = rt._snapshot_for("terminal:t1")
    assert changed is True
    assert snap.active_turn_id == "turn-1"
    assert snap.phase == "running"

    fake_future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    rt._approval._pending["appr-x"] = (fake_future, "u1", "terminal:t1")
    rt._update_session_snapshot("terminal:t1", "approval_request", {"approval_id": "appr-x"})
    snap = rt._snapshot_for("terminal:t1")
    assert snap.pending_interactions == ["appr-x"]
    assert snap.phase == "waiting_interaction"

    rt._approval._pending.pop("appr-x", None)
    rt._update_session_snapshot("terminal:t1", "approval_resolved", {"approval_id": "appr-x"})
    snap = rt._snapshot_for("terminal:t1")
    assert snap.pending_interactions == []
    assert snap.phase == "running"

    rt._update_session_snapshot("terminal:t1", "agent_end", {"cause": "boom"})
    snap = rt._snapshot_for("terminal:t1")
    assert snap.phase == "errored"

    rt._update_session_snapshot("terminal:t1", "agent_end", {"cause": "none"})
    snap = rt._snapshot_for("terminal:t1")
    assert snap.phase == "idle"

    rt._update_session_snapshot("terminal:t1", "turn_end", {})
    snap = rt._snapshot_for("terminal:t1")
    assert snap.active_turn_id is None
    assert snap.phase == "idle"

    # Ensure no unintended side effect is sent when only snapshot helpers run.
    assert all(item["metadata"]["kind"] != "session_snapshot" for item in outbound)


@pytest.mark.asyncio
async def test_gateway_debug_state_includes_session_and_global_projection() -> None:
    rt, _sessions, _outbound, _session = _runtime()
    rt._session_routes["terminal:t1"] = ("terminal", "chat-1", "thread-1")
    rt._session_commands["terminal:t1"] = {"help", "status", "gateway_debug"}
    rt._turn_counts["terminal:t1"] = 3
    rt._inflight.add(asyncio.create_task(asyncio.sleep(0)))
    try:
        rt._update_session_snapshot("terminal:t1", "turn_start", {"turn_id": "turn-1"})
        rt._snapshots["terminal:t1"].tool_names.extend(["bash"])
        state = rt._get_gateway_debug_state("terminal:t1")
    finally:
        pending = list(rt._inflight)
        for task in list(rt._inflight):
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert state["session_key"] == "terminal:t1"
    assert state["session"]["session_id"] == "sid-debug"
    assert state["session"]["turn_count"] == 3
    assert state["session"]["route"]["chat_id"] == "chat-1"
    assert state["sessions"]["terminal:t1"]["route"]["thread_id"] == "thread-1"
    assert state["global"]["inflight_tasks"] >= 0
