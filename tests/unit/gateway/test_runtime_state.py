from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from agentm.gateway.approval import ApprovalManager
from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.child_registry import ChildSessionRegistry
from agentm.gateway.runtime_state import GatewaySessionState
from agentm.gateway.session_manager import SessionManager


class _Sessions:
    def session_id(self, session_key: str) -> str:
        return f"id:{session_key}"


class _Approval:
    pending_count = 0

    def pending_for_session(self, session_key: str) -> list[str]:
        del session_key
        return []


class _Children:
    def ids(self) -> list[str]:
        return []


class _ChatMap:
    def snapshot(self) -> dict[str, str]:
        return {}

    def snapshot_metadata(self) -> dict[str, dict[str, Any]]:
        return {}


def _state(
    sink: Any,
) -> GatewaySessionState:
    return GatewaySessionState(
        chat_map=cast(ChatSessionMap, _ChatMap()),
        sessions=cast(SessionManager, _Sessions()),
        approval=cast(ApprovalManager, _Approval()),
        child_registry=cast(ChildSessionRegistry, _Children()),
        active_model_name=lambda _key: "model",
        active_scenario_name=lambda _key: "scenario",
        outbound_sink=sink,
    )


@pytest.mark.asyncio
async def test_snapshot_scheduling_is_retained_coalesced_and_repeatable() -> None:
    deliveries: list[dict[str, Any]] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_finished = asyncio.Event()

    async def sink(payload: dict[str, Any]) -> None:
        deliveries.append(payload)
        if len(deliveries) == 1:
            first_started.set()
            await release_first.wait()
        else:
            second_finished.set()

    state = _state(sink)
    state.record_route("chat", {"channel": "test", "chat_id": "1"})

    state.schedule_snapshot("chat")
    state.schedule_snapshot("chat")
    await asyncio.wait_for(first_started.wait(), timeout=1)
    assert len(deliveries) == 1

    state.schedule_snapshot("chat")
    assert len(deliveries) == 1

    release_first.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    state.schedule_snapshot("chat")
    await asyncio.wait_for(second_finished.wait(), timeout=1)
    assert len(deliveries) == 2


@pytest.mark.asyncio
async def test_emitted_snapshot_does_not_alias_internal_lists() -> None:
    deliveries: list[dict[str, Any]] = []

    async def sink(payload: dict[str, Any]) -> None:
        deliveries.append(payload)

    state = _state(sink)
    state.record_route("chat", {"channel": "test", "chat_id": "1"})
    state.update_snapshot(
        "chat",
        "session_ready",
        {"tool_names": ["read"], "command_names": ["help"]},
    )

    await state.emit_snapshot("chat")
    emitted_tools = deliveries[0]["metadata"]["tool_names"]
    emitted_tools.append("write")

    assert state.snapshot_for("chat").tool_names == ["read"]
