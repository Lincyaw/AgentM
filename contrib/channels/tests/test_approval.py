from __future__ import annotations

import asyncio

import pytest

from agentm.core.abi import ToolCallEvent

from agentm_channels.approval import ApprovalBridge, ApprovalContext, ApprovalPolicy
from agentm_channels.bus import MessageBus


def _make(policy: ApprovalPolicy, ctx: ApprovalContext | None = None) -> tuple[ApprovalBridge, MessageBus]:
    bus = MessageBus()
    bridge = ApprovalBridge(bus, policy, get_context=lambda: ctx)
    return bridge, bus


def _tc(name: str = "bash") -> ToolCallEvent:
    return ToolCallEvent(tool_call_id="t1", tool_name=name, args={"cmd": "echo hi"})


@pytest.mark.asyncio
async def test_always_allow_no_request_published() -> None:
    bridge, bus = _make(ApprovalPolicy(always_allow=frozenset({"bash"})))
    assert await bridge.handle_tool_call(_tc()) is None
    assert bus.outbound.qsize() == 0


@pytest.mark.asyncio
async def test_always_block_short_circuits() -> None:
    bridge, bus = _make(ApprovalPolicy(always_block=frozenset({"bash"})))
    result = await bridge.handle_tool_call(_tc())
    assert result is not None and result["block"] is True
    assert bus.outbound.qsize() == 0


@pytest.mark.asyncio
async def test_approve_flow_publishes_card_then_resolves() -> None:
    ctx = ApprovalContext(channel="stub", chat_id="c", sender_id="u")
    bridge, bus = _make(
        ApprovalPolicy(require_approval=frozenset({"bash"}), timeout_seconds=2.0),
        ctx,
    )

    async def click_approve() -> None:
        # Wait for the request card.
        for _ in range(200):
            if bus.outbound.qsize() > 0:
                msg = await bus.consume_outbound()
                if msg.metadata.get("kind") == "approval_request":
                    approve_value = msg.buttons[0][1]  # ["Approve", "<id>:approve"]
                    approval_id = approve_value.split(":", 1)[0]
                    await bridge.resolve(
                        approval_id, value=approve_value, sender_id="u"
                    )
                    return
            await asyncio.sleep(0.01)

    clicker = asyncio.create_task(click_approve())
    result = await bridge.handle_tool_call(_tc())
    await clicker
    assert result is None
    # Resolution card was also published.
    assert bus.outbound.qsize() >= 1


@pytest.mark.asyncio
async def test_deny_flow_blocks() -> None:
    ctx = ApprovalContext(channel="stub", chat_id="c", sender_id="u")
    bridge, bus = _make(
        ApprovalPolicy(require_approval=frozenset({"bash"}), timeout_seconds=2.0),
        ctx,
    )

    async def click_deny() -> None:
        for _ in range(200):
            if bus.outbound.qsize() > 0:
                msg = await bus.consume_outbound()
                if msg.metadata.get("kind") == "approval_request":
                    deny_value = msg.buttons[1][1]
                    approval_id = deny_value.split(":", 1)[0]
                    await bridge.resolve(approval_id, value=deny_value, sender_id="u")
                    return
            await asyncio.sleep(0.01)

    clicker = asyncio.create_task(click_deny())
    result = await bridge.handle_tool_call(_tc())
    await clicker
    assert result is not None and result["block"] is True
    assert "denied by u" in result["reason"]


@pytest.mark.asyncio
async def test_timeout_blocks() -> None:
    ctx = ApprovalContext(channel="stub", chat_id="c", sender_id="u")
    bridge, _bus = _make(
        ApprovalPolicy(require_approval=frozenset({"bash"}), timeout_seconds=0.05),
        ctx,
    )
    result = await bridge.handle_tool_call(_tc())
    assert result is not None and result["block"] is True
    assert "timed out" in result["reason"]


@pytest.mark.asyncio
async def test_other_user_click_is_ignored_until_rightful_user_acts() -> None:
    ctx = ApprovalContext(channel="stub", chat_id="c", sender_id="u")
    bridge, bus = _make(
        ApprovalPolicy(require_approval=frozenset({"bash"}), timeout_seconds=2.0),
        ctx,
    )

    async def attacker_then_owner() -> None:
        for _ in range(200):
            if bus.outbound.qsize() > 0:
                msg = await bus.consume_outbound()
                if msg.metadata.get("kind") != "approval_request":
                    continue
                approve_value = msg.buttons[0][1]
                approval_id = approve_value.split(":", 1)[0]
                # Outsider — must NOT consume.
                ok = await bridge.resolve(
                    approval_id, value=approve_value, sender_id="someone-else"
                )
                assert ok is False
                # Rightful user — should resolve.
                ok = await bridge.resolve(
                    approval_id, value=approve_value, sender_id="u"
                )
                assert ok is True
                return
            await asyncio.sleep(0.01)

    clicker = asyncio.create_task(attacker_then_owner())
    result = await bridge.handle_tool_call(_tc())
    await clicker
    assert result is None
