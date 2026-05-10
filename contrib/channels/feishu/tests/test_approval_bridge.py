from __future__ import annotations

import asyncio

import pytest

from agentm.core.abi import ToolCallEvent

from agentm_feishu.approval import ApprovalBridge, ApprovalContext, ApprovalPolicy
from agentm_feishu.chat_source import CardActionEvent, StubChatSource


def _make_bridge(
    policy: ApprovalPolicy, ctx: ApprovalContext | None = ApprovalContext("c", "u")
) -> tuple[ApprovalBridge, StubChatSource]:
    src = StubChatSource()
    bridge = ApprovalBridge(src, policy, get_context=lambda: ctx)
    return bridge, src


def _tc(name: str = "lark", args: dict | None = None) -> ToolCallEvent:
    return ToolCallEvent(tool_call_id="t1", tool_name=name, args=args or {"argv": ["status"]})


@pytest.mark.asyncio
async def test_always_allow_returns_none_and_sends_no_card() -> None:
    bridge, src = _make_bridge(ApprovalPolicy(always_allow=frozenset({"lark"})))
    assert await bridge.handle_tool_call(_tc()) is None
    assert src.outbox == []


@pytest.mark.asyncio
async def test_always_block_short_circuits() -> None:
    bridge, src = _make_bridge(ApprovalPolicy(always_block=frozenset({"lark"})))
    result = await bridge.handle_tool_call(_tc())
    assert result == {"block": True, "reason": "tool 'lark' is denied by gateway policy"}
    assert src.outbox == []


@pytest.mark.asyncio
async def test_approve_flow_sends_card_then_updates_to_resolved() -> None:
    bridge, src = _make_bridge(
        ApprovalPolicy(require_approval=frozenset({"lark"}), timeout_seconds=2.0)
    )

    async def click_approve() -> None:
        await asyncio.sleep(0.01)
        # The bridge minted a card_id from the future identity; pull it
        # off the outbox after send_card has run.
        for entry in src.outbox:
            if entry["kind"] == "card":
                buttons = entry["card"]["body"]["elements"][1]["actions"]
                approve_value = next(b for b in buttons if b["name"] == "approve")["value"]
                await bridge.resolve(
                    CardActionEvent(
                        card_id=approve_value["card_id"],
                        user_id="u",
                        action="approve",
                    )
                )
                return

    clicker = asyncio.create_task(click_approve())
    result = await bridge.handle_tool_call(_tc())
    await clicker
    assert result is None
    kinds = [entry["kind"] for entry in src.outbox]
    assert kinds == ["card", "update_card"]
    final_card = src.outbox[-1]["card"]
    assert "approved" in final_card["header"]["title"]["content"]


@pytest.mark.asyncio
async def test_deny_flow_blocks_with_reason() -> None:
    bridge, src = _make_bridge(
        ApprovalPolicy(require_approval=frozenset({"lark"}), timeout_seconds=2.0)
    )

    async def click_deny() -> None:
        await asyncio.sleep(0.01)
        for entry in src.outbox:
            if entry["kind"] == "card":
                buttons = entry["card"]["body"]["elements"][1]["actions"]
                deny_value = next(b for b in buttons if b["name"] == "deny")["value"]
                await bridge.resolve(
                    CardActionEvent(
                        card_id=deny_value["card_id"],
                        user_id="u",
                        action="deny",
                    )
                )
                return

    clicker = asyncio.create_task(click_deny())
    result = await bridge.handle_tool_call(_tc())
    await clicker
    assert result is not None
    assert result["block"] is True
    assert "denied by u" in result["reason"]


@pytest.mark.asyncio
async def test_timeout_blocks_when_no_one_clicks() -> None:
    bridge, src = _make_bridge(
        ApprovalPolicy(require_approval=frozenset({"lark"}), timeout_seconds=0.05)
    )
    result = await bridge.handle_tool_call(_tc())
    assert result is not None
    assert result["block"] is True
    assert "timed out" in result["reason"]


@pytest.mark.asyncio
async def test_other_user_click_is_ignored() -> None:
    bridge, src = _make_bridge(
        ApprovalPolicy(require_approval=frozenset({"lark"}), timeout_seconds=2.0)
    )

    async def click_then_correct() -> None:
        await asyncio.sleep(0.01)
        # First an outsider click — should be ignored.
        card_entry = next(e for e in src.outbox if e["kind"] == "card")
        approve_value = card_entry["card"]["body"]["elements"][1]["actions"][0]["value"]
        await bridge.resolve(
            CardActionEvent(
                card_id=approve_value["card_id"], user_id="someone-else", action="approve"
            )
        )
        # Then the rightful user — bridge should still resolve.
        await bridge.resolve(
            CardActionEvent(
                card_id=approve_value["card_id"], user_id="u", action="approve"
            )
        )

    clicker = asyncio.create_task(click_then_correct())
    result = await bridge.handle_tool_call(_tc())
    await clicker
    assert result is None


@pytest.mark.asyncio
async def test_no_chat_context_blocks() -> None:
    bridge, src = _make_bridge(
        ApprovalPolicy(require_approval=frozenset({"lark"})), ctx=None
    )
    result = await bridge.handle_tool_call(_tc())
    assert result is not None
    assert "no originating chat" in result["reason"]
