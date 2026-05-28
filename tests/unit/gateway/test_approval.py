"""Fail-stop: ApprovalManager identity check + timeout (§3.6).

The approval gate is a human-in-the-loop safety boundary. If a click
from the wrong user resolves the gate, or a stale/expired approval can
be reused, the safety guarantee is void.
"""

from __future__ import annotations

import asyncio

import pytest

from agentm.gateway.approval import ApprovalManager


def _sink_collect(out: list[dict]) -> object:
    async def sink(body: dict) -> None:
        out.append(body)

    return sink


@pytest.mark.asyncio
async def test_matching_click_approves() -> None:
    out: list[dict] = []
    mgr = ApprovalManager(_sink_collect(out), require_approval=frozenset({"bash"}))
    task = asyncio.create_task(
        mgr.request(
            session_key="terminal:t1",
            sender_id="u1",
            channel="terminal",
            chat_id="t1",
            thread_id=None,
            tool_name="bash",
            tool_args={"cmd": "ls"},
        )
    )
    await asyncio.sleep(0)  # let the card render
    appr_id = out[0]["metadata"]["approval_id"]
    assert mgr.resolve(f"{appr_id}:approve", "u1") is True
    assert await task is True


@pytest.mark.asyncio
async def test_identity_mismatch_is_silently_dropped() -> None:
    out: list[dict] = []
    mgr = ApprovalManager(_sink_collect(out), require_approval=frozenset({"bash"}))
    task = asyncio.create_task(
        mgr.request(
            session_key="terminal:t1",
            sender_id="u1",
            channel="terminal",
            chat_id="t1",
            thread_id=None,
            tool_name="bash",
            tool_args={},
        )
    )
    await asyncio.sleep(0)
    appr_id = out[0]["metadata"]["approval_id"]
    # Wrong user clicks — dropped, future still pending.
    assert mgr.resolve(f"{appr_id}:approve", "intruder") is False
    assert not task.done()
    # Rightful user can still act.
    assert mgr.resolve(f"{appr_id}:deny", "u1") is True
    assert await task is False


@pytest.mark.asyncio
async def test_timeout_returns_false() -> None:
    out: list[dict] = []
    mgr = ApprovalManager(
        _sink_collect(out),
        require_approval=frozenset({"bash"}),
        timeout_seconds=0.05,
    )
    result = await mgr.request(
        session_key="terminal:t1",
        sender_id="u1",
        channel="terminal",
        chat_id="t1",
        thread_id=None,
        tool_name="bash",
        tool_args={},
    )
    assert result is False
    # A late click for the timed-out approval resolves nothing.
    appr_id = out[0]["metadata"]["approval_id"]
    assert mgr.resolve(f"{appr_id}:approve", "u1") is False


@pytest.mark.asyncio
async def test_always_block_denies_without_card() -> None:
    out: list[dict] = []
    mgr = ApprovalManager(_sink_collect(out), always_block=frozenset({"rm"}))
    assert mgr.requires("rm") is True
    result = await mgr.request(
        session_key="terminal:t1",
        sender_id="u1",
        channel="terminal",
        chat_id="t1",
        thread_id=None,
        tool_name="rm",
        tool_args={},
    )
    assert result is False
    assert out == []  # no card rendered for a hard-blocked tool


def test_requires_policy() -> None:
    mgr = ApprovalManager(_sink_collect([]), require_approval=frozenset({"*"}))
    assert mgr.requires("anything") is True
    mgr2 = ApprovalManager(_sink_collect([]), require_approval=frozenset({"bash"}))
    assert mgr2.requires("bash") is True
    assert mgr2.requires("read") is False
