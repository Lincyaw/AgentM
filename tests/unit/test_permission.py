"""Focused regression tests for permission policy and middleware contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentm.harness.permission import PermissionMiddleware, PermissionMode, PermissionPolicy
from agentm.harness.types import LoopContext


@dataclass
class MockTool:
    name: str
    readonly: bool = False


def _ctx() -> LoopContext:
    return LoopContext(agent_id="a", step=0, max_steps=10, tool_call_count=0, metadata={})


def _mw(
    mode: PermissionMode,
    *,
    tools: dict[str, MockTool] | None = None,
    tool_overrides: dict[str, bool] | None = None,
    audit_fn: Any = None,
) -> PermissionMiddleware:
    return PermissionMiddleware(
        policy=PermissionPolicy(mode=mode, tool_overrides=tool_overrides or {}, audit_fn=audit_fn),
        tools_dict=tools or {},  # type: ignore[arg-type]
    )


def test_can_execute_enforces_readonly_mode_with_override_precedence() -> None:
    readonly = PermissionPolicy(mode=PermissionMode.READONLY)
    assert readonly.can_execute("delete", MockTool("delete", readonly=False)) is False
    assert readonly.can_execute("search", MockTool("search", readonly=True)) is True

    overridden = PermissionPolicy(mode=PermissionMode.READONLY, tool_overrides={"delete": True})
    assert overridden.can_execute("delete", MockTool("delete", readonly=False)) is True


@pytest.mark.asyncio
async def test_on_tool_call_readonly_denies_non_readonly_tool() -> None:
    call_next = AsyncMock(return_value="should-not-run")
    mw = _mw(PermissionMode.READONLY, tools={"delete": MockTool("delete", readonly=False)})
    result = await mw.on_tool_call("delete", {}, call_next, _ctx())
    assert "Permission denied" in result
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_tool_call_readonly_allows_readonly_tool() -> None:
    call_next = AsyncMock(return_value="ok")
    mw = _mw(PermissionMode.READONLY, tools={"search": MockTool("search", readonly=True)})
    result = await mw.on_tool_call("search", {"q": "x"}, call_next, _ctx())
    assert result == "ok"
    call_next.assert_awaited_once_with("search", {"q": "x"})


@pytest.mark.asyncio
async def test_on_tool_call_supervised_executes_and_audits() -> None:
    call_next = AsyncMock(return_value="exec-result")
    audit_fn = AsyncMock()
    mw = _mw(PermissionMode.SUPERVISED, audit_fn=audit_fn)
    result = await mw.on_tool_call("write", {"path": "/tmp/x"}, call_next, _ctx())
    assert result == "exec-result"
    audit_fn.assert_awaited_once_with("write", {"path": "/tmp/x"}, "exec-result")


@pytest.mark.asyncio
async def test_on_llm_start_readonly_injects_constraint() -> None:
    mw = _mw(PermissionMode.READONLY)
    messages: list[Any] = [{"role": "system", "content": "You are helpful."}, {"role": "human", "content": "hi"}]
    result = await mw.on_llm_start(messages, _ctx())
    assert "<permission_constraint>" in result[0]["content"]
    assert "READONLY" in result[0]["content"]


@pytest.mark.asyncio
async def test_on_llm_start_default_mode_keeps_messages_unchanged() -> None:
    mw = _mw(PermissionMode.DEFAULT)
    messages: list[Any] = [{"role": "system", "content": "You are helpful."}]
    result = await mw.on_llm_start(messages, _ctx())
    assert result is messages
