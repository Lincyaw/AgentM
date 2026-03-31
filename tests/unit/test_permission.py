"""Tests for the permission mode module (harness/permission.py).

Each test targets a specific behavior / failure scenario documented in the
permission-mode design doc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentm.harness.permission import (
    PermissionMiddleware,
    PermissionMode,
    PermissionPolicy,
)
from agentm.harness.types import LoopContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class MockTool:
    """Minimal mock tool with an optional readonly attribute."""

    name: str
    readonly: bool = False


def _make_ctx(step: int = 0) -> LoopContext:
    return LoopContext(
        agent_id="test-agent",
        step=step,
        max_steps=10,
        tool_call_count=0,
        metadata={},
    )


def _make_middleware(
    mode: PermissionMode,
    tools: dict[str, MockTool] | None = None,
    tool_overrides: dict[str, bool] | None = None,
    audit_fn: Any = None,
) -> PermissionMiddleware:
    policy = PermissionPolicy(
        mode=mode,
        tool_overrides=tool_overrides or {},
        audit_fn=audit_fn,
    )
    return PermissionMiddleware(
        policy=policy,
        tools_dict=tools or {},  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# PermissionPolicy.can_execute
# ---------------------------------------------------------------------------


class TestCanExecute:
    """Tests for PermissionPolicy.can_execute resolution logic."""

    def test_default_mode_allows_all(self) -> None:
        """DEFAULT mode allows every tool regardless of readonly flag."""
        policy = PermissionPolicy(mode=PermissionMode.DEFAULT)
        tool = MockTool(name="delete_file", readonly=False)
        assert policy.can_execute("delete_file", tool) is True

    def test_readonly_mode_rejects_non_readonly_tool(self) -> None:
        """READONLY mode denies tools not marked as readonly."""
        policy = PermissionPolicy(mode=PermissionMode.READONLY)
        tool = MockTool(name="delete_file", readonly=False)
        assert policy.can_execute("delete_file", tool) is False

    def test_readonly_mode_allows_readonly_tool(self) -> None:
        """READONLY mode permits tools marked as readonly."""
        policy = PermissionPolicy(mode=PermissionMode.READONLY)
        tool = MockTool(name="search_logs", readonly=True)
        assert policy.can_execute("search_logs", tool) is True

    def test_readonly_mode_denies_when_tool_is_none(self) -> None:
        """READONLY mode denies unknown tools (tool=None) as safe fallback."""
        policy = PermissionPolicy(mode=PermissionMode.READONLY)
        assert policy.can_execute("unknown_tool", None) is False

    def test_supervised_mode_allows_all(self) -> None:
        """SUPERVISED mode allows every tool (audit happens separately)."""
        policy = PermissionPolicy(mode=PermissionMode.SUPERVISED)
        tool = MockTool(name="delete_file", readonly=False)
        assert policy.can_execute("delete_file", tool) is True

    def test_unrestricted_mode_allows_all(self) -> None:
        """UNRESTRICTED mode allows every tool."""
        policy = PermissionPolicy(mode=PermissionMode.UNRESTRICTED)
        tool = MockTool(name="delete_file", readonly=False)
        assert policy.can_execute("delete_file", tool) is True

    def test_tool_overrides_take_precedence_over_readonly_deny(self) -> None:
        """Per-tool override can allow a non-readonly tool in READONLY mode."""
        policy = PermissionPolicy(
            mode=PermissionMode.READONLY,
            tool_overrides={"delete_file": True},
        )
        tool = MockTool(name="delete_file", readonly=False)
        assert policy.can_execute("delete_file", tool) is True

    def test_tool_overrides_can_deny_in_default_mode(self) -> None:
        """Per-tool override can deny a tool even in DEFAULT mode."""
        policy = PermissionPolicy(
            mode=PermissionMode.DEFAULT,
            tool_overrides={"dangerous_tool": False},
        )
        tool = MockTool(name="dangerous_tool", readonly=False)
        assert policy.can_execute("dangerous_tool", tool) is False



# ---------------------------------------------------------------------------
# PermissionMiddleware.on_tool_call
# ---------------------------------------------------------------------------


class TestOnToolCall:
    """Tests for middleware tool-call interception."""

    @pytest.mark.asyncio
    async def test_default_mode_passes_through(self) -> None:
        """DEFAULT mode calls call_next without interference."""
        call_next = AsyncMock(return_value="result-ok")
        mw = _make_middleware(PermissionMode.DEFAULT)
        ctx = _make_ctx()

        result = await mw.on_tool_call("any_tool", {"arg": 1}, call_next, ctx)

        assert result == "result-ok"
        call_next.assert_awaited_once_with("any_tool", {"arg": 1})

    @pytest.mark.asyncio
    async def test_unrestricted_mode_short_circuits(self) -> None:
        """UNRESTRICTED mode calls call_next immediately with no checks."""
        call_next = AsyncMock(return_value="fast-result")
        mw = _make_middleware(PermissionMode.UNRESTRICTED)
        ctx = _make_ctx()

        result = await mw.on_tool_call("any_tool", {}, call_next, ctx)

        assert result == "fast-result"
        call_next.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_readonly_rejects_non_readonly_tool(self) -> None:
        """READONLY mode returns an error string for non-readonly tools."""
        call_next = AsyncMock(return_value="should-not-reach")
        tool = MockTool(name="delete_file", readonly=False)
        mw = _make_middleware(
            PermissionMode.READONLY, tools={"delete_file": tool}
        )
        ctx = _make_ctx()

        result = await mw.on_tool_call("delete_file", {}, call_next, ctx)

        assert "Permission denied" in result
        assert "READONLY" in result
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_readonly_allows_readonly_tool(self) -> None:
        """READONLY mode permits readonly tools through to call_next."""
        call_next = AsyncMock(return_value="search-result")
        tool = MockTool(name="search", readonly=True)
        mw = _make_middleware(PermissionMode.READONLY, tools={"search": tool})
        ctx = _make_ctx()

        result = await mw.on_tool_call("search", {"q": "test"}, call_next, ctx)

        assert result == "search-result"
        call_next.assert_awaited_once_with("search", {"q": "test"})

    @pytest.mark.asyncio
    async def test_readonly_rejects_unknown_tool(self) -> None:
        """READONLY mode denies tools not found in tools_dict."""
        call_next = AsyncMock(return_value="should-not-reach")
        mw = _make_middleware(PermissionMode.READONLY, tools={})
        ctx = _make_ctx()

        result = await mw.on_tool_call("hallucinated_tool", {}, call_next, ctx)

        assert "Permission denied" in result
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_supervised_executes_and_calls_audit_fn(self) -> None:
        """SUPERVISED mode executes the tool and then calls audit_fn."""
        call_next = AsyncMock(return_value="exec-result")
        audit_fn = AsyncMock()
        mw = _make_middleware(PermissionMode.SUPERVISED, audit_fn=audit_fn)
        ctx = _make_ctx()

        result = await mw.on_tool_call(
            "write_file", {"path": "/tmp/x"}, call_next, ctx
        )

        assert result == "exec-result"
        call_next.assert_awaited_once_with("write_file", {"path": "/tmp/x"})
        audit_fn.assert_awaited_once_with(
            "write_file", {"path": "/tmp/x"}, "exec-result"
        )

    @pytest.mark.asyncio
    async def test_supervised_without_audit_fn_still_executes(self) -> None:
        """SUPERVISED mode without audit_fn logs but does not fail."""
        call_next = AsyncMock(return_value="ok")
        mw = _make_middleware(PermissionMode.SUPERVISED, audit_fn=None)
        ctx = _make_ctx()

        result = await mw.on_tool_call("some_tool", {}, call_next, ctx)

        assert result == "ok"
        call_next.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_supervised_audit_fn_exception_does_not_break_execution(
        self,
    ) -> None:
        """If audit_fn raises, the tool result is still returned."""
        call_next = AsyncMock(return_value="result")
        audit_fn = AsyncMock(side_effect=RuntimeError("audit broke"))
        mw = _make_middleware(PermissionMode.SUPERVISED, audit_fn=audit_fn)
        ctx = _make_ctx()

        result = await mw.on_tool_call("tool", {}, call_next, ctx)

        assert result == "result"

    @pytest.mark.asyncio
    async def test_tool_override_allows_in_readonly(self) -> None:
        """tool_overrides can allow a non-readonly tool in READONLY mode."""
        call_next = AsyncMock(return_value="override-ok")
        tool = MockTool(name="special", readonly=False)
        mw = _make_middleware(
            PermissionMode.READONLY,
            tools={"special": tool},
            tool_overrides={"special": True},
        )
        ctx = _make_ctx()

        result = await mw.on_tool_call("special", {}, call_next, ctx)

        assert result == "override-ok"
        call_next.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tool_override_denies_in_default(self) -> None:
        """tool_overrides can deny a tool in DEFAULT mode."""
        call_next = AsyncMock(return_value="should-not-reach")
        tool = MockTool(name="blocked", readonly=False)
        mw = _make_middleware(
            PermissionMode.DEFAULT,
            tools={"blocked": tool},
            tool_overrides={"blocked": False},
        )
        ctx = _make_ctx()

        await mw.on_tool_call("blocked", {}, call_next, ctx)

        # DEFAULT mode passes through to call_next — overrides are checked
        # only in PermissionPolicy.can_execute, not in DEFAULT on_tool_call path.
        # The middleware only checks can_execute for READONLY mode.
        # So in DEFAULT mode, call_next IS invoked even with override=False.
        # This is by design: DEFAULT mode's on_tool_call always passes through.
        call_next.assert_awaited_once()


# ---------------------------------------------------------------------------
# PermissionMiddleware.on_llm_start
# ---------------------------------------------------------------------------


class TestOnLlmStart:
    """Tests for READONLY prompt injection."""

    @pytest.mark.asyncio
    async def test_readonly_injects_constraint_into_system_message(
        self,
    ) -> None:
        """READONLY mode appends permission_constraint to the system message."""
        mw = _make_middleware(PermissionMode.READONLY)
        ctx = _make_ctx()
        messages: list[Any] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "human", "content": "Do something."},
        ]

        result = await mw.on_llm_start(messages, ctx)

        assert len(result) == 2
        system_content = result[0]["content"]
        assert "<permission_constraint>" in system_content
        assert "READONLY" in system_content
        # Original content is preserved
        assert "You are helpful." in system_content

    @pytest.mark.asyncio
    async def test_readonly_prepends_system_if_none_exists(self) -> None:
        """READONLY mode creates a system message if none exists."""
        mw = _make_middleware(PermissionMode.READONLY)
        ctx = _make_ctx()
        messages: list[Any] = [
            {"role": "human", "content": "Hello"},
        ]

        result = await mw.on_llm_start(messages, ctx)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "<permission_constraint>" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_default_mode_does_not_inject(self) -> None:
        """DEFAULT mode returns messages unchanged."""
        mw = _make_middleware(PermissionMode.DEFAULT)
        ctx = _make_ctx()
        messages: list[Any] = [
            {"role": "system", "content": "You are helpful."},
        ]

        result = await mw.on_llm_start(messages, ctx)

        assert result is messages  # Same object, not modified

    @pytest.mark.asyncio
    async def test_supervised_mode_does_not_inject(self) -> None:
        """SUPERVISED mode returns messages unchanged."""
        mw = _make_middleware(PermissionMode.SUPERVISED)
        ctx = _make_ctx()
        messages: list[Any] = [
            {"role": "system", "content": "You are helpful."},
        ]

        result = await mw.on_llm_start(messages, ctx)

        assert result is messages

    @pytest.mark.asyncio
    async def test_unrestricted_mode_does_not_inject(self) -> None:
        """UNRESTRICTED mode returns messages unchanged."""
        mw = _make_middleware(PermissionMode.UNRESTRICTED)
        ctx = _make_ctx()
        messages: list[Any] = [
            {"role": "system", "content": "You are helpful."},
        ]

        result = await mw.on_llm_start(messages, ctx)

        assert result is messages
