"""Tests for agentm.harness.tool_filter — three-layer tool filtering."""

from __future__ import annotations

import pytest

from agentm.core.tool import Tool
from agentm.harness.tool_filter import WORKER_DISALLOWED_TOOLS, resolve_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str) -> Tool:
    """Create a minimal Tool for testing."""
    return Tool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        func=lambda: "",
    )


def _names(tools: list[Tool]) -> list[str]:
    return [t.name for t in tools]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def all_tools() -> list[Tool]:
    """A representative set of available tools."""
    return [
        _make_tool("dispatch_agent"),
        _make_tool("check_tasks"),
        _make_tool("inject_instruction"),
        _make_tool("abort_task"),
        _make_tool("check_metrics"),
        _make_tool("query_logs"),
        _make_tool("think"),
    ]


# ---------------------------------------------------------------------------
# Layer 1: Global disallowed
# ---------------------------------------------------------------------------

class TestGlobalDisallowed:
    def test_removes_global_disallowed_tools(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=["*"],
            global_disallowed=WORKER_DISALLOWED_TOOLS,
        )
        result_names = _names(result)
        for name in WORKER_DISALLOWED_TOOLS:
            assert name not in result_names

    def test_global_disallowed_not_applied_when_none(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=["*"],
            global_disallowed=None,
        )
        assert len(result) == len(all_tools)


# ---------------------------------------------------------------------------
# Layer 2: Whitelist
# ---------------------------------------------------------------------------

class TestWhitelist:
    def test_wildcard_returns_all_minus_global(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=["*"],
            global_disallowed=frozenset({"dispatch_agent"}),
        )
        assert _names(result) == [
            "check_tasks",
            "inject_instruction",
            "abort_task",
            "check_metrics",
            "query_logs",
            "think",
        ]

    def test_explicit_names_returns_only_listed(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=["check_metrics", "query_logs"],
        )
        assert _names(result) == ["check_metrics", "query_logs"]

    def test_whitelist_preserves_whitelist_order(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=["think", "check_metrics"],
        )
        assert _names(result) == ["think", "check_metrics"]

    def test_unknown_whitelist_name_raises(self, all_tools: list[Tool]) -> None:
        with pytest.raises(ValueError, match="no_such_tool"):
            resolve_tools(
                all_tools,
                whitelist=["check_metrics", "no_such_tool"],
            )

    def test_wildcard_mixed_with_names_raises(self, all_tools: list[Tool]) -> None:
        with pytest.raises(ValueError, match=r'\["\*"\] must be the sole element'):
            resolve_tools(
                all_tools,
                whitelist=["*", "check_metrics"],
            )

    def test_whitelist_name_removed_by_global_raises(self, all_tools: list[Tool]) -> None:
        """A whitelisted name that was removed by global disallowed is unknown."""
        with pytest.raises(ValueError, match="dispatch_agent"):
            resolve_tools(
                all_tools,
                whitelist=["dispatch_agent", "check_metrics"],
                global_disallowed=frozenset({"dispatch_agent"}),
            )


# ---------------------------------------------------------------------------
# Layer 3: Agent disallowed
# ---------------------------------------------------------------------------

class TestAgentDisallowed:
    def test_disallowed_removes_tools(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=["*"],
            disallowed_tools=["think", "query_logs"],
        )
        result_names = _names(result)
        assert "think" not in result_names
        assert "query_logs" not in result_names

    def test_unknown_disallowed_name_silently_ignored(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=["*"],
            disallowed_tools=["nonexistent_tool"],
        )
        # No error raised, all original tools still present
        assert len(result) == len(all_tools)


# ---------------------------------------------------------------------------
# Combined layers
# ---------------------------------------------------------------------------

class TestCombinedFiltering:
    def test_whitelist_and_disallowed_together(self, all_tools: list[Tool]) -> None:
        """Disallowed wins when a tool is in both whitelist and disallowed."""
        result = resolve_tools(
            all_tools,
            whitelist=["check_metrics", "query_logs", "think"],
            disallowed_tools=["think"],
        )
        assert _names(result) == ["check_metrics", "query_logs"]

    def test_all_three_layers(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=["*"],
            disallowed_tools=["think"],
            global_disallowed=WORKER_DISALLOWED_TOOLS,
        )
        assert _names(result) == ["check_metrics", "query_logs"]

    def test_empty_result_returns_empty_list(self) -> None:
        tools = [_make_tool("only_tool")]
        result = resolve_tools(
            tools,
            whitelist=["*"],
            disallowed_tools=["only_tool"],
        )
        assert result == []

    def test_empty_available_tools_with_wildcard(self) -> None:
        result = resolve_tools(
            [],
            whitelist=["*"],
        )
        assert result == []

    def test_empty_whitelist_returns_empty(self, all_tools: list[Tool]) -> None:
        result = resolve_tools(
            all_tools,
            whitelist=[],
        )
        assert result == []
