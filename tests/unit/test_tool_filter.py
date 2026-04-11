"""Focused regression tests for three-layer tool filtering."""

from __future__ import annotations

import pytest

from agentm.core.tool import Tool
from agentm.harness.tool_filter import WORKER_DISALLOWED_TOOLS, resolve_tools


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        func=lambda: "",
    )


def _names(tools: list[Tool]) -> list[str]:
    return [t.name for t in tools]


@pytest.fixture()
def all_tools() -> list[Tool]:
    return [
        _tool("dispatch_agent"),
        _tool("check_tasks"),
        _tool("inject_instruction"),
        _tool("abort_task"),
        _tool("check_metrics"),
        _tool("query_logs"),
        _tool("think"),
    ]


def test_global_disallowed_removes_disallowed_tools(all_tools: list[Tool]) -> None:
    result = resolve_tools(all_tools, whitelist=["*"], global_disallowed=WORKER_DISALLOWED_TOOLS)
    result_names = _names(result)
    for name in WORKER_DISALLOWED_TOOLS:
        assert name not in result_names


def test_explicit_whitelist_keeps_only_ordered_subset(all_tools: list[Tool]) -> None:
    result = resolve_tools(all_tools, whitelist=["think", "check_metrics"])
    assert _names(result) == ["think", "check_metrics"]


def test_unknown_whitelist_name_raises(all_tools: list[Tool]) -> None:
    with pytest.raises(ValueError, match="no_such_tool"):
        resolve_tools(all_tools, whitelist=["check_metrics", "no_such_tool"])


def test_wildcard_mixed_with_names_raises(all_tools: list[Tool]) -> None:
    with pytest.raises(ValueError, match=r'\["\*"\] must be the sole element'):
        resolve_tools(all_tools, whitelist=["*", "check_metrics"])


def test_agent_disallowed_wins_over_whitelist(all_tools: list[Tool]) -> None:
    result = resolve_tools(
        all_tools,
        whitelist=["check_metrics", "query_logs", "think"],
        disallowed_tools=["think"],
    )
    assert _names(result) == ["check_metrics", "query_logs"]


def test_empty_whitelist_or_fully_filtered_result_returns_empty() -> None:
    assert resolve_tools([], whitelist=["*"]) == []
    result = resolve_tools([_tool("only")], whitelist=["*"], disallowed_tools=["only"])
    assert result == []
