"""Focused tests for tool-call concurrency partitioning and env override."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agentm.core.tool import Tool
from agentm.harness.tool_concurrency import (
    DEFAULT_MAX_TOOL_CONCURRENCY,
    get_max_tool_concurrency,
    partition_tool_calls,
)


def _tool(name: str, concurrency_safe: bool = False) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        func=lambda: "",
        concurrency_safe=concurrency_safe,
    )


def _tc(name: str) -> dict[str, object]:
    return {"name": name, "args": {}}


def test_partition_all_safe_tools_into_single_concurrent_batch() -> None:
    tools = {"a": _tool("a", True), "b": _tool("b", True)}
    assert partition_tool_calls([_tc("a"), _tc("b")], tools) == [(True, [_tc("a"), _tc("b")])]


def test_partition_mixed_safe_and_unsafe_tools_including_unknown() -> None:
    tools = {"s1": _tool("s1", True), "u": _tool("u", False), "s2": _tool("s2", True)}
    result = partition_tool_calls([_tc("s1"), _tc("u"), _tc("unknown"), _tc("s2")], tools)

    assert result[0] == (True, [_tc("s1")])
    assert result[1] == (False, [_tc("u")])
    assert result[2] == (False, [_tc("unknown")])
    assert result[3] == (True, [_tc("s2")])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4", 4),
        ("notanumber", DEFAULT_MAX_TOOL_CONCURRENCY),
    ],
)
def test_get_max_tool_concurrency_respects_or_falls_back_from_env(raw: str, expected: int) -> None:
    with patch.dict(os.environ, {"AGENTM_MAX_TOOL_CONCURRENCY": raw}):
        assert get_max_tool_concurrency() == expected
