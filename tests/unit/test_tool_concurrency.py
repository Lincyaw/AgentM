"""Tests for tool call concurrency partitioning."""
from __future__ import annotations

import os
from unittest.mock import patch

from agentm.core.tool import Tool
from agentm.harness.tool_concurrency import (
    DEFAULT_MAX_TOOL_CONCURRENCY,
    get_max_tool_concurrency,
    partition_tool_calls,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str, concurrency_safe: bool = False) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        func=lambda: "",
        concurrency_safe=concurrency_safe,
    )


def _tc(name: str) -> dict[str, object]:
    """Shorthand for a tool call dict."""
    return {"name": name, "args": {}}


# ---------------------------------------------------------------------------
# partition_tool_calls tests
# ---------------------------------------------------------------------------

class TestPartitionToolCalls:
    """Partition tool calls into concurrent/serial batches."""

    def test_all_safe_tools_single_batch(self) -> None:
        """All concurrency-safe tools are grouped into one concurrent batch."""
        tools = {
            "a": _make_tool("a", concurrency_safe=True),
            "b": _make_tool("b", concurrency_safe=True),
            "c": _make_tool("c", concurrency_safe=True),
        }
        result = partition_tool_calls([_tc("a"), _tc("b"), _tc("c")], tools)
        assert len(result) == 1
        is_concurrent, batch = result[0]
        assert is_concurrent is True
        assert len(batch) == 3

    def test_all_unsafe_tools_separate_batches(self) -> None:
        """Each unsafe tool gets its own serial batch."""
        tools = {
            "x": _make_tool("x", concurrency_safe=False),
            "y": _make_tool("y", concurrency_safe=False),
        }
        result = partition_tool_calls([_tc("x"), _tc("y")], tools)
        assert len(result) == 2
        assert result[0] == (False, [_tc("x")])
        assert result[1] == (False, [_tc("y")])

    def test_mixed_safe_unsafe_safe(self) -> None:
        """safe, safe, unsafe, safe, safe -> three batches."""
        s1 = _make_tool("s1", concurrency_safe=True)
        s2 = _make_tool("s2", concurrency_safe=True)
        s3 = _make_tool("s3", concurrency_safe=True)
        s4 = _make_tool("s4", concurrency_safe=True)
        u = _make_tool("u", concurrency_safe=False)
        tools = {"s1": s1, "s2": s2, "u": u, "s3": s3, "s4": s4}
        calls = [_tc("s1"), _tc("s2"), _tc("u"), _tc("s3"), _tc("s4")]
        result = partition_tool_calls(calls, tools)
        assert len(result) == 3
        assert result[0] == (True, [_tc("s1"), _tc("s2")])
        assert result[1] == (False, [_tc("u")])
        assert result[2] == (True, [_tc("s3"), _tc("s4")])

    def test_single_safe_tool(self) -> None:
        """Single safe tool -> one concurrent batch."""
        tools = {"a": _make_tool("a", concurrency_safe=True)}
        result = partition_tool_calls([_tc("a")], tools)
        assert result == [(True, [_tc("a")])]

    def test_single_unsafe_tool(self) -> None:
        """Single unsafe tool -> one serial batch."""
        tools = {"a": _make_tool("a", concurrency_safe=False)}
        result = partition_tool_calls([_tc("a")], tools)
        assert result == [(False, [_tc("a")])]

    def test_empty_list(self) -> None:
        """Empty tool call list -> empty result."""
        result = partition_tool_calls([], {})
        assert result == []

    def test_unknown_tool_treated_as_unsafe(self) -> None:
        """Tool not in tools_dict is treated as non-concurrent-safe."""
        tools = {"known": _make_tool("known", concurrency_safe=True)}
        calls = [_tc("known"), _tc("unknown"), _tc("known")]
        result = partition_tool_calls(calls, tools)
        # known -> concurrent batch, unknown -> serial, known -> concurrent
        assert len(result) == 3
        assert result[0] == (True, [_tc("known")])
        assert result[1] == (False, [_tc("unknown")])
        assert result[2] == (True, [_tc("known")])

    def test_unsafe_isolates_safe_groups(self) -> None:
        """A single unsafe between two safe groups produces three batches."""
        sa = _make_tool("sa", concurrency_safe=True)
        sb = _make_tool("sb", concurrency_safe=True)
        sc = _make_tool("sc", concurrency_safe=True)
        sd = _make_tool("sd", concurrency_safe=True)
        ux = _make_tool("ux", concurrency_safe=False)
        tools = {"sa": sa, "sb": sb, "ux": ux, "sc": sc, "sd": sd}
        calls = [_tc("sa"), _tc("sb"), _tc("ux"), _tc("sc"), _tc("sd")]
        result = partition_tool_calls(calls, tools)
        assert len(result) == 3
        assert result[0][0] is True
        assert len(result[0][1]) == 2
        assert result[1][0] is False
        assert len(result[1][1]) == 1
        assert result[2][0] is True
        assert len(result[2][1]) == 2


# ---------------------------------------------------------------------------
# get_max_tool_concurrency tests
# ---------------------------------------------------------------------------

class TestGetMaxToolConcurrency:
    """Max concurrency setting from environment."""

    def test_default_value(self) -> None:
        """Without env var, returns the default constant."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure the env var is not set
            os.environ.pop("AGENTM_MAX_TOOL_CONCURRENCY", None)
            assert get_max_tool_concurrency() == DEFAULT_MAX_TOOL_CONCURRENCY

    def test_env_var_override(self) -> None:
        """With valid env var, returns parsed integer."""
        with patch.dict(os.environ, {"AGENTM_MAX_TOOL_CONCURRENCY": "4"}):
            assert get_max_tool_concurrency() == 4

    def test_invalid_env_var_returns_default(self) -> None:
        """With non-integer env var, falls back to default."""
        with patch.dict(os.environ, {"AGENTM_MAX_TOOL_CONCURRENCY": "notanumber"}):
            assert get_max_tool_concurrency() == DEFAULT_MAX_TOOL_CONCURRENCY
