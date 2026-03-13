"""Tests for tool call deduplication system.

Ref: designs/tool-dedup.md

Tests behavior boundaries — dedup key generation, FIFO eviction,
hook injection/passthrough, wrapper interception, and backward compat.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from agentm.middleware.dedup import (
    DedupTracker,
    build_dedup_hook,
    wrap_tool_with_dedup,
)
from agentm.config.schema import ExecutionConfig


# ---------------------------------------------------------------------------
# DedupTracker — make_key
# ---------------------------------------------------------------------------


class TestMakeKey:
    """Bug: different arg order produces different keys → cache miss on identical calls."""

    def test_arg_order_irrelevant(self):
        tracker = DedupTracker()
        key_a = tracker.make_key("tool_x", {"b": 2, "a": 1})
        key_b = tracker.make_key("tool_x", {"a": 1, "b": 2})
        assert key_a == key_b

    def test_different_tool_names_produce_different_keys(self):
        tracker = DedupTracker()
        key_a = tracker.make_key("tool_x", {"a": 1})
        key_b = tracker.make_key("tool_y", {"a": 1})
        assert key_a != key_b


# ---------------------------------------------------------------------------
# DedupTracker — FIFO eviction
# ---------------------------------------------------------------------------


class TestFIFOEviction:
    """Bug: unbounded cache grows without limit → OOM on long-running agents."""

    def test_evicts_oldest_when_full(self):
        tracker = DedupTracker(max_cache_size=3)
        tracker.store("k1", "v1")
        tracker.store("k2", "v2")
        tracker.store("k3", "v3")
        tracker.store("k4", "v4")  # should evict k1

        assert tracker.lookup("k1") is None
        assert tracker.lookup("k2") == "v2"
        assert tracker.lookup("k4") == "v4"
        assert tracker.size == 3


# ---------------------------------------------------------------------------
# DedupTracker — store/lookup
# ---------------------------------------------------------------------------


class TestStoreLookup:
    """Bug: cached result not returned → tool re-executed despite identical args."""

    def test_store_then_lookup_returns_result(self):
        tracker = DedupTracker()
        tracker.store("k1", "result-1")
        assert tracker.lookup("k1") == "result-1"

    def test_lookup_missing_returns_none(self):
        tracker = DedupTracker()
        assert tracker.lookup("nonexistent") is None


# ---------------------------------------------------------------------------
# build_dedup_hook — injection
# ---------------------------------------------------------------------------


def _make_tool_call_pair(
    tool_name: str, args: dict, result: str, call_id: str = "tc-1"
) -> tuple[AIMessage, ToolMessage]:
    """Helper: create an AI tool call + matching ToolMessage pair."""
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": tool_name, "args": args}],
    )
    tool_msg = ToolMessage(content=result, tool_call_id=call_id)
    return ai_msg, tool_msg


class TestDedupHookInjection:
    """Bug: LLM re-calls a tool with identical args because it forgot the result."""

    def test_injects_system_message_when_cached(self):
        tracker = DedupTracker()
        hook = build_dedup_hook(tracker)

        ai1, tool1 = _make_tool_call_pair(
            "query_metrics", {"svc": "A"}, "cpu=85%", "tc-1"
        )
        # Second identical call in a new AI message
        ai2 = AIMessage(
            content="",
            tool_calls=[{"id": "tc-2", "name": "query_metrics", "args": {"svc": "A"}}],
        )

        state = {"messages": [ai1, tool1, ai2]}
        result = hook(state)

        messages = result["messages"]
        injected_msgs = [
            m
            for m in messages
            if isinstance(m, HumanMessage) and "DEDUP" in getattr(m, "content", "")
        ]
        assert len(injected_msgs) >= 1
        assert "DEDUP WARNING" in injected_msgs[-1].content
        assert "query_metrics" in injected_msgs[-1].content
        # Must NOT contain the actual result — token savings
        assert "cpu=85%" not in injected_msgs[-1].content

    def test_passthrough_when_no_cache(self):
        tracker = DedupTracker()
        hook = build_dedup_hook(tracker)

        ai1 = AIMessage(
            content="",
            tool_calls=[{"id": "tc-1", "name": "query_metrics", "args": {"svc": "A"}}],
        )
        state = {"messages": [ai1]}
        result = hook(state)

        messages = result["messages"]
        injected_msgs = [
            m
            for m in messages
            if isinstance(m, HumanMessage) and "DEDUP" in getattr(m, "content", "")
        ]
        assert len(injected_msgs) == 0


class TestDedupHookHistoryScanning:
    """Bug: tracker not populated from history → first repeat not caught by wrapper."""

    def test_populates_tracker_from_history(self):
        tracker = DedupTracker()
        hook = build_dedup_hook(tracker)

        ai1, tool1 = _make_tool_call_pair(
            "get_graph", {"svc": "B"}, "graph-data", "tc-1"
        )
        state = {"messages": [ai1, tool1]}
        hook(state)

        key = tracker.make_key("get_graph", {"svc": "B"})
        assert tracker.lookup(key) == "graph-data"


class TestDedupHookExclusions:
    """Bug: think tool cached → reasoning disrupted because LLM gets stale thoughts."""

    def test_excludes_think_tool(self):
        tracker = DedupTracker()
        hook = build_dedup_hook(tracker)

        ai1, tool1 = _make_tool_call_pair("think", {"thought": "hmm"}, "hmm", "tc-1")
        state = {"messages": [ai1, tool1]}
        hook(state)

        key = tracker.make_key("think", {"thought": "hmm"})
        assert tracker.lookup(key) is None


# ---------------------------------------------------------------------------
# wrap_tool_with_dedup — wrapper
# ---------------------------------------------------------------------------


def _make_simple_tool(name: str = "my_tool") -> StructuredTool:
    """Create a simple StructuredTool for testing."""

    class Args(BaseModel):
        query: str

    def func(query: str) -> str:
        return f"result-for-{query}"

    return StructuredTool(
        name=name,
        description=f"A test tool called {name}",
        func=func,
        args_schema=Args,
    )


class TestWrapToolWithDedup:
    """Bug: identical tool call executes twice → wasted API call and tokens."""

    def test_blocks_repeat_call_with_short_message(self):
        tracker = DedupTracker()
        tool = _make_simple_tool()
        wrapped = wrap_tool_with_dedup(tool, tracker)

        first = wrapped.invoke({"query": "abc"})
        second = wrapped.invoke({"query": "abc"})

        assert first == "result-for-abc"
        assert "BLOCKED" in second
        # Must NOT contain the original result — that's the whole point
        assert "result-for-abc" not in second

    def test_allows_first_call(self):
        tracker = DedupTracker()
        tool = _make_simple_tool()
        wrapped = wrap_tool_with_dedup(tool, tracker)

        result = wrapped.invoke({"query": "xyz"})
        assert result == "result-for-xyz"

    def test_preserves_tool_name_and_description(self):
        tracker = DedupTracker()
        tool = _make_simple_tool("special_tool")
        wrapped = wrap_tool_with_dedup(tool, tracker)

        assert wrapped.name == "special_tool"
        assert "special_tool" in wrapped.description


# ---------------------------------------------------------------------------
# Backward compatibility — ExecutionConfig without dedup
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Bug: adding dedup field to ExecutionConfig breaks existing configs without it."""

    def test_execution_config_defaults_dedup_none(self):
        cfg = ExecutionConfig()
        assert cfg.dedup is None
