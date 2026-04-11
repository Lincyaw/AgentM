"""Focused tests for tool-call dedup keying and cache behavior."""

from __future__ import annotations

from agentm.config.schema import ExecutionConfig
from agentm.harness.middleware import DedupMiddleware, _make_tool_call_key


def test_make_tool_call_key_is_stable_across_arg_order() -> None:
    assert _make_tool_call_key("tool_x", {"b": 2, "a": 1}) == _make_tool_call_key("tool_x", {"a": 1, "b": 2})


def test_make_tool_call_key_differs_for_different_tools() -> None:
    assert _make_tool_call_key("tool_x", {"a": 1}) != _make_tool_call_key("tool_y", {"a": 1})


def test_fifo_eviction_evicts_oldest_entry_when_cache_is_full() -> None:
    mw = DedupMiddleware(max_cache_size=3)
    mw._cache["k1"] = "v1"
    mw._cache["k2"] = "v2"
    mw._cache["k3"] = "v3"
    mw._cache["k4"] = "v4"
    while len(mw._cache) > 3:
        mw._cache.popitem(last=False)
    assert "k1" not in mw._cache
    assert set(mw._cache) == {"k2", "k3", "k4"}


def test_execution_config_defaults_dedup_to_none_for_backward_compat() -> None:
    assert ExecutionConfig().dedup is None
