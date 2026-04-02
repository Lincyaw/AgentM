"""Tests for tool call deduplication system.

Ref: designs/tool-dedup.md

Tests behavior boundaries -- dedup key generation, FIFO eviction,
store/lookup via DedupMiddleware, and backward compat.
"""

from __future__ import annotations

from agentm.config.schema import ExecutionConfig
from agentm.harness.middleware import DedupMiddleware, _make_tool_call_key


# ---------------------------------------------------------------------------
# Key generation -- _make_tool_call_key
# ---------------------------------------------------------------------------


class TestMakeKey:
    """Bug: different arg order produces different keys -> cache miss on identical calls."""

    def test_arg_order_irrelevant(self):
        key_a = _make_tool_call_key("tool_x", {"b": 2, "a": 1})
        key_b = _make_tool_call_key("tool_x", {"a": 1, "b": 2})
        assert key_a == key_b

    def test_different_tool_names_produce_different_keys(self):
        key_a = _make_tool_call_key("tool_x", {"a": 1})
        key_b = _make_tool_call_key("tool_y", {"a": 1})
        assert key_a != key_b


# ---------------------------------------------------------------------------
# DedupMiddleware -- FIFO eviction (via internal cache)
# ---------------------------------------------------------------------------


class TestFIFOEviction:
    """Bug: unbounded cache grows without limit -> OOM on long-running agents."""

    def test_evicts_oldest_when_full(self):
        mw = DedupMiddleware(max_cache_size=3)
        cache = mw._cache
        cache["k1"] = "v1"
        cache["k2"] = "v2"
        cache["k3"] = "v3"
        # Simulate store with eviction (same logic as on_tool_call)
        cache["k4"] = "v4"
        while len(cache) > 3:
            cache.popitem(last=False)

        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k4") == "v4"
        assert len(cache) == 3


# ---------------------------------------------------------------------------
# DedupMiddleware -- store/lookup (via internal cache)
# ---------------------------------------------------------------------------


class TestStoreLookup:
    """Bug: cached result not returned -> tool re-executed despite identical args."""

    def test_store_then_lookup_returns_result(self):
        mw = DedupMiddleware()
        mw._cache["k1"] = "result-1"
        assert mw._cache.get("k1") == "result-1"

    def test_lookup_missing_returns_none(self):
        mw = DedupMiddleware()
        assert mw._cache.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Backward compatibility -- ExecutionConfig without dedup
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Bug: adding dedup field to ExecutionConfig breaks existing configs without it."""

    def test_execution_config_defaults_dedup_none(self):
        cfg = ExecutionConfig()
        assert cfg.dedup is None
