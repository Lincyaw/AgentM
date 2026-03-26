"""Tests for tool call deduplication system.

Ref: designs/tool-dedup.md

Tests behavior boundaries — dedup key generation, FIFO eviction,
store/lookup, and backward compat.
"""

from __future__ import annotations

from agentm.config.schema import ExecutionConfig
from agentm.harness.middleware import DedupTracker


# ---------------------------------------------------------------------------
# DedupTracker — make_key
# ---------------------------------------------------------------------------


class TestMakeKey:
    """Bug: different arg order produces different keys -> cache miss on identical calls."""

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
    """Bug: unbounded cache grows without limit -> OOM on long-running agents."""

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
    """Bug: cached result not returned -> tool re-executed despite identical args."""

    def test_store_then_lookup_returns_result(self):
        tracker = DedupTracker()
        tracker.store("k1", "result-1")
        assert tracker.lookup("k1") == "result-1"

    def test_lookup_missing_returns_none(self):
        tracker = DedupTracker()
        assert tracker.lookup("nonexistent") is None


# ---------------------------------------------------------------------------
# Backward compatibility — ExecutionConfig without dedup
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Bug: adding dedup field to ExecutionConfig breaks existing configs without it."""

    def test_execution_config_defaults_dedup_none(self):
        cfg = ExecutionConfig()
        assert cfg.dedup is None
