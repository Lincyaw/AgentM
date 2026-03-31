"""Tests for ToolResultBudgetMiddleware.

Ref: designs/loop-resilience.md, section 1.

Tests behavior boundaries — truncation, persistence, aggregate tracking,
and fallback on misconfiguration.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentm.harness.tool_result_budget import (
    OverflowStrategy,
    ToolResultBudgetConfig,
    ToolResultBudgetMiddleware,
)
from agentm.harness.types import LoopContext


def _make_ctx(**overrides: object) -> LoopContext:
    """Build a minimal LoopContext for tests."""
    defaults: dict[str, object] = {
        "agent_id": "test-agent",
        "step": 0,
        "max_steps": 10,
        "tool_call_count": 0,
        "metadata": {},
    }
    defaults.update(overrides)
    return LoopContext(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pass-through: result under limit
# ---------------------------------------------------------------------------


class TestUnderLimit:
    """Bug: middleware modifies results that are within budget."""

    @pytest.mark.asyncio
    async def test_small_result_passes_through_unchanged(self) -> None:
        config = ToolResultBudgetConfig(max_result_chars=100)
        mw = ToolResultBudgetMiddleware(config)
        call_next = AsyncMock(return_value="short result")

        result = await mw.on_tool_call("my_tool", {}, call_next, _make_ctx())

        assert result == "short result"
        call_next.assert_awaited_once_with("my_tool", {})


# ---------------------------------------------------------------------------
# TRUNCATE mode
# ---------------------------------------------------------------------------


class TestTruncateMode:
    """Bug: oversized results consume full context window, causing OOM / degraded quality."""

    @pytest.mark.asyncio
    async def test_over_limit_result_is_truncated(self) -> None:
        config = ToolResultBudgetConfig(
            max_result_chars=50,
            preview_chars=10,
            overflow_strategy=OverflowStrategy.TRUNCATE,
        )
        mw = ToolResultBudgetMiddleware(config)
        big_result = "x" * 200
        call_next = AsyncMock(return_value=big_result)

        result = await mw.on_tool_call("sql", {}, call_next, _make_ctx())

        assert "<truncated_result>" in result
        assert "</truncated_result>" in result
        assert "200" in result  # original size mentioned
        assert "x" * 10 in result  # preview present
        # Full content must NOT be present
        assert "x" * 200 not in result

    @pytest.mark.asyncio
    async def test_truncation_mentions_remaining_chars(self) -> None:
        config = ToolResultBudgetConfig(
            max_result_chars=50,
            preview_chars=10,
        )
        mw = ToolResultBudgetMiddleware(config)
        call_next = AsyncMock(return_value="a" * 100)

        result = await mw.on_tool_call("sql", {}, call_next, _make_ctx())

        # Remaining = 100 - 10 = 90
        assert "90" in result
        assert "Refine your query" in result


# ---------------------------------------------------------------------------
# PERSIST mode
# ---------------------------------------------------------------------------


class TestPersistMode:
    """Bug: no way to access full output when agent needs it for analysis."""

    @pytest.mark.asyncio
    async def test_over_limit_persists_to_disk(self, tmp_path: Path) -> None:
        config = ToolResultBudgetConfig(
            max_result_chars=50,
            preview_chars=10,
            overflow_strategy=OverflowStrategy.PERSIST,
            persist_dir=str(tmp_path),
        )
        mw = ToolResultBudgetMiddleware(config)
        big_result = "data-" * 100  # 500 chars
        call_next = AsyncMock(return_value=big_result)

        result = await mw.on_tool_call("sql", {}, call_next, _make_ctx())

        # Preview returned
        assert "<persisted_result>" in result
        assert "</persisted_result>" in result
        assert "data-data-" in result  # first 10 chars of preview

        # File written on disk
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].read_text(encoding="utf-8") == big_result

        # Filepath mentioned in result
        assert str(files[0]) in result

    @pytest.mark.asyncio
    async def test_persist_uses_tool_call_id_from_metadata(
        self, tmp_path: Path
    ) -> None:
        config = ToolResultBudgetConfig(
            max_result_chars=50,
            preview_chars=10,
            overflow_strategy=OverflowStrategy.PERSIST,
            persist_dir=str(tmp_path),
        )
        mw = ToolResultBudgetMiddleware(config)
        call_next = AsyncMock(return_value="z" * 200)
        ctx = _make_ctx(metadata={"tool_call_id": "call_abc123"})

        await mw.on_tool_call("sql", {}, call_next, ctx)

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "call_abc123.txt"

    @pytest.mark.asyncio
    async def test_persist_missing_dir_falls_back_to_truncate(self) -> None:
        """When persist_dir is empty, fallback to TRUNCATE instead of crashing."""
        config = ToolResultBudgetConfig(
            max_result_chars=50,
            preview_chars=10,
            overflow_strategy=OverflowStrategy.PERSIST,
            persist_dir="",  # empty — should not crash
        )
        mw = ToolResultBudgetMiddleware(config)
        call_next = AsyncMock(return_value="y" * 200)

        result = await mw.on_tool_call("sql", {}, call_next, _make_ctx())

        # Should truncate, not persist
        assert "<truncated_result>" in result
        assert "<persisted_result>" not in result


# ---------------------------------------------------------------------------
# Aggregate budget
# ---------------------------------------------------------------------------


class TestAggregateBudget:
    """Bug: multiple large-but-under-limit results in one turn exceed context window."""

    @pytest.mark.asyncio
    async def test_aggregate_exceeds_limit_force_truncates(self) -> None:
        config = ToolResultBudgetConfig(
            max_result_chars=100,
            max_aggregate_chars=150,
            preview_chars=10,
        )
        mw = ToolResultBudgetMiddleware(config)
        ctx = _make_ctx()

        # First call: 80 chars — under both limits
        call_next_1 = AsyncMock(return_value="a" * 80)
        r1 = await mw.on_tool_call("t1", {}, call_next_1, ctx)
        assert r1 == "a" * 80

        # Second call: 80 chars — under per-result but pushes aggregate to 160
        call_next_2 = AsyncMock(return_value="b" * 80)
        r2 = await mw.on_tool_call("t2", {}, call_next_2, ctx)
        assert "<truncated_result>" in r2

    @pytest.mark.asyncio
    async def test_aggregate_with_oversized_result_force_truncates(
        self, tmp_path: Path
    ) -> None:
        """When a result exceeds per-tool limit AND aggregate is blown, force-truncate."""
        config = ToolResultBudgetConfig(
            max_result_chars=50,
            max_aggregate_chars=8,
            preview_chars=5,
            overflow_strategy=OverflowStrategy.PERSIST,
            persist_dir=str(tmp_path),
        )
        mw = ToolResultBudgetMiddleware(config)
        ctx = _make_ctx()

        # First call: 60 chars — over per-tool, aggregate goes to 5 (preview_chars)
        # Uses PERSIST strategy (aggregate 5 < 8, not yet blown)
        call_next_1 = AsyncMock(return_value="m" * 60)
        r1 = await mw.on_tool_call("t1", {}, call_next_1, ctx)
        assert "<persisted_result>" in r1

        # Second call: 60 chars — aggregate would be 5+5=10 > 8
        # Aggregate blown → force-truncate regardless of PERSIST strategy
        call_next_2 = AsyncMock(return_value="n" * 60)
        r2 = await mw.on_tool_call("t2", {}, call_next_2, ctx)
        assert "<truncated_result>" in r2
        assert "<persisted_result>" not in r2


# ---------------------------------------------------------------------------
# Aggregate resets on new turn
# ---------------------------------------------------------------------------


class TestAggregateReset:
    """Bug: aggregate budget from previous turn leaks into the next, starving later tools."""

    @pytest.mark.asyncio
    async def test_on_llm_start_resets_aggregate(self) -> None:
        config = ToolResultBudgetConfig(
            max_result_chars=100,
            max_aggregate_chars=80,
            preview_chars=10,
        )
        mw = ToolResultBudgetMiddleware(config)
        ctx = _make_ctx()

        # First turn: push aggregate to 70
        call_next = AsyncMock(return_value="x" * 70)
        await mw.on_tool_call("t1", {}, call_next, ctx)

        # Second call in same turn: 20 chars → aggregate 90 > 80 → truncated
        call_next_2 = AsyncMock(return_value="y" * 20)
        r = await mw.on_tool_call("t2", {}, call_next_2, ctx)
        assert "<truncated_result>" in r

        # New turn: on_llm_start resets aggregate
        await mw.on_llm_start([], ctx)

        # Now same 20 chars should pass through (aggregate is 20 < 80)
        call_next_3 = AsyncMock(return_value="z" * 20)
        r2 = await mw.on_tool_call("t3", {}, call_next_3, ctx)
        assert r2 == "z" * 20
