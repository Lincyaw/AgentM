"""Focused tests for ToolResultBudgetMiddleware overflow and aggregate behavior."""
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


def _ctx(**overrides: object) -> LoopContext:
    defaults: dict[str, object] = {
        "agent_id": "test-agent",
        "step": 0,
        "max_steps": 10,
        "tool_call_count": 0,
        "metadata": {},
    }
    defaults.update(overrides)
    return LoopContext(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_small_result_passes_through_and_aggregate_resets_on_new_turn() -> None:
    mw = ToolResultBudgetMiddleware(
        ToolResultBudgetConfig(max_result_chars=100, max_aggregate_chars=80, preview_chars=10)
    )
    ctx = _ctx()

    assert await mw.on_tool_call("t1", {}, AsyncMock(return_value="ok"), ctx) == "ok"
    await mw.on_tool_call("t2", {}, AsyncMock(return_value="x" * 70), ctx)
    truncated = await mw.on_tool_call("t3", {}, AsyncMock(return_value="y" * 20), ctx)
    assert "<truncated_result>" in truncated

    await mw.on_llm_start([], ctx)
    assert await mw.on_tool_call("t4", {}, AsyncMock(return_value="z" * 20), ctx) == "z" * 20


@pytest.mark.asyncio
async def test_truncate_mode_for_oversized_result_includes_preview_and_metadata() -> None:
    mw = ToolResultBudgetMiddleware(
        ToolResultBudgetConfig(
            max_result_chars=50,
            preview_chars=10,
            overflow_strategy=OverflowStrategy.TRUNCATE,
        )
    )

    result = await mw.on_tool_call("sql", {}, AsyncMock(return_value="x" * 200), _ctx())

    assert "<truncated_result>" in result
    assert "200" in result
    assert "x" * 10 in result
    assert "x" * 200 not in result


@pytest.mark.asyncio
async def test_persist_mode_writes_full_output_and_returns_reference(tmp_path: Path) -> None:
    mw = ToolResultBudgetMiddleware(
        ToolResultBudgetConfig(
            max_result_chars=50,
            preview_chars=10,
            overflow_strategy=OverflowStrategy.PERSIST,
            persist_dir=str(tmp_path),
        )
    )
    big = "data-" * 100

    result = await mw.on_tool_call(
        "sql",
        {},
        AsyncMock(return_value=big),
        _ctx(metadata={"tool_call_id": "call_abc123"}),
    )

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "call_abc123.txt"
    assert files[0].read_text(encoding="utf-8") == big
    assert "<persisted_result>" in result


@pytest.mark.asyncio
async def test_aggregate_budget_force_truncates_when_total_exceeds_limit() -> None:
    mw = ToolResultBudgetMiddleware(
        ToolResultBudgetConfig(max_result_chars=100, max_aggregate_chars=150, preview_chars=10)
    )
    ctx = _ctx()

    assert await mw.on_tool_call("t1", {}, AsyncMock(return_value="a" * 80), ctx) == "a" * 80
    second = await mw.on_tool_call("t2", {}, AsyncMock(return_value="b" * 80), ctx)
    assert "<truncated_result>" in second
