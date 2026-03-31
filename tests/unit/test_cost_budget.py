"""Tests for CostBudgetMiddleware — usage tracking and budget enforcement."""
from __future__ import annotations

import pytest

from agentm.harness.cost_budget import (
    CostBudgetConfig,
    CostBudgetExceeded,
    CostBudgetMiddleware,
    TokenBudgetExceeded,
)
from agentm.harness.types import LoopContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockResponse:
    """LLM response with usage_metadata for testing."""

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


class BareResponse:
    """LLM response without usage_metadata."""


def _make_ctx() -> LoopContext:
    return LoopContext(
        agent_id="test-agent",
        step=0,
        max_steps=10,
        tool_call_count=0,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_without_usage_passes_through() -> None:
    """Response lacking usage_metadata should pass through with no accumulation."""
    mw = CostBudgetMiddleware(CostBudgetConfig(max_cost_usd=1.0))
    resp = BareResponse()
    result = await mw.on_llm_end(resp, _make_ctx())

    assert result is resp
    assert mw.accumulated_cost == 0.0
    assert mw.accumulated_tokens == 0


@pytest.mark.asyncio
async def test_cost_accumulated_correctly() -> None:
    """Usage metadata should be extracted and cost computed from per-million rates."""
    config = CostBudgetConfig(
        cost_per_million_input=3.0,   # $3 / 1M input tokens
        cost_per_million_output=15.0, # $15 / 1M output tokens
    )
    mw = CostBudgetMiddleware(config)

    resp = MockResponse(input_tokens=1_000_000, output_tokens=100_000)
    await mw.on_llm_end(resp, _make_ctx())

    # Expected: 1M * 3/1M + 100K * 15/1M = 3.0 + 1.5 = 4.5
    assert mw.accumulated_cost == pytest.approx(4.5)
    assert mw.accumulated_tokens == 1_100_000


@pytest.mark.asyncio
async def test_cost_under_budget_passes_through() -> None:
    """When cost is under the limit, response is returned unchanged."""
    config = CostBudgetConfig(
        max_cost_usd=10.0,
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
    )
    mw = CostBudgetMiddleware(config)

    resp = MockResponse(input_tokens=1000, output_tokens=500)
    result = await mw.on_llm_end(resp, _make_ctx())

    assert result is resp


@pytest.mark.asyncio
async def test_cost_exceeds_budget_raises() -> None:
    """When accumulated cost exceeds max_cost_usd, CostBudgetExceeded is raised."""
    config = CostBudgetConfig(
        max_cost_usd=0.001,
        cost_per_million_input=10.0,
    )
    mw = CostBudgetMiddleware(config)

    resp = MockResponse(input_tokens=1000, output_tokens=0)
    # Cost = 1000 * 10 / 1M = 0.01 > 0.001
    with pytest.raises(CostBudgetExceeded) as exc_info:
        await mw.on_llm_end(resp, _make_ctx())

    assert exc_info.value.actual_cost == pytest.approx(0.01)
    assert exc_info.value.limit == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_tokens_exceed_budget_raises() -> None:
    """When accumulated tokens exceed max_total_tokens, TokenBudgetExceeded is raised."""
    config = CostBudgetConfig(max_total_tokens=500)
    mw = CostBudgetMiddleware(config)

    resp = MockResponse(input_tokens=300, output_tokens=300)
    # Total = 600 > 500
    with pytest.raises(TokenBudgetExceeded) as exc_info:
        await mw.on_llm_end(resp, _make_ctx())

    assert exc_info.value.actual_tokens == 600
    assert exc_info.value.limit == 500


@pytest.mark.asyncio
async def test_multiple_calls_accumulate() -> None:
    """Cost and tokens accumulate across multiple on_llm_end calls."""
    config = CostBudgetConfig(
        cost_per_million_input=2.0,
        cost_per_million_output=10.0,
    )
    mw = CostBudgetMiddleware(config)
    ctx = _make_ctx()

    await mw.on_llm_end(MockResponse(input_tokens=500, output_tokens=100), ctx)
    await mw.on_llm_end(MockResponse(input_tokens=500, output_tokens=100), ctx)

    # Each call: (500*2 + 100*10) / 1M = (1000+1000)/1M = 0.002
    # Two calls: 0.004 total
    assert mw.accumulated_cost == pytest.approx(0.004)
    assert mw.accumulated_tokens == 1200


@pytest.mark.asyncio
async def test_both_limits_cost_triggers_first() -> None:
    """When both limits are set, whichever is hit first triggers the exception."""
    config = CostBudgetConfig(
        max_cost_usd=0.001,
        max_total_tokens=100_000,
        cost_per_million_input=10.0,
    )
    mw = CostBudgetMiddleware(config)

    # Cost = 1000 * 10 / 1M = 0.01 > 0.001; tokens = 1000 < 100_000
    # Cost limit triggers first
    with pytest.raises(CostBudgetExceeded):
        await mw.on_llm_end(MockResponse(input_tokens=1000), _make_ctx())


@pytest.mark.asyncio
async def test_both_limits_tokens_trigger_first() -> None:
    """When both limits are set but token limit is lower, TokenBudgetExceeded fires."""
    config = CostBudgetConfig(
        max_cost_usd=100.0,
        max_total_tokens=50,
        cost_per_million_input=1.0,
    )
    mw = CostBudgetMiddleware(config)

    # Tokens = 100 > 50; cost = 100 * 1/1M ~ 0 < 100
    with pytest.raises(TokenBudgetExceeded):
        await mw.on_llm_end(MockResponse(input_tokens=100), _make_ctx())


@pytest.mark.asyncio
async def test_no_limits_set_never_raises() -> None:
    """When both limits are None, usage is tracked but no exception is raised."""
    config = CostBudgetConfig(
        cost_per_million_input=100.0,
        cost_per_million_output=100.0,
    )
    mw = CostBudgetMiddleware(config)
    ctx = _make_ctx()

    # Call many times with large token counts — should never raise
    for _ in range(10):
        resp = MockResponse(input_tokens=1_000_000, output_tokens=1_000_000)
        result = await mw.on_llm_end(resp, ctx)
        assert result is resp

    # But usage is still tracked
    assert mw.accumulated_tokens == 20_000_000
    assert mw.accumulated_cost == pytest.approx(2000.0)
