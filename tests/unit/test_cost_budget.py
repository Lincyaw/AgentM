"""Focused regression tests for CostBudgetMiddleware accounting and limits."""
from __future__ import annotations

import pytest

from agentm.harness.cost_budget import CostBudgetConfig, CostBudgetExceeded, CostBudgetMiddleware, TokenBudgetExceeded
from agentm.harness.types import LoopContext


class MockResponse:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}


class BareResponse:
    pass


def _ctx() -> LoopContext:
    return LoopContext(agent_id="test-agent", step=0, max_steps=10, tool_call_count=0, metadata={})


@pytest.mark.asyncio
async def test_response_without_usage_metadata_passes_through() -> None:
    mw = CostBudgetMiddleware(CostBudgetConfig(max_cost_usd=1.0))
    resp = BareResponse()
    assert await mw.on_llm_end(resp, _ctx()) is resp
    assert mw.accumulated_cost == 0.0
    assert mw.accumulated_tokens == 0


@pytest.mark.asyncio
async def test_usage_cost_and_tokens_accumulate_across_calls() -> None:
    mw = CostBudgetMiddleware(CostBudgetConfig(cost_per_million_input=2.0, cost_per_million_output=10.0))
    await mw.on_llm_end(MockResponse(input_tokens=500, output_tokens=100), _ctx())
    await mw.on_llm_end(MockResponse(input_tokens=500, output_tokens=100), _ctx())
    assert mw.accumulated_cost == pytest.approx(0.004)
    assert mw.accumulated_tokens == 1200


@pytest.mark.asyncio
async def test_exceeding_cost_limit_raises_cost_budget_exceeded() -> None:
    mw = CostBudgetMiddleware(CostBudgetConfig(max_cost_usd=0.001, cost_per_million_input=10.0))
    with pytest.raises(CostBudgetExceeded) as exc:
        await mw.on_llm_end(MockResponse(input_tokens=1000, output_tokens=0), _ctx())
    assert exc.value.actual_cost == pytest.approx(0.01)
    assert exc.value.limit == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_exceeding_token_limit_raises_token_budget_exceeded() -> None:
    mw = CostBudgetMiddleware(CostBudgetConfig(max_total_tokens=500))
    with pytest.raises(TokenBudgetExceeded) as exc:
        await mw.on_llm_end(MockResponse(input_tokens=300, output_tokens=300), _ctx())
    assert exc.value.actual_tokens == 600
    assert exc.value.limit == 500


@pytest.mark.asyncio
async def test_when_both_limits_set_the_first_hit_limit_is_raised() -> None:
    cost_first = CostBudgetMiddleware(
        CostBudgetConfig(max_cost_usd=0.001, max_total_tokens=100_000, cost_per_million_input=10.0)
    )
    with pytest.raises(CostBudgetExceeded):
        await cost_first.on_llm_end(MockResponse(input_tokens=1000), _ctx())

    token_first = CostBudgetMiddleware(
        CostBudgetConfig(max_cost_usd=100.0, max_total_tokens=50, cost_per_million_input=1.0)
    )
    with pytest.raises(TokenBudgetExceeded):
        await token_first.on_llm_end(MockResponse(input_tokens=100), _ctx())
