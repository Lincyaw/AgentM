"""Cost and token budget middleware for agent loops.

Tracks accumulated LLM usage (tokens and cost) across loop iterations and
raises when configurable limits are exceeded.
"""
from __future__ import annotations

from dataclasses import dataclass

from agentm.exceptions import AgentMError
from agentm.harness.middleware import MiddlewareBase
from agentm.harness.types import LoopContext


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExceeded(AgentMError):
    """Base exception for budget limit violations."""


class CostBudgetExceeded(BudgetExceeded):
    """Raised when accumulated USD cost exceeds the configured limit."""

    def __init__(self, actual_cost: float, limit: float) -> None:
        self.actual_cost = actual_cost
        self.limit = limit
        super().__init__(
            f"Cost budget exceeded: ${actual_cost:.6f} > ${limit:.6f}"
        )


class TokenBudgetExceeded(BudgetExceeded):
    """Raised when accumulated token count exceeds the configured limit."""

    def __init__(self, actual_tokens: int, limit: int) -> None:
        self.actual_tokens = actual_tokens
        self.limit = limit
        super().__init__(
            f"Token budget exceeded: {actual_tokens} > {limit}"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostBudgetConfig:
    """Configuration for cost and token budget limits.

    Attributes:
        max_cost_usd: Maximum allowed accumulated cost in USD, or None for unlimited.
        max_total_tokens: Maximum allowed accumulated tokens, or None for unlimited.
        cost_per_million_input: Cost in USD per 1M input tokens.
        cost_per_million_output: Cost in USD per 1M output tokens.
    """

    max_cost_usd: float | None = None
    max_total_tokens: int | None = None
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class CostBudgetMiddleware(MiddlewareBase):
    """Tracks LLM usage costs and enforces budget limits.

    Extracts ``usage_metadata`` from LLM responses to accumulate token counts
    and compute costs.  Raises :class:`CostBudgetExceeded` or
    :class:`TokenBudgetExceeded` when limits are breached.
    """

    def __init__(self, config: CostBudgetConfig) -> None:
        self._config = config
        self._accumulated_cost: float = 0.0
        self._accumulated_tokens: int = 0

    # -- Read-only properties ------------------------------------------------

    @property
    def accumulated_cost(self) -> float:
        """Total accumulated cost in USD so far."""
        return self._accumulated_cost

    @property
    def accumulated_tokens(self) -> int:
        """Total accumulated tokens (input + output) so far."""
        return self._accumulated_tokens

    # -- Middleware hook ------------------------------------------------------

    async def on_llm_end(self, response: object, ctx: LoopContext) -> object:
        """Extract usage from *response* and enforce budget limits."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return response

        # usage may be a dict or an object with attributes
        if isinstance(usage, dict):
            input_tokens: int = usage.get("input_tokens", 0)
            output_tokens: int = usage.get("output_tokens", 0)
        else:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)

        step_cost = (
            input_tokens * self._config.cost_per_million_input
            + output_tokens * self._config.cost_per_million_output
        ) / 1_000_000

        self._accumulated_cost += step_cost
        self._accumulated_tokens += input_tokens + output_tokens

        # Check cost limit
        if (
            self._config.max_cost_usd is not None
            and self._accumulated_cost > self._config.max_cost_usd
        ):
            raise CostBudgetExceeded(
                actual_cost=self._accumulated_cost,
                limit=self._config.max_cost_usd,
            )

        # Check token limit
        if (
            self._config.max_total_tokens is not None
            and self._accumulated_tokens > self._config.max_total_tokens
        ):
            raise TokenBudgetExceeded(
                actual_tokens=self._accumulated_tokens,
                limit=self._config.max_total_tokens,
            )

        return response
