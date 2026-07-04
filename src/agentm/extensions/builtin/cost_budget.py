"""Policy-gate atom for the §7 ``extensions.builtin.cost_budget`` row.

Uses provider-reported usage tokens only; it does not infer tokens from
character counts. When accumulated spend crosses the configured limit, this
extension emits the custom ``cost_budget_exceeded`` event instead of throwing
across handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    COST_QUERY_SERVICE,
    BeforeAgentStartEvent,
    BudgetExhausted,
    CostBudgetExceededEvent,
    DiagnosticEvent,
    ExtensionAPI,
    TurnEndEvent,
)
from agentm.extensions import ExtensionManifest

# ---------------------------------------------------------------------------
# Cost-query service contract — inlined from the former
# agentm.core.abi.services. cost_budget is the sole consumer; the CLI and
# textual_app fetch the service via api.get_service("cost_query") string
# lookup, not via Protocol import, so this stays scoped to the atom.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    amount: float
    currency: str = "usd"


class CostQueryService(Protocol):
    def estimate(self, usage: Any, *, provider: str | None = None) -> CostBreakdown: ...


class CostBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    limit: float = Field(gt=0)
    currency: str = "usd"
    pricing: dict[str, tuple[float, float]] | None = None


@dataclass(slots=True)
class _CostBudgetState:
    used: float = 0.0
    overflowed: bool = False


MANIFEST = ExtensionManifest(
    name="cost_budget",
    description=(
        "Track LLM spend from provider usage tokens and emit "
        "cost_budget_exceeded on overflow."
    ),
    registers=(
        "event:before_agent_start",
        "event:turn_end",
        "event:cost_budget_exceeded",
    ),
    config_schema=CostBudgetConfig,
    requires=(),  # Leaf atom: consumes model/events only.
    tier=2,
)


class _RuntimeCostQueryService:
    def __init__(self, runtime: "_CostBudgetRuntime") -> None:
        self._runtime = runtime

    def estimate(self, usage: Any, *, provider: str | None = None) -> CostBreakdown:
        return self._runtime.estimate(usage, provider=provider)


class _CostBudgetRuntime:
    def __init__(self, api: ExtensionAPI, config: CostBudgetConfig) -> None:
        self._api = api
        self._limit = config.limit
        self._currency = config.currency
        self._pricing = dict(config.pricing or {})
        self._warned_unpriced: set[str] = set()
        self._state = _CostBudgetState()

    def install(self) -> None:
        self._api.set_service(COST_QUERY_SERVICE, _RuntimeCostQueryService(self))
        self._api.on(BeforeAgentStartEvent.CHANNEL, self.before_agent_start)
        self._api.on(TurnEndEvent.CHANNEL, self.on_turn_end)

    def estimate(self, usage: Any, *, provider: str | None = None) -> CostBreakdown:
        selected = provider or self._current_provider()
        amount = _usage_amount(usage, self._pricing.get(selected, (0.0, 0.0)))
        return CostBreakdown(amount=amount, currency=self._currency)

    def before_agent_start(
        self, event: BeforeAgentStartEvent
    ) -> dict[str, object] | None:
        if not self._state.overflowed:
            return None
        event.veto = BudgetExhausted(detail="cost")
        return {"block": True, "cause": BudgetExhausted(detail="cost")}

    async def on_turn_end(self, event: TurnEndEvent) -> None:
        usage = event.message.usage
        if usage is None:
            return
        provider_pricing = await self._pricing_for(self._current_provider())
        self._state.used += _usage_amount(usage, provider_pricing)
        await self._emit_if_needed()

    def _current_provider(self) -> str:
        return self._api.model.provider if self._api.model is not None else ""

    async def _pricing_for(self, provider: str) -> tuple[float, float]:
        configured = self._pricing.get(provider)
        if configured is not None:
            return configured
        if provider not in self._warned_unpriced:
            self._warned_unpriced.add(provider)
            await self._api.events.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="warning",
                    source="cost_budget",
                    message=(
                        f"cost_budget has no pricing for provider {provider!r}; "
                        "usage for that provider is counted as zero"
                    ),
                ),
            )
        return (0.0, 0.0)

    async def _emit_if_needed(self) -> None:
        if self._state.overflowed or self._state.used <= self._limit:
            return
        self._state.overflowed = True
        await self._api.events.emit(
            CostBudgetExceededEvent.CHANNEL,
            CostBudgetExceededEvent(
                used=self._state.used,
                limit=self._limit,
                currency=self._currency,
            ),
        )


def _usage_amount(usage: Any, pricing: tuple[float, float]) -> float:
    input_price, output_price = pricing
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    return (input_tokens / 1_000_000.0) * input_price + (
        output_tokens / 1_000_000.0
    ) * output_price


def install(api: ExtensionAPI, config: CostBudgetConfig) -> None:
    _CostBudgetRuntime(api, config).install()
