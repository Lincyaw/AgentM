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

def install(api: ExtensionAPI, config: CostBudgetConfig) -> None:
    limit = config.limit
    currency = config.currency
    pricing = dict(config.pricing or {})
    warned_unpriced: set[str] = set()
    state = {
        "used": 0.0,
        "overflowed": False,
    }

    class _CostQueryService:
        def estimate(self, usage: Any, *, provider: str | None = None) -> CostBreakdown:
            selected = provider or (api.model.provider if api.model is not None else "")
            input_price, output_price = pricing.get(selected, (0.0, 0.0))
            amount = (
                (getattr(usage, "input_tokens", 0) / 1_000_000.0) * input_price
                + (getattr(usage, "output_tokens", 0) / 1_000_000.0) * output_price
            )
            return CostBreakdown(amount=amount, currency=currency)

    from agentm.core.abi import COST_QUERY_SERVICE
    api.set_service(COST_QUERY_SERVICE, _CostQueryService())

    async def _pricing_for(provider: str) -> tuple[float, float]:
        configured = pricing.get(provider)
        if configured is not None:
            return configured
        if provider not in warned_unpriced:
            warned_unpriced.add(provider)
            await api.events.emit(
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

    async def _emit_if_needed() -> None:
        if state["overflowed"] or state["used"] <= limit:
            return
        state["overflowed"] = True
        await api.events.emit(
            CostBudgetExceededEvent.CHANNEL,
            CostBudgetExceededEvent(
                used=state["used"],
                limit=limit,
                currency=currency,
            ),
        )

    def _before_agent_start(event: BeforeAgentStartEvent) -> dict[str, object] | None:
        if not state["overflowed"]:
            return None
        event.veto = BudgetExhausted(detail="cost")
        return {"block": True, "cause": BudgetExhausted(detail="cost")}

    async def _on_turn_end(event: TurnEndEvent) -> None:
        usage = event.message.usage
        if usage is None:
            return
        provider = api.model.provider if api.model is not None else ""
        provider_pricing = await _pricing_for(provider)
        state["used"] += (
            (usage.input_tokens / 1_000_000.0) * provider_pricing[0]
            + (usage.output_tokens / 1_000_000.0) * provider_pricing[1]
        )
        await _emit_if_needed()

    api.on(BeforeAgentStartEvent.CHANNEL, _before_agent_start)
    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
