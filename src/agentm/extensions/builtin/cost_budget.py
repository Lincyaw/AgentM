"""Policy-gate atom for the §7 ``extensions.builtin.cost_budget`` row.

Uses a coarse ``len(json.dumps(messages)) // 4`` heuristic for input tokens on
``before_send_to_llm`` because the v2 kernel intentionally has no tokenizer.
When accumulated spend crosses the configured limit, this extension emits the
custom ``cost_budget_exceeded`` event instead of throwing across handlers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol

from agentm.core.abi import BeforeSendToLlmEvent, BudgetExhausted, TurnEndEvent
from agentm.core.abi.events import DiagnosticEvent
from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent, CostBudgetExceededEvent
from agentm.harness.extension import ExtensionAPI


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


MANIFEST = ExtensionManifest(
    name="cost_budget",
    description="Track estimated LLM spend and emit cost_budget_exceeded on overflow.",
    registers=(
        "event:before_agent_start",
        "event:before_send_to_llm",
        "event:turn_end",
        "event:cost_budget_exceeded",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "number", "minimum": 0},
            "currency": {"type": "string"},
            "pricing": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "prefixItems": [{"type": "number"}, {"type": "number"}],
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        "required": ["limit"],
        "additionalProperties": True,
    },
    requires=(),  # Leaf atom: consumes model/events only.
    tier=2,
)


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _estimate_input_tokens(messages: list[Any]) -> int:
    encoded = json.dumps(_serialize(messages), default=str, sort_keys=True)
    return len(encoded) // 4


def _coerce_pricing(raw: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(raw, dict):
        return {}
    pricing: dict[str, tuple[float, float]] = {}
    for provider, pair in raw.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        pricing[str(provider)] = (float(pair[0]), float(pair[1]))
    return pricing


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # ``limit`` is in ``MANIFEST.config_schema.required``, so the discovery
    # filter (session.py:_atom_requires_unsupplied_config) skips this atom
    # when no explicit config was supplied. ``install`` therefore assumes
    # the key is present.
    limit = float(config["limit"])
    currency = str(config.get("currency", "usd"))
    pricing = _coerce_pricing(config.get("pricing"))
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

    api.set_service("cost_query", _CostQueryService())

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

    def _before_agent_start(_: BeforeAgentStartEvent) -> dict[str, object] | None:
        if not state["overflowed"]:
            return None
        return {"block": True, "cause": BudgetExhausted(detail="cost")}

    async def _before_send(event: BeforeSendToLlmEvent) -> None:
        provider_pricing = await _pricing_for(event.model.provider)
        estimated_input_tokens = _estimate_input_tokens(event.messages)
        state["used"] += (estimated_input_tokens / 1_000_000.0) * provider_pricing[0]
        await _emit_if_needed()

    async def _on_turn_end(event: TurnEndEvent) -> None:
        usage = event.message.usage
        if usage is None:
            return
        provider = api.model.provider if api.model is not None else ""
        provider_pricing = await _pricing_for(provider)
        state["used"] += (usage.output_tokens / 1_000_000.0) * provider_pricing[1]
        await _emit_if_needed()

    api.on(BeforeAgentStartEvent.CHANNEL, _before_agent_start)
    api.on(BeforeSendToLlmEvent.CHANNEL, _before_send)
    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
