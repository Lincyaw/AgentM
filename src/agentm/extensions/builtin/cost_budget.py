"""Policy-gate atom for the §7 ``extensions.builtin.cost_budget`` row.

Uses a coarse ``len(json.dumps(messages)) // 4`` heuristic for input tokens on
``before_send_to_llm`` because the v2 kernel intentionally has no tokenizer.
When accumulated spend crosses the configured limit, this extension emits the
custom ``cost_budget_exceeded`` event instead of throwing across handlers.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from typing import Any

from agentm.core.kernel import BeforeSendToLlmEvent, TurnEndEvent
from agentm.extensions import ExtensionManifest
from agentm.harness.events import CostBudgetExceededEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="cost_budget",
    description="Track estimated LLM spend and emit cost_budget_exceeded on overflow.",
    registers=(
        "event:before_send_to_llm",
        "event:turn_end",
        "event:cost_budget_exceeded",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "number", "minimum": 0},
            "currency": {"type": "string"},
        },
        "required": ["limit"],
        "additionalProperties": True,
    },
    tier=2,
)


_PRICING: dict[str, tuple[float, float]] = {
    "anthropic": (15.0, 75.0),
    "fake": (1.0, 1.0),
}


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _estimate_input_tokens(messages: list[Any]) -> int:
    encoded = json.dumps(_serialize(messages), default=str, sort_keys=True)
    return len(encoded) // 4


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    limit = float(config.get("limit", 0.0))
    currency = str(config.get("currency", "usd"))
    state = {
        "used": 0.0,
        "overflowed": False,
    }

    async def _emit_if_needed() -> None:
        if state["overflowed"] or state["used"] <= limit:
            return
        state["overflowed"] = True
        await api.events.emit(
            "cost_budget_exceeded",
            CostBudgetExceededEvent(
                used=state["used"],
                limit=limit,
                currency=currency,
            ),
        )

    async def _before_send(event: BeforeSendToLlmEvent) -> None:
        pricing = _PRICING.get(event.model.provider, (0.0, 0.0))
        estimated_input_tokens = _estimate_input_tokens(event.messages)
        state["used"] += (estimated_input_tokens / 1_000_000.0) * pricing[0]
        await _emit_if_needed()

    async def _on_turn_end(event: TurnEndEvent) -> None:
        usage = event.message.usage
        if usage is None:
            return
        pricing = _PRICING.get(api.model.provider if api.model is not None else "", (0.0, 0.0))
        state["used"] += (usage.output_tokens / 1_000_000.0) * pricing[1]
        await _emit_if_needed()

    api.on("before_send_to_llm", _before_send)
    api.on("turn_end", _on_turn_end)
