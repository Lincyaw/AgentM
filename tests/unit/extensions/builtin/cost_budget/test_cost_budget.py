from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.kernel import (
    AssistantMessage,
    EventBus,
    MessageEnd,
    Model,
    TextContent,
    UserMessage,
)
from agentm.harness.events import CostBudgetExceededEvent
from agentm.harness.extension import ProviderConfig, ReadonlySession, _ExtensionAPIImpl
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.session_manager import InMemorySessionManager
from tests.support.provider_registry import temporary_provider

from agentm.extensions.builtin import cost_budget


class _Session(ReadonlySession):
    def get_messages(self):
        return []

    def append_entry(self, type: str, payload: Any, parent_id: str | None = None) -> str:
        return "entry"


@pytest.mark.asyncio
async def test_handler_emits_cost_budget_exceeded_payload() -> None:
    bus = EventBus()
    tools: list[Any] = []
    commands: dict[str, Any] = {}
    providers: dict[str, ProviderConfig] = {}
    renderers: dict[str, Any] = {}
    pending: list[Any] = []
    model = Model(id="fake", provider="fake", context_window=1000, max_output_tokens=100)

    api = _ExtensionAPIImpl(
        bus=bus,
        cwd=".",
        session=_Session(),
        tools=tools,
        commands=commands,
        providers=providers,
        renderers=renderers,
        pending_user_messages=pending,
        model_getter=lambda: model,
        provider_getter=lambda: None,
    )

    seen: list[CostBudgetExceededEvent] = []
    bus.on("cost_budget_exceeded", lambda event: seen.append(event))
    cost_budget.install(api, {"limit": 0.0, "currency": "usd"})

    await bus.emit(
        "before_send_to_llm",
        cost_budget.BeforeSendToLlmEvent(
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(type="text", text="hello")],
                    timestamp=0.0,
                )
            ],
            model=model,
            tools=[],
            system=None,
        ),
    )

    assert len(seen) == 1
    assert seen[0].limit == 0.0
    assert seen[0].currency == "usd"
    assert seen[0].used > 0.0


@pytest.mark.asyncio
async def test_integration_prompt_emits_budget_agent_end(tmp_path) -> None:
    class FinalStream:
        def __call__(self, **_: Any) -> AsyncIterator[Any]:
            return self._iter()

        async def _iter(self) -> AsyncIterator[Any]:
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="done")],
                    timestamp=1.0,
                    stop_reason="end_turn",
                )
            )

    with temporary_provider(
        FinalStream(),
        provider_id="fake-budget",
        model_provider="fake",
    ) as provider_id:
        config = AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[("agentm.extensions.builtin.cost_budget", {"limit": 0.0})],
            provider=provider_id,
            resource_loader=InMemoryResourceLoader(),
            session_manager=InMemorySessionManager(),
        )
        session = await AgentSession.create(config)

        seen_budget: list[str] = []
        seen_events: list[CostBudgetExceededEvent] = []
        session.bus.on(
            "agent_end",
            lambda event: seen_budget.append(event.stop_reason),
        )
        session.bus.on(
            "cost_budget_exceeded",
            lambda event: seen_events.append(event),
        )

        await session.prompt("hello")

        assert seen_events
        assert seen_budget[-1] == "budget"

        await session.shutdown()
