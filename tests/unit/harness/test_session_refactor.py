from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from pathlib import Path
import sys
import types
from typing import Any

import pytest

from agentm.core.abi import AssistantMessage, MessageEnd, Model, TextContent
from agentm.harness.extension import ProviderConfig
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.session_manager import SessionEntry, SessionManager


async def _stream_fn(
    *,
    messages: list[Any],
    model: Model,
    tools: list[Any],
    system: str | None = None,
    signal: Any = None,
    thinking: str = "off",
) -> AsyncIterator[Any]:
    del messages, model, tools, system, signal, thinking
    yield MessageEnd(
        message=AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="ok")],
            timestamp=0.0,
            stop_reason="end_turn",
        )
    )


class _PickFirstResolver:
    def resolve_provider(self, providers: Mapping[str, Any]) -> str | None:
        del providers
        return "first"


@pytest.mark.asyncio
async def test_custom_provider_resolver_selects_named_provider(tmp_path: Path) -> None:
    module_name = f"tests.unit.harness._provider_resolver_{id(tmp_path)}"
    module = types.ModuleType(module_name)

    def install(api: Any, config: dict[str, Any]) -> None:
        del config
        api.register_provider(
            "second",
            ProviderConfig(
                stream_fn=_stream_fn,
                model=Model(
                    id="second-model",
                    provider="second-provider",
                    context_window=1024,
                    max_output_tokens=64,
                ),
                name="second",
            ),
        )
        api.register_provider(
            "first",
            ProviderConfig(
                stream_fn=_stream_fn,
                model=Model(
                    id="first-model",
                    provider="first-provider",
                    context_window=1024,
                    max_output_tokens=64,
                ),
                name="first",
            ),
        )

    setattr(module, "install", install)
    sys.modules[module_name] = module
    try:
        session = await AgentSession.create(
            AgentSessionConfig(
                cwd=str(tmp_path),
                provider=(module_name, {}),
                provider_resolver=_PickFirstResolver(),
                no_extensions=True,
            )
        )
        assert session.model is not None
        assert session.model.id == "first-model"
    finally:
        sys.modules.pop(module_name, None)


def test_fork_at_deep_copies_entry_payloads(tmp_path: Path) -> None:
    manager = SessionManager.in_memory(str(tmp_path))
    entry = SessionEntry(
        type="custom",
        id="entry-1",
        parent_id=None,
        timestamp=0.0,
        payload={"nested": {"value": "parent"}},
    )
    manager.append(entry)

    fork = manager.fork_at(entry.id)
    fork_entry = fork.get_entry(entry.id)
    assert fork_entry is not None
    fork_entry.payload["nested"]["value"] = "fork"

    parent_entry = manager.get_entry(entry.id)
    assert parent_entry is not None
    assert parent_entry.payload["nested"]["value"] == "parent"


@pytest.mark.asyncio
async def test_cost_budget_veto_emits_budget_exhausted_agent_end(
    tmp_path: Path,
) -> None:
    from agentm.core.abi import AgentEndEvent, BudgetExhausted

    module_name = f"tests.unit.harness._budget_provider_{id(tmp_path)}"
    module = types.ModuleType(module_name)

    def install(api: Any, config: dict[str, Any]) -> None:
        del config
        api.register_provider(
            "budget-provider",
            ProviderConfig(
                stream_fn=_stream_fn,
                model=Model(
                    id="budget-model",
                    provider="fake",
                    context_window=1024,
                    max_output_tokens=64,
                ),
                name="budget-provider",
            ),
        )

    setattr(module, "install", install)
    sys.modules[module_name] = module
    causes: list[Any] = []
    try:
        session = await AgentSession.create(
            AgentSessionConfig(
                cwd=str(tmp_path),
                provider=(module_name, {}),
                extensions=[
                    (
                        "agentm.extensions.builtin.cost_budget",
                        {"limit": 0, "pricing": {"fake": (1.0, 1.0)}},
                    )
                ],
            )
        )
        session.bus.on(AgentEndEvent.CHANNEL, lambda event: causes.append(event.cause))

        await session.prompt("first turn crosses the budget")
        await session.prompt("second turn is vetoed")

        assert any(isinstance(cause, BudgetExhausted) for cause in causes)
    finally:
        sys.modules.pop(module_name, None)
