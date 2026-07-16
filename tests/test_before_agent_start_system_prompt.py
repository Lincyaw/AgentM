"""BeforeAgentStartEvent has one typed mutation channel."""

from __future__ import annotations

import pytest

from agentm.core.abi.bus import EventBus
from agentm.core.abi.events import BeforeAgentStartEvent


@pytest.mark.asyncio
async def test_mutation_only_handler() -> None:
    bus = EventBus()

    def handler(event: BeforeAgentStartEvent) -> None:
        event.system = "prompt B"

    bus.on(BeforeAgentStartEvent.CHANNEL, handler)
    event = BeforeAgentStartEvent(messages=[], system="")
    await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    assert event.system == "prompt B"


@pytest.mark.asyncio
async def test_return_value_is_not_a_second_system_prompt_channel() -> None:
    bus = EventBus()

    def handler(_event: BeforeAgentStartEvent) -> dict[str, str]:
        return {"system": "legacy"}

    bus.on(BeforeAgentStartEvent.CHANNEL, handler)
    event = BeforeAgentStartEvent(messages=[], system="")
    await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    assert event.system == ""


@pytest.mark.asyncio
async def test_chained_append_via_mutation() -> None:
    """Two handlers chain via mutation: each reads event.system and appends."""
    bus = EventBus()

    def handler_1(event: BeforeAgentStartEvent) -> None:
        accumulated = (event.system or "") + "\n\nblock1"
        event.system = accumulated

    def handler_2(event: BeforeAgentStartEvent) -> None:
        accumulated = (event.system or "") + "\n\nblock2"
        event.system = accumulated

    bus.on(BeforeAgentStartEvent.CHANNEL, handler_1)
    bus.on(BeforeAgentStartEvent.CHANNEL, handler_2)
    event = BeforeAgentStartEvent(messages=[], system="")
    await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    assert event.system == "\n\nblock1\n\nblock2"


@pytest.mark.asyncio
async def test_no_handlers_returns_empty() -> None:
    bus = EventBus()
    event = BeforeAgentStartEvent(messages=[], system="")
    await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    assert event.system == ""
