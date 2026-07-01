"""BeforeAgentStartEvent system prompt injection contract.

Protects the two channels for setting the system prompt:
1. Return: handler returns {"system": "..."} — runtime reads via collect_system_replacement
2. Mutation: handler sets event.system = "..." — used as fallback when no return

Runtime resolution: replacement (from return) or event.system (mutation fallback) or "".
The mutation-only path was silently dropping prompts before the fix.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm.core.abi.bus import EventBus
from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.core.runtime.session_helpers import collect_system_replacement


def _resolve_system_prompt(
    returns: list[Any], event: BeforeAgentStartEvent
) -> str:
    """Mirror the runtime resolution logic from session.py."""
    replacement = collect_system_replacement(returns)
    return replacement or event.system or ""


@pytest.mark.asyncio
async def test_return_only_handler() -> None:
    """Handler that returns {"system": ...} without mutating event.system."""
    bus = EventBus()

    def handler(event: BeforeAgentStartEvent) -> dict[str, str]:
        return {"system": "prompt A"}

    bus.on(BeforeAgentStartEvent.CHANNEL, handler)
    event = BeforeAgentStartEvent(messages=[], system="")
    returns = await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    assert _resolve_system_prompt(returns, event) == "prompt A"


@pytest.mark.asyncio
async def test_mutation_only_handler() -> None:
    """Handler that only mutates event.system without returning.

    This is the bug we fixed: before the fix, mutation-only handlers
    silently dropped the system prompt because the runtime only checked
    the return path (collect_system_replacement).
    """
    bus = EventBus()

    def handler(event: BeforeAgentStartEvent) -> None:
        event.system = "prompt B"

    bus.on(BeforeAgentStartEvent.CHANNEL, handler)
    event = BeforeAgentStartEvent(messages=[], system="")
    returns = await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    assert _resolve_system_prompt(returns, event) == "prompt B"


@pytest.mark.asyncio
async def test_both_mutation_and_return_return_wins() -> None:
    """Handler that both mutates and returns — return takes precedence."""
    bus = EventBus()

    def handler(event: BeforeAgentStartEvent) -> dict[str, str]:
        event.system = "X"
        return {"system": "Y"}

    bus.on(BeforeAgentStartEvent.CHANNEL, handler)
    event = BeforeAgentStartEvent(messages=[], system="")
    returns = await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    assert _resolve_system_prompt(returns, event) == "Y"


@pytest.mark.asyncio
async def test_chained_append_via_mutation_and_return() -> None:
    """Two handlers chain via mutation: each reads event.system and appends."""
    bus = EventBus()

    def handler_1(event: BeforeAgentStartEvent) -> dict[str, str]:
        accumulated = (event.system or "") + "\n\nblock1"
        event.system = accumulated
        return {"system": accumulated}

    def handler_2(event: BeforeAgentStartEvent) -> dict[str, str]:
        accumulated = (event.system or "") + "\n\nblock2"
        event.system = accumulated
        return {"system": accumulated}

    bus.on(BeforeAgentStartEvent.CHANNEL, handler_1)
    bus.on(BeforeAgentStartEvent.CHANNEL, handler_2)
    event = BeforeAgentStartEvent(messages=[], system="")
    returns = await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    result = _resolve_system_prompt(returns, event)
    assert "\n\nblock1" in result
    assert "\n\nblock2" in result


@pytest.mark.asyncio
async def test_mutation_only_does_not_see_return_from_prior_handler() -> None:
    """Return-only handler followed by mutation-only handler.

    Known limitation: handler 1 returns {"system": "base"} but does NOT
    mutate event.system, so handler 2 reads event.system as "" and
    produces "\\n\\nappended". collect_system_replacement picks "base"
    (handler 1's return) as the winner because it is the last non-None
    return. Handler 2's mutation ("\\n\\nappended") is shadowed.

    This documents the contract: if a handler needs downstream handlers
    to see its contribution, it must mutate event.system (not just return).
    """
    bus = EventBus()

    def handler_return_only(event: BeforeAgentStartEvent) -> dict[str, str]:
        return {"system": "base"}

    def handler_mutation_only(event: BeforeAgentStartEvent) -> None:
        event.system = (event.system or "") + "\n\nappended"

    bus.on(BeforeAgentStartEvent.CHANNEL, handler_return_only)
    bus.on(BeforeAgentStartEvent.CHANNEL, handler_mutation_only)
    event = BeforeAgentStartEvent(messages=[], system="")
    returns = await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    # Return wins over mutation — handler 1's return is the result.
    assert _resolve_system_prompt(returns, event) == "base"
    # Handler 2 only saw "" (not "base") because handler 1 didn't mutate.
    assert event.system == "\n\nappended"


@pytest.mark.asyncio
async def test_no_handlers_returns_empty() -> None:
    """No handlers registered — system prompt stays empty."""
    bus = EventBus()
    event = BeforeAgentStartEvent(messages=[], system="")
    returns = await bus.emit(BeforeAgentStartEvent.CHANNEL, event)

    assert _resolve_system_prompt(returns, event) == ""
