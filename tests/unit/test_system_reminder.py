"""Focused tests for SystemReminderMiddleware cadence and injection behavior."""

from __future__ import annotations

import pytest

from agentm.harness.system_reminder import SystemReminderConfig, SystemReminderMiddleware
from agentm.harness.types import JsonValue, LoopContext, Message

REMINDER = "<system-reminder>Stay focused.</system-reminder>"


def _ctx(step: int) -> LoopContext:
    return LoopContext(agent_id="test-agent", step=step, max_steps=100, tool_call_count=0, metadata={})


def _system(content: str = "You are helpful.") -> dict[str, JsonValue]:
    return {"role": "system", "content": content}


def _user(content: str = "Hello") -> dict[str, JsonValue]:
    return {"role": "user", "content": content}


@pytest.mark.asyncio
async def test_reminder_not_injected_before_start_after() -> None:
    mw = SystemReminderMiddleware(SystemReminderConfig(reminder_text=REMINDER, interval=5, start_after=5))
    msgs: list[Message] = [_system(), _user()]
    for step in range(5):
        assert await mw.on_llm_start(list(msgs), _ctx(step=step)) == msgs


@pytest.mark.asyncio
async def test_reminder_injected_when_start_after_and_interval_match() -> None:
    mw = SystemReminderMiddleware(SystemReminderConfig(reminder_text=REMINDER, interval=5, start_after=5))
    result = await mw.on_llm_start([_system(), _user()], _ctx(step=5))
    assert REMINDER in result[0]["content"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_reminder_appends_to_existing_system_message() -> None:
    mw = SystemReminderMiddleware(SystemReminderConfig(reminder_text=REMINDER, interval=1, start_after=0))
    result = await mw.on_llm_start([_system("Base system"), _user()], _ctx(step=0))
    assert result[0]["content"].startswith("Base system")  # type: ignore[index]
    assert REMINDER in result[0]["content"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_reminder_prepends_system_when_missing() -> None:
    mw = SystemReminderMiddleware(SystemReminderConfig(reminder_text=REMINDER, interval=1, start_after=0))
    result = await mw.on_llm_start([_user()], _ctx(step=0))
    assert result[0]["role"] == "system"  # type: ignore[index]
    assert result[0]["content"] == REMINDER  # type: ignore[index]
    assert result[1]["role"] == "user"  # type: ignore[index]
