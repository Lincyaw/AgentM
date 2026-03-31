"""Tests for SystemReminderMiddleware.

Bug prevented: long-running agent loops lose track of constraints as the
conversation grows; periodic re-injection keeps instructions fresh.
"""

from __future__ import annotations

import pytest

from agentm.harness.system_reminder import SystemReminderConfig, SystemReminderMiddleware
from agentm.harness.types import JsonValue, LoopContext, Message

REMINDER = "<system-reminder>Stay focused.</system-reminder>"


def _make_ctx(step: int) -> LoopContext:
    """Build a minimal LoopContext at the given step."""
    return LoopContext(
        agent_id="test-agent",
        step=step,
        max_steps=100,
        tool_call_count=0,
        metadata={},
    )


def _system_msg(content: str = "You are helpful.") -> dict[str, JsonValue]:
    return {"role": "system", "content": content}


def _user_msg(content: str = "Hello") -> dict[str, JsonValue]:
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Injection timing — start_after / interval gating
# ---------------------------------------------------------------------------


class TestInjectionTiming:
    """Bug: reminder injected too early (instructions still fresh) or at
    wrong cadence wastes context tokens or drifts between injections."""

    @pytest.mark.asyncio
    async def test_no_injection_before_start_after(self) -> None:
        mw = SystemReminderMiddleware(
            SystemReminderConfig(reminder_text=REMINDER, interval=5, start_after=5)
        )
        msgs: list[Message] = [_system_msg(), _user_msg()]

        for step in range(5):  # steps 0-4
            result = await mw.on_llm_start(list(msgs), _make_ctx(step=step))
            assert result == msgs, f"Injection should not happen at step {step}"

    @pytest.mark.asyncio
    async def test_injection_at_start_after_with_matching_interval(self) -> None:
        mw = SystemReminderMiddleware(
            SystemReminderConfig(reminder_text=REMINDER, interval=5, start_after=5)
        )
        msgs: list[Message] = [_system_msg(), _user_msg()]

        result = await mw.on_llm_start(list(msgs), _make_ctx(step=5))
        sys_content: str = result[0]["content"]  # type: ignore[index,assignment]
        assert REMINDER in sys_content

    @pytest.mark.asyncio
    async def test_no_injection_at_start_after_when_interval_mismatches(self) -> None:
        mw = SystemReminderMiddleware(
            SystemReminderConfig(reminder_text=REMINDER, interval=3, start_after=5)
        )
        msgs: list[Message] = [_system_msg(), _user_msg()]

        # step=5 and interval=3: 5 % 3 == 2 != 0 → no injection
        result = await mw.on_llm_start(list(msgs), _make_ctx(step=5))
        assert result == msgs

    @pytest.mark.asyncio
    async def test_injection_at_matching_interval_after_start(self) -> None:
        mw = SystemReminderMiddleware(
            SystemReminderConfig(reminder_text=REMINDER, interval=3, start_after=5)
        )
        msgs: list[Message] = [_system_msg(), _user_msg()]

        # step=6: 6 >= 5 and 6 % 3 == 0 → injection
        result = await mw.on_llm_start(list(msgs), _make_ctx(step=6))
        sys_content: str = result[0]["content"]  # type: ignore[index,assignment]
        assert REMINDER in sys_content

    @pytest.mark.asyncio
    async def test_no_injection_when_interval_does_not_match(self) -> None:
        mw = SystemReminderMiddleware(
            SystemReminderConfig(reminder_text=REMINDER, interval=5, start_after=5)
        )
        msgs: list[Message] = [_system_msg(), _user_msg()]

        # step=7: 7 % 5 == 2 → no injection
        result = await mw.on_llm_start(list(msgs), _make_ctx(step=7))
        assert result == msgs

    @pytest.mark.asyncio
    async def test_interval_one_injects_every_step_after_start(self) -> None:
        mw = SystemReminderMiddleware(
            SystemReminderConfig(reminder_text=REMINDER, interval=1, start_after=3)
        )
        msgs: list[Message] = [_system_msg(), _user_msg()]

        for step in range(3, 8):
            result = await mw.on_llm_start(list(msgs), _make_ctx(step=step))
            sys_content: str = result[0]["content"]  # type: ignore[index,assignment]
            assert REMINDER in sys_content, f"Should inject at step {step}"


# ---------------------------------------------------------------------------
# System message content behaviour
# ---------------------------------------------------------------------------


class TestSystemMessageContent:
    """Bug: reminder replaces system message instead of appending, or gets
    lost when no system message exists."""

    @pytest.mark.asyncio
    async def test_appends_to_existing_system_message(self) -> None:
        mw = SystemReminderMiddleware(
            SystemReminderConfig(reminder_text=REMINDER, interval=5, start_after=0)
        )
        original = "You are a helpful assistant."
        msgs: list[Message] = [_system_msg(original), _user_msg()]

        result = await mw.on_llm_start(list(msgs), _make_ctx(step=0))

        sys_content: str = result[0]["content"]  # type: ignore[index,assignment]
        assert sys_content.startswith(original)
        assert REMINDER in sys_content
        # Non-system messages preserved
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_prepends_system_message_when_none_exists(self) -> None:
        mw = SystemReminderMiddleware(
            SystemReminderConfig(reminder_text=REMINDER, interval=5, start_after=0)
        )
        msgs: list[Message] = [_user_msg()]

        result = await mw.on_llm_start(list(msgs), _make_ctx(step=0))

        assert len(result) == 2
        assert result[0]["role"] == "system"  # type: ignore[index]
        assert result[0]["content"] == REMINDER  # type: ignore[index]
        # Original user message is still at position 1
        assert result[1] == msgs[0]
