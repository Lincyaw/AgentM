"""SystemReminderMiddleware — periodically re-injects constraint reminders.

Long-running agent loops can drift from their original instructions as the
conversation grows.  This middleware re-injects a configurable reminder block
into the system message at a fixed interval, starting only after the initial
instructions have had a chance to fade.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentm.harness.middleware import MiddlewareBase, inject_into_system_message
from agentm.harness.types import LoopContext, Message


@dataclass(frozen=True)
class SystemReminderConfig:
    """Configuration for periodic system-message reminders.

    Attributes:
        reminder_text: The constraint block to re-inject.
        interval: Re-inject every N steps.
        start_after: Skip the first N steps (constraints are fresh
            in the system prompt).
    """

    reminder_text: str
    interval: int = 5
    start_after: int = 5


class SystemReminderMiddleware(MiddlewareBase):
    """Re-injects a constraint reminder into the system message at a fixed cadence."""

    def __init__(self, config: SystemReminderConfig) -> None:
        self._config = config

    async def on_llm_start(
        self, messages: list[Message], ctx: LoopContext
    ) -> list[Message]:
        cfg = self._config
        if ctx.step < cfg.start_after:
            return messages
        if ctx.step % cfg.interval != 0:
            return messages
        return inject_into_system_message(messages, cfg.reminder_text)
