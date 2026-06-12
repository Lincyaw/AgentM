"""Retry when the model emits only thinking with no actionable output.

Some models (e.g. doubao-seed-2.0-code) occasionally emit tool-call
intentions inside ``reasoning_content`` instead of as structured
``tool_calls``. The default loop action treats this as a clean end-turn
(``ModelEndTurn``), terminating the session with no useful work.

This atom hooks ``decide_turn_action`` and overrides ``ModelEndTurn`` with
``Step()`` when the assistant message contains only ``ThinkingBlock``
content — no ``TextContent`` and no ``ToolCallBlock``. A per-session
counter caps consecutive thinking-only retries to prevent infinite loops.
"""

from __future__ import annotations

from loguru import logger
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    DecideTurnActionEvent,
    ExtensionAPI,
    ModelEndTurn,
    Step,
    Stop,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
)
from agentm.extensions import ExtensionManifest


class ThinkingRetryConfig(BaseModel):
    max_retries: int = 3

MANIFEST = ExtensionManifest(
    name="thinking_retry",
    description=(
        "Override ModelEndTurn with Step() when the assistant message "
        "contains only ThinkingBlock content (no text, no tool calls). "
        "Prevents models that leak tool-call intentions into reasoning "
        "from silently aborting the session."
    ),
    registers=("event:decide_turn_action",),
    config_schema=ThinkingRetryConfig,
    requires=(),
)

def _is_thinking_only(content: list[Any]) -> bool:
    has_thinking = False
    for block in content:
        if isinstance(block, ThinkingBlock):
            has_thinking = True
        elif isinstance(block, TextContent) and block.text.strip():
            return False
        elif isinstance(block, ToolCallBlock):
            return False
    return has_thinking

def install(api: ExtensionAPI, config: ThinkingRetryConfig) -> None:
    max_retries: int = config.max_retries
    consecutive_count = 0

    def on_decide(event: DecideTurnActionEvent) -> Step | None:
        nonlocal consecutive_count

        obs = event.observation
        default = obs.default_action

        if not (isinstance(default, Stop) and isinstance(default.cause, ModelEndTurn)):
            consecutive_count = 0
            return None

        msg = obs.assistant_message
        if msg is None or not _is_thinking_only(msg.content):
            consecutive_count = 0
            return None

        consecutive_count += 1
        if consecutive_count > max_retries:
            logger.warning(f"thinking_retry: exhausted {max_retries} retries; stopping")
            consecutive_count = 0
            return None

        logger.info(f"thinking_retry: thinking-only response ({consecutive_count}/{max_retries}); stepping")
        return Step()

    api.on(DecideTurnActionEvent.CHANNEL, on_decide)
