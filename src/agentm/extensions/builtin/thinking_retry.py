"""Retry when the model emits only thinking with no actionable output.

Some models (e.g. doubao-seed-2.0-code) occasionally emit tool-call
intentions inside ``reasoning_content`` instead of as structured
``tool_calls``. The default loop action treats this as a clean end-turn
(``ModelEndTurn``), terminating the session with no useful work.

This atom hooks ``decide`` and overrides ``Stop(ModelEndTurn)`` with
``Step()`` when the assistant message contains only ``ThinkingBlock``
content — no ``TextContent`` and no ``ToolCallBlock``. A per-session
counter caps consecutive thinking-only retries to prevent infinite loops.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    DecideEvent,
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
        "Override Stop(ModelEndTurn) with Step() when the assistant message "
        "contains only ThinkingBlock content (no text, no tool calls). "
        "Prevents models that leak tool-call intentions into reasoning "
        "from silently aborting the session."
    ),
    registers=("event:decide",),
    config_schema=ThinkingRetryConfig,
    requires=(),
    priority=AtomInstallPriority.POLICY,
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


class _ThinkingRetryRuntime:
    def __init__(self, api: AtomAPI, config: ThinkingRetryConfig) -> None:
        self._api = api
        self._max_retries = config.max_retries
        self._consecutive_count = 0

    def install(self) -> None:
        self._api.on(DecideEvent.CHANNEL, self.on_decide)

    def on_decide(self, event: DecideEvent) -> Step | None:
        obs = event.observation
        default = obs.default_action

        if not (isinstance(default, Stop) and isinstance(default.cause, ModelEndTurn)):
            self._reset()
            return None

        msg = obs.assistant_message
        if msg is None or not _is_thinking_only(list(msg.content)):
            self._reset()
            return None

        self._consecutive_count += 1
        if self._consecutive_count > self._max_retries:
            logger.warning(
                f"thinking_retry: exhausted {self._max_retries} retries; stopping"
            )
            self._reset()
            return None

        logger.info(
            "thinking_retry: thinking-only response "
            f"({self._consecutive_count}/{self._max_retries}); stepping"
        )
        return Step()

    def _reset(self) -> None:
        self._consecutive_count = 0


def install(api: AtomAPI, config: ThinkingRetryConfig) -> None:
    _ThinkingRetryRuntime(api, config).install()
