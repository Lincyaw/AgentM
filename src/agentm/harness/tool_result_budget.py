# ruff: noqa: ARG002  — ctx parameter is required by the MiddlewareBase protocol
"""Middleware for enforcing per-tool and aggregate result size budgets.

Prevents context window blowup from large tool outputs (e.g., SQL dumps)
by truncating or persisting oversized results.

Ref: designs/loop-resilience.md, section 1.
"""
from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentm.harness.middleware import MiddlewareBase
from agentm.harness.types import LoopContext, Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class OverflowStrategy(StrEnum):
    """What to do when a tool result exceeds the size limit."""

    TRUNCATE = "truncate"
    PERSIST = "persist"


@dataclass(frozen=True)
class ToolResultBudgetConfig:
    """Configuration for tool result size budgets."""

    max_result_chars: int = 30_000
    max_aggregate_chars: int = 150_000
    preview_chars: int = 2_000
    overflow_strategy: OverflowStrategy = OverflowStrategy.TRUNCATE
    persist_dir: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_with_notice(
    result: str, original_size: int, preview_chars: int
) -> str:
    """Return an XML-tagged truncated result with a preview."""
    preview = result[:preview_chars]
    remaining = original_size - preview_chars
    return (
        "<truncated_result>\n"
        f"Output too large ({original_size:,} chars). "
        f"Showing first {preview_chars:,} chars:\n"
        f"\n{preview}\n...\n\n"
        f"[Remaining {remaining:,} chars truncated. "
        f"Refine your query to reduce output size.]\n"
        "</truncated_result>"
    )


def _persist_and_preview(
    result: str,
    original_size: int,
    preview_chars: int,
    persist_dir: str,
    tool_call_id: str,
) -> str:
    """Write full result to disk and return an XML-tagged preview with path."""
    os.makedirs(persist_dir, exist_ok=True)
    filepath = os.path.join(persist_dir, f"{tool_call_id}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(result)

    preview = result[:preview_chars]
    return (
        "<persisted_result>\n"
        f"Output too large ({original_size:,} chars). "
        f"Full output saved to: {filepath}\n"
        f"\nPreview (first {preview_chars:,} chars):\n"
        f"{preview}\n...\n"
        "</persisted_result>"
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class ToolResultBudgetMiddleware(MiddlewareBase):
    """Enforces per-tool and per-turn aggregate result size limits.

    Resets the aggregate counter at the start of each turn (on_llm_start).
    """

    def __init__(self, config: ToolResultBudgetConfig) -> None:
        self._config = config
        self._aggregate_chars: int = 0

    async def on_llm_start(
        self, messages: list[Message], ctx: LoopContext
    ) -> list[Message]:
        """Reset aggregate counter at the start of each new turn."""
        self._aggregate_chars = 0
        return messages

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        result = await call_next(tool_name, tool_args)
        original_size = len(result)

        if original_size <= self._config.max_result_chars:
            self._aggregate_chars += original_size
            # Even under per-result limit, check aggregate
            if self._aggregate_chars > self._config.max_aggregate_chars:
                logger.warning(
                    "Aggregate budget exceeded (%d/%d chars) — "
                    "force-truncating tool '%s' result",
                    self._aggregate_chars,
                    self._config.max_aggregate_chars,
                    tool_name,
                )
                result = _truncate_with_notice(
                    result, original_size, self._config.preview_chars
                )
                # Correct aggregate to reflect actual output size
                self._aggregate_chars += len(result) - original_size
            return result

        # Result exceeds per-tool limit
        logger.info(
            "Tool '%s' result exceeds budget (%d > %d chars)",
            tool_name,
            original_size,
            self._config.max_result_chars,
        )

        # Check if aggregate is already blown — force truncate regardless
        self._aggregate_chars += self._config.preview_chars
        if self._aggregate_chars > self._config.max_aggregate_chars:
            logger.warning(
                "Aggregate budget exceeded (%d/%d chars) — "
                "force-truncating regardless of strategy",
                self._aggregate_chars,
                self._config.max_aggregate_chars,
            )
            return _truncate_with_notice(
                result, original_size, self._config.preview_chars
            )

        # Apply configured strategy
        if self._config.overflow_strategy == OverflowStrategy.PERSIST:
            persist_dir = self._config.persist_dir
            if not persist_dir:
                logger.warning(
                    "PERSIST strategy requested but persist_dir is empty — "
                    "falling back to TRUNCATE"
                )
                return _truncate_with_notice(
                    result, original_size, self._config.preview_chars
                )
            tool_call_id = ctx.metadata.get("tool_call_id", "") or ""
            if not tool_call_id:
                tool_call_id = uuid.uuid4().hex[:12]
            return _persist_and_preview(
                result,
                original_size,
                self._config.preview_chars,
                persist_dir,
                str(tool_call_id),
            )

        # Default: TRUNCATE
        return _truncate_with_notice(
            result, original_size, self._config.preview_chars
        )
