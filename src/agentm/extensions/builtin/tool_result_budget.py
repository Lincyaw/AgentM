"""Policy-gate atom for the §7 ``extensions.builtin.tool_result_budget`` row.

Truncates oversized text payloads on ``tool_result`` while preserving any
image content blocks untouched.  Uses **middle-out** truncation: the first
half and last half of the text budget are preserved so the model sees both
the initial context (headers, imports) and the trailing state (errors,
final output).  Tool results flagged ``is_error=True`` are guaranteed at
least ``_ERROR_FLOOR`` characters of payload so block-reason messages
survive aggressive ``max_chars`` budgets.
"""

from __future__ import annotations


from pydantic import BaseModel

from agentm.core.abi import (
    ExtensionAPI,
    ImageContent,
    TextContent,
    ToolResult,
    ToolResultEvent,
)
from agentm.extensions import ExtensionManifest

_ERROR_FLOOR = 256

class ToolResultBudgetConfig(BaseModel):
    model_config = {"extra": "allow"}

    max_chars: int = 50_000
    error_floor: int = _ERROR_FLOOR

MANIFEST = ExtensionManifest(
    name="tool_result_budget",
    description=(
        "Middle-out truncation of oversized text tool results; "
        f"error tool results retain at least {_ERROR_FLOOR} chars so block "
        "reasons survive."
    ),
    registers=("event:tool_result",),
    config_schema=ToolResultBudgetConfig,
    requires=(),  # Leaf atom: post-processes tool results only.
)


def _truncate_middle(text: str, budget: int) -> str:
    """Keep the first *half* and last *half* of *budget*, eliding the middle."""
    if len(text) <= budget:
        return text
    removed = len(text) - budget
    half = budget // 2
    tail_start = len(text) - (budget - half)
    return (
        text[:half]
        + f"\n\n…[{removed} chars truncated]…\n\n"
        + text[tail_start:]
    )


def install(api: ExtensionAPI, config: ToolResultBudgetConfig) -> None:
    max_chars = max(0, config.max_chars)
    error_floor = max(0, config.error_floor)

    def _on_tool_result(event: ToolResultEvent) -> ToolResult | None:
        text_total = sum(
            len(block.text)
            for block in event.result.content
            if isinstance(block, TextContent)
        )
        if text_total <= max_chars:
            return None

        effective_max = max_chars
        if event.result.is_error:
            effective_max = max(max_chars, min(error_floor, text_total))
            if text_total <= effective_max:
                return None

        remaining = effective_max
        new_content: list[TextContent | ImageContent] = []
        for block in event.result.content:
            if isinstance(block, ImageContent):
                new_content.append(block)
                continue
            if remaining <= 0:
                continue
            kept = _truncate_middle(block.text, remaining)
            new_content.append(TextContent(type="text", text=kept))
            remaining -= min(len(block.text), remaining)

        truncated_chars = text_total - effective_max
        total_lines = sum(
            block.text.count("\n") + 1
            for block in event.result.content
            if isinstance(block, TextContent)
        )
        new_content.append(
            TextContent(
                type="text",
                text=(
                    f"\n[tool_result_budget: truncated {truncated_chars} chars; "
                    f"original {text_total} chars / {total_lines} lines "
                    f"from {event.tool_name}]"
                ),
            )
        )
        return ToolResult(
            content=new_content,
            is_error=event.result.is_error,
            extras=event.result.extras,
        )

    api.on(ToolResultEvent.CHANNEL, _on_tool_result)
