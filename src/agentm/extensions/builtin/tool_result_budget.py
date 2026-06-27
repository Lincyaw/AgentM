"""Token-limit atom for the legacy ``tool_result_budget`` row.

Truncates oversized text payloads on ``tool_result`` while preserving any
image content blocks untouched.  Uses **middle-out** truncation: the first
half and last half of the token limit are preserved so the model sees both
the initial context (headers, imports) and the trailing state (errors,
final output).  Tool results flagged ``is_error=True`` are guaranteed at
least ``error_floor_tokens`` tokens of payload so block-reason messages
survive aggressive ``max_tokens`` limits.
"""

from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    ExtensionAPI,
    ImageContent,
    TextContent,
    ToolResult,
    ToolResultEvent,
)
from agentm.core.lib import count_text_tokens, truncate_text_tokens_middle
from agentm.extensions import ExtensionManifest


class ToolResultBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(gt=0)
    error_floor_tokens: int = Field(ge=0)

MANIFEST = ExtensionManifest(
    name="tool_result_budget",
    description=(
        "Middle-out token truncation of oversized text tool results; "
        "error tool results retain the configured token floor."
    ),
    registers=("event:tool_result",),
    config_schema=ToolResultBudgetConfig,
    requires=(),  # Leaf atom: post-processes tool results only.
)


def install(api: ExtensionAPI, config: ToolResultBudgetConfig) -> None:
    max_tokens = config.max_tokens
    error_floor_tokens = config.error_floor_tokens

    def _on_tool_result(event: ToolResultEvent) -> ToolResult | None:
        model_name = api.model.id if api.model is not None else None
        text_total_tokens = sum(
            count_text_tokens(block.text, model=model_name)
            for block in event.result.content
            if isinstance(block, TextContent)
        )
        if text_total_tokens <= max_tokens:
            return None

        effective_max_tokens = max_tokens
        if event.result.is_error:
            effective_max_tokens = max(
                max_tokens, min(error_floor_tokens, text_total_tokens)
            )
            if text_total_tokens <= effective_max_tokens:
                return None

        remaining_tokens = effective_max_tokens
        kept_tokens = 0
        new_content: list[TextContent | ImageContent] = []
        for block in event.result.content:
            if isinstance(block, ImageContent):
                new_content.append(block)
                continue
            if remaining_tokens <= 0:
                continue
            truncated = truncate_text_tokens_middle(
                block.text,
                remaining_tokens,
                model=model_name,
            )
            new_content.append(TextContent(type="text", text=truncated.text))
            kept_tokens += truncated.kept_tokens
            remaining_tokens -= truncated.kept_tokens

        truncated_tokens = text_total_tokens - kept_tokens
        total_lines = sum(
            block.text.count("\n") + 1
            for block in event.result.content
            if isinstance(block, TextContent)
        )
        new_content.append(
            TextContent(
                type="text",
                text=(
                    f"\n[tool_result_budget: truncated {truncated_tokens} tokens; "
                    f"original {text_total_tokens} tokens / {total_lines} lines "
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
