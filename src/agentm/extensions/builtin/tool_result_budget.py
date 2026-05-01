"""Policy-gate atom for the §7 ``extensions.builtin.tool_result_budget`` row.

Truncates oversized text payloads on ``tool_result`` while preserving any
image content blocks untouched. Tool results flagged ``is_error=True`` are
guaranteed at least ``_ERROR_FLOOR`` characters of payload so block-reason
messages (e.g. permission/dedup denials) survive aggressive ``max_chars``
budgets and the model can still see why the call was rejected.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import ImageContent, TextContent, ToolResult, ToolResultEvent
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


# Minimum text payload preserved for ``is_error=True`` tool results, so that
# the kernel-synthesized "Tool call blocked: <reason>" strings are not chopped
# below readability by aggressive ``max_chars`` settings.
_ERROR_FLOOR = 256


MANIFEST = ExtensionManifest(
    name="tool_result_budget",
    description=(
        "Truncate oversized text tool results while preserving images; "
        f"error tool results retain at least {_ERROR_FLOOR} chars so block "
        "reasons survive."
    ),
    registers=("event:tool_result",),
    config_schema={
        "type": "object",
        "properties": {
            "max_chars": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": True,
    },
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    max_chars = max(0, int(config.get("max_chars", 50_000)))

    def _on_tool_result(event: ToolResultEvent) -> ToolResult | None:
        text_total = sum(
            len(block.text)
            for block in event.result.content
            if isinstance(block, TextContent)
        )
        if text_total <= max_chars:
            return None

        # Error tool results carry block reasons that must remain legible; lift
        # the budget to ``_ERROR_FLOOR`` (or the full payload, whichever is
        # smaller) when the kernel marked this result as an error.
        effective_max = max_chars
        if event.result.is_error:
            effective_max = max(max_chars, min(_ERROR_FLOOR, text_total))
            if text_total <= effective_max:
                return None

        remaining = effective_max
        new_content: list[TextContent | ImageContent] = []
        truncated_chars = text_total - effective_max
        for block in event.result.content:
            if isinstance(block, ImageContent):
                new_content.append(block)
                continue
            if remaining <= 0:
                continue
            kept_text = block.text[:remaining]
            if kept_text:
                new_content.append(TextContent(type="text", text=kept_text))
                remaining -= len(kept_text)

        new_content.append(
            TextContent(
                type="text",
                text=(
                    "\n\n[tool_result_budget truncated "
                    f"{truncated_chars} chars from {event.tool_name}]"
                ),
            )
        )
        return ToolResult(
            content=new_content,
            is_error=event.result.is_error,
            details=event.result.details,
        )

    api.on("tool_result", _on_tool_result)
