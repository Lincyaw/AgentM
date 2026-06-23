"""single-file atom: ``submit_rewrite`` terminal tool.

Used by the Stage B GT-blind rewriter child of the distill labeler.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel


class DistillSubmitRewriterConfig(BaseModel):
    model_config = {"extra": "allow"}


MANIFEST = ExtensionManifest(
    name="distill_submit_rewriter",
    description=(
        "Register submit_rewrite — terminal tool for the Stage B GT-blind "
        "rewriter child of the distill labeler."
    ),
    registers=("tool:submit_rewrite",),
    config_schema=DistillSubmitRewriterConfig,
    api_version=1,
    tier=1,
)

SUBMIT_REWRITE_TOOL_NAME = "submit_rewrite"

SUBMIT_REWRITE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "justifiable_from_index": {
            "type": "boolean",
            "description": (
                "True iff the upstream selection can be defended using "
                "ONLY context-index-visible information. False = drop the sample."
            ),
        },
        "reminder_text": {
            "type": "string",
            "description": (
                "Short methodological reminder (≤ 40 words). Empty when "
                "justifiable_from_index=false."
            ),
        },
        "drop_reason": {
            "type": "string",
            "description": (
                "When justifiable_from_index=false, a one-sentence reason. Empty otherwise."
            ),
        },
        "matched_event_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "Event ids the rewriter actually leans on. Usually a subset "
                "of the upstream selection."
            ),
        },
    },
    "required": [
        "justifiable_from_index",
        "reminder_text",
        "drop_reason",
        "matched_event_ids",
    ],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: DistillSubmitRewriterConfig) -> None:

    async def _submit(args: dict[str, Any]) -> ToolTerminate:
        del args
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="rewrite submitted")]),
            reason="llmharness.distill:submit_rewrite",
        )

    api.register_tool(
        FunctionTool(
            name=SUBMIT_REWRITE_TOOL_NAME,
            description=(
                "Submit the rewrite for this firing. Call exactly once "
                "as your final action — the child loop ends on return."
            ),
            parameters=SUBMIT_REWRITE_PARAMETERS,
            fn=_submit,
        )
    )


__all__ = ["MANIFEST", "SUBMIT_REWRITE_TOOL_NAME", "install"]
