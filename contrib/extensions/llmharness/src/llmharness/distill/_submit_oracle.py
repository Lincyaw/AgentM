"""single-file atom: ``submit_oracle_label`` terminal tool.

Used by the Stage A oracle child of the distill labeler. The child sees
the GT-laden prompt (prompts/oracle.md), reasons over the snapshot, and
terminates by calling this tool exactly once.

Single tool, schema enforced by the kernel; the labeler reads the
structured payload directly off the assistant message.
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


class DistillSubmitOracleConfig(BaseModel):
    model_config = {"extra": "allow"}


MANIFEST = ExtensionManifest(
    name="distill_submit_oracle",
    description=(
        "Register submit_oracle_label — terminal tool for the Stage A "
        "GT-aware oracle child of the distill labeler."
    ),
    registers=("tool:submit_oracle_label",),
    config_schema=DistillSubmitOracleConfig,
    api_version=1,
    tier=1,
)

SUBMIT_ORACLE_TOOL_NAME = "submit_oracle_label"

SUBMIT_ORACLE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_finding_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "description": (
                "Indices into the `findings` array of the prompt payload "
                "that should be surfaced to the main agent. Empty = stay silent."
            ),
        },
        "matched_event_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Audit event ids that support the selection.",
        },
        "rationale_with_gt": {
            "type": "string",
            "description": (
                "Free-text reasoning. May reference GT — this field is "
                "dropped before the student model sees the data."
            ),
        },
        "continuation_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short notes for the next firing's auditor.",
        },
    },
    "required": [
        "selected_finding_indices",
        "matched_event_ids",
        "rationale_with_gt",
        "continuation_notes",
    ],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: DistillSubmitOracleConfig) -> None:

    async def _submit(args: dict[str, Any]) -> ToolTerminate:
        del args
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="oracle label submitted")]),
            reason="llmharness.distill:submit_oracle_label",
        )

    api.register_tool(
        FunctionTool(
            name=SUBMIT_ORACLE_TOOL_NAME,
            description=(
                "Submit the oracle label for this firing. Call exactly once "
                "as your final action — the child loop ends on return."
            ),
            parameters=SUBMIT_ORACLE_PARAMETERS,
            fn=_submit,
        )
    )


__all__ = ["MANIFEST", "SUBMIT_ORACLE_TOOL_NAME", "install"]
