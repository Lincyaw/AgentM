"""Auditor tool surface: submit_verdict (terminal)."""

from __future__ import annotations

from typing import Any, Self

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SUBMIT_VERDICT_TOOL_NAME = "submit_verdict"

class _VerdictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_reminder: bool = Field(
        description="True to surface a reminder to the main agent. False for a silent verdict.",
    )
    reminder_text: str = Field(
        description="Advisory for the main agent. Non-empty when surface_reminder=true.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Verifiable support for the reminder, one item per fact. Each item "
            "names its source and quotes what it shows, e.g. "
            "'turn 42 bash result: 3 tests FAILED (copyout)' or "
            "'read kernel/memlayout.h: USYSCALL already defined at line 61'. "
            "Required (non-empty) when surface_reminder=true; every claim in "
            "reminder_text must be covered by an item here."
        ),
    )
    continuation_notes: list[str] = Field(
        description="Notes for your next firing. Auditor-internal.",
    )
    matched_event_ids: list[int] = Field(
        description=(
            "Event IDs where an error originates or is committed to. "
            "Only include events that introduced, propagated, or finalized "
            "an unsupported or incorrect claim. Do NOT include events you "
            "merely reviewed but found no fault in."
        ),
    )

    @model_validator(mode="after")
    def _check_reminder(self) -> Self:
        if self.surface_reminder and not self.reminder_text.strip():
            raise ValueError("reminder_text must be non-empty when surface_reminder=true")
        if self.surface_reminder and not any(e.strip() for e in self.evidence):
            raise ValueError(
                "evidence must contain at least one non-empty item when "
                "surface_reminder=true — cite the turn or file that supports "
                "the reminder, or submit a silent verdict instead"
            )
        return self

class SubmitVerdictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: _VerdictModel

async def _submit_handler(args: dict[str, Any]) -> ToolTerminate | ToolResult:
    try:
        SubmitVerdictArgs.model_validate(args)
    except ValidationError as exc:
        return ToolResult(
            content=[TextContent(type="text", text=f"submit_verdict rejected: {exc}")],
            is_error=True,
        )
    return ToolTerminate(
        result=ToolResult(content=[TextContent(type="text", text="verdict submitted")]),
        reason="llmharness:submit_verdict",
    )

SUBMIT_VERDICT_TOOL = FunctionTool(
    name=SUBMIT_VERDICT_TOOL_NAME,
    description="Submit the cognitive-audit verdict. Call exactly ONCE per firing as your final action.",
    parameters=SubmitVerdictArgs,
    fn=_submit_handler,
    metadata={"terminates": True},
)

MANIFEST = ExtensionManifest(
    name="auditor_tools",
    description="Register the auditor submit_verdict tool.",
    registers=("tool:submit_verdict",),
)

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    api.register_tool(SUBMIT_VERDICT_TOOL)
