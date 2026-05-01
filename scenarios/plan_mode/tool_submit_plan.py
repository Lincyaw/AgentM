"""Tool atom for the ``extensions.builtin.tool_submit_plan`` §7.1 row."""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.events import PlanSubmittedEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_submit_plan",
    description="Register the submit_plan tool for plan mode.",
    registers=("tool:submit_plan", "event:plan_submitted"),
    config_schema=None,
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
    },
    "required": ["plan"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config

    async def _execute(args: dict[str, Any]) -> ToolResult:
        plan = str(args["plan"])
        try:
            plan_id = api.session.append_entry("plan", {"text": plan})
            await api.events.emit(
                "plan_submitted",
                PlanSubmittedEvent(plan_id=plan_id, plan_text=plan),
            )
            return ToolResult(
                content=[TextContent(type="text", text="plan submitted")],
                extras={"plan_submitted": True, "plan_id": plan_id},
            )
        except Exception as exc:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Failed to submit plan: {exc}",
                    )
                ],
                is_error=True,
            )

    api.register_tool(
        FunctionTool(
            name="submit_plan",
            description="Persist and emit a finished execution plan.",
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )
