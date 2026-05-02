"""Termination protocol for the RCA orchestrator.

Complements the sub-agent lifecycle floor shipped in #57: that floor injects
unread child findings on ``before_agent_end`` so background work isn't lost,
but it does not stop the orchestrator from declaring the investigation over
once every child is read. We still observed the model digest a long scout
report, write a prose summary, and emit ``end_turn`` without producing the
final RCA deliverable.

This extension layers a stricter policy on top:

  * Registers ``submit_final_report``: the only sanctioned termination tool.
    Calling it flips an extension-local ``submitted`` flag; the report
    payload itself reaches downstream observers via the ``tool_call`` /
    ``tool_result`` channels, so we deliberately do not stash a duplicate
    copy here.
  * Subscribes to ``before_agent_end``: while ``submitted`` is still false,
    returns ``{"cancel": True, "append": [<continuation user message>]}``.
    Cancel is OR-ed across handlers and append lists are concatenated, so
    this co-exists cleanly with sub_agent's own ``before_agent_end`` handler.

Safety net: the loop's ``max_turns`` cap still applies. ``max_turns``
terminations bypass the cancel field entirely (PR #57 design), so a model
that refuses to call the tool eventually exits with ``stop_reason="max_turns"``
rather than spinning forever.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import BeforeAgentEndEvent
from agentm.core.abi.messages import TextContent, UserMessage
from agentm.core.abi.tool import FunctionTool, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="finalize",
    description=(
        "Termination protocol: the orchestrator must call submit_final_report "
        "to end the investigation. Otherwise the loop continues."
    ),
    registers=("tool:submit_final_report",),
)


_DEFAULT_INSTRUCTION = (
    "Your last assistant message had no tool_call. Prose-only turns are "
    "rejected. You MUST emit exactly one tool_call now.\n\n"
    "Choose one of two paths:\n"
    "  (A) If you believe you have a confirmed root cause backed by "
    "evidence — call `submit_final_report` now to end the investigation.\n"
    "  (B) Otherwise — call any other registered investigation tool to "
    "continue: dispatch a worker, run a SQL query, poll task status, "
    "update or remove a hypothesis, etc.\n\n"
    "Do not respond with prose alone. Do not say `Let me ...` without "
    "calling the tool you just named. The only way out of this loop is to "
    "call `submit_final_report` (path A) or run more investigation tools "
    "(path B)."
)


@dataclass
class _State:
    submitted: bool = False


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    state = _State()
    instruction = str(
        config.get("continuation_instruction") or _DEFAULT_INSTRUCTION
    )

    async def _submit(args: dict[str, Any]) -> ToolResult:
        root_cause = _require_str(args.get("root_cause"))
        triggering_signal = _require_str(args.get("triggering_signal"))
        evidence = _require_str(args.get("evidence"))
        remediation = _require_str(args.get("remediation"))
        missing = [
            name
            for name, value in (
                ("root_cause", root_cause),
                ("triggering_signal", triggering_signal),
                ("evidence", evidence),
                ("remediation", remediation),
            )
            if not value
        ]
        if missing:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "missing required fields",
                                "missing": missing,
                            }
                        ),
                    )
                ],
                is_error=True,
            )
        state.submitted = True
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Final report accepted. You may now end this turn; "
                        "the investigation is complete."
                    ),
                )
            ]
        )

    api.register_tool(
        FunctionTool(
            name="submit_final_report",
            description=(
                "Submit the final root-cause analysis report. This is the "
                "only way to terminate the investigation. Until you call "
                "this tool, ending your turn without a tool_call will be "
                "rejected and you will be asked to keep investigating."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "root_cause": {
                        "type": "string",
                        "description": (
                            "Service or component identified as the root "
                            "cause."
                        ),
                    },
                    "triggering_signal": {
                        "type": "string",
                        "description": (
                            "First metric, span, or log line that "
                            "deviated from baseline."
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "Citations of the SQL queries or worker "
                            "findings that support the conclusion."
                        ),
                    },
                    "remediation": {
                        "type": "string",
                        "description": "Suggested fix or mitigation.",
                    },
                },
                "required": [
                    "root_cause",
                    "triggering_signal",
                    "evidence",
                    "remediation",
                ],
                "additionalProperties": False,
            },
            fn=_submit,
        )
    )

    def _on_before_agent_end(event: BeforeAgentEndEvent) -> Any:
        if state.submitted:
            return None
        return {
            "cancel": True,
            "append": [
                UserMessage(
                    role="user",
                    content=[TextContent(type="text", text=instruction)],
                    timestamp=time.time(),
                ),
            ],
        }

    api.on("before_agent_end", _on_before_agent_end)


def _require_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


__all__ = ["MANIFEST", "install"]
