"""Termination protocol for the RCA orchestrator.

Complements the sub-agent lifecycle floor shipped in #57: that floor injects
unread child findings on ``decide_turn_action`` so background work isn't lost,
but it does not stop the orchestrator from declaring the investigation over
once every child is read. We still observed the model digest a long scout
report, write a prose summary, and emit ``end_turn`` without producing the
final RCA deliverable.

This extension layers a stricter policy on top:

  * Registers ``submit_final_report``: the only sanctioned termination tool.
    A successful call returns :class:`ToolTerminate` so the loop's default
    action becomes ``Stop(ToolTerminated(...))`` and the orchestrator exits
    cleanly. The report payload reaches downstream observers via the
    ``tool_call`` / ``tool_result`` channels, so we deliberately do not stash
    a duplicate copy here.
  * Subscribes to ``decide_turn_action``: while ``submitted`` is still false
    AND the kernel default is a voluntary :class:`ModelEndTurn`, returns
    :class:`Inject` carrying a continuation user message that pushes the
    model back into investigation. Coexists cleanly with sub_agent's own
    handler (multiple ``Inject`` returns concatenate in registration order).

Safety net: the loop's ``max_turns`` cap still applies. ``MaxTurnsExhausted``
has ``final=True`` so any ``Inject`` we return is ignored — a model that
refuses to call the tool eventually exits with that cause rather than
spinning forever.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import (
    DecideTurnActionEvent,
    Inject,
    LoopAction,
    ModelEndTurn,
    Stop,
)
from agentm.core.abi.messages import TextContent, UserMessage
from agentm.core.abi.tool import (
    FunctionTool,
    ToolResult,
    ToolTerminate,
)
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

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        root_cause = _require_str(args.get("root_cause"))
        triggering_signal = _require_str(args.get("triggering_signal"))
        evidence = _require_str(args.get("evidence"))
        remediation = _require_str(args.get("remediation"))
        causal_graph = args.get("causal_graph")
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
        cg_error = _validate_causal_graph(causal_graph)
        if cg_error is not None:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps({"error": cg_error}),
                    )
                ],
                is_error=True,
            )
        if missing:
            # Validation failure: stay in the loop so the model can retry.
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
        return ToolTerminate(
            result=ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "Final report accepted. The investigation is "
                            "complete."
                        ),
                    )
                ]
            ),
            reason="rca:final-report-submitted",
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
                    "causal_graph": {
                        "type": "object",
                        "description": (
                            "Machine-readable RCA conclusion as a CausalGraph. "
                            "``root_causes`` is the only field downstream "
                            "evaluation cares about; populate it with one "
                            "node per implicated service. ``component`` is "
                            "the service name (e.g. 'ts-payment-service'). "
                            "``nodes`` and ``edges`` are optional and may be "
                            "empty arrays when no propagation graph is built."
                        ),
                        "properties": {
                            "nodes": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": (
                                    "All nodes considered in the analysis "
                                    "(may be empty)."
                                ),
                            },
                            "edges": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": (
                                    "Causal edges source -> target (may be "
                                    "empty)."
                                ),
                            },
                            "root_causes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component": {"type": "string"},
                                    },
                                    "required": ["component"],
                                },
                                "description": (
                                    "Nodes flagged as root cause(s). At "
                                    "least one entry."
                                ),
                            },
                        },
                        "required": ["root_causes"],
                    },
                },
                "required": [
                    "root_cause",
                    "triggering_signal",
                    "evidence",
                    "remediation",
                    "causal_graph",
                ],
                "additionalProperties": False,
            },
            fn=_submit,
        )
    )

    def _on_decide_turn_action(event: DecideTurnActionEvent) -> LoopAction | None:
        if state.submitted:
            return None
        default = event.observation.default_action
        # Only fight a voluntary ``ModelEndTurn`` exit. ``ToolTerminated``
        # only fires when ``submit_final_report`` itself runs (we already
        # filtered above), so any other terminal tool is policy-violating
        # but final causes (max_turns / signal / budget) are not ours to
        # override anyway.
        if not isinstance(default, Stop) or not isinstance(
            default.cause, ModelEndTurn
        ):
            return None
        return Inject(
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(type="text", text=instruction)],
                    timestamp=time.time(),
                ),
            ]
        )

    api.on("decide_turn_action", _on_decide_turn_action)


def _require_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _validate_causal_graph(value: Any) -> str | None:
    """Lightweight shape check for the ``causal_graph`` argument.

    Mirrors the JSON schema's ``required`` constraints so a bad call gets a
    structured error (and a retry) instead of crashing the loop. Returns an
    error message string on failure, ``None`` on success.
    """
    if not isinstance(value, dict):
        return "causal_graph must be an object"
    root_causes = value.get("root_causes")
    if not isinstance(root_causes, list) or not root_causes:
        return "causal_graph.root_causes must be a non-empty list"
    for idx, node in enumerate(root_causes):
        if not isinstance(node, dict):
            return f"causal_graph.root_causes[{idx}] must be an object"
        component = node.get("component")
        if not isinstance(component, str) or not component.strip():
            return (
                f"causal_graph.root_causes[{idx}].component "
                "must be a non-empty string"
            )
    return None


__all__ = ["MANIFEST", "install"]
