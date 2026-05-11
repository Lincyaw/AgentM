"""Termination protocol for the RCA orchestrator.

The orchestrator submits its final answer via ``submit_final_report``. The
schema is the **official rcabench-platform agent contract**
(:class:`rcabench_platform.v3.sdk.evaluation.v2.AgentRCAOutput`):

* ``root_causes[]`` — each carries ``service`` (must match a string present
  in the data), ``fault_kind`` (one of the platform's enum values), and
  ``evidence[]`` (DuckDB SQL + natural-language claim).
* ``propagation[]`` — directed edges from upstream failing service toward
  user-facing tier, each with evidence.

We validate by handing the raw tool args to :class:`AgentRCAOutput` —
``service`` vocabulary alignment is enforced in the prompt (see
``rcabench_contract`` atom which splices in
``get_agent_contract_prompt()``); shape correctness is enforced by Pydantic
here. A failed validation returns ``is_error=True`` so the model can retry;
a successful validation returns :class:`ToolTerminate` so the loop exits
cleanly.

A ``decide_turn_action`` handler keeps the orchestrator from voluntarily
ending its turn before submitting: while ``submitted`` is still false AND
the kernel default is :class:`ModelEndTurn`, we inject a continuation
instruction that pushes the model back into investigation. The
``max_turns`` cap remains the ultimate safety net (its
:class:`MaxTurnsExhausted` is ``final=True`` so our ``Inject`` is ignored).
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
from agentm.core.abi import (
    FunctionTool,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

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
    # Imported lazily so ``import agentm_rca.tools.finalize`` does not
    # require rcabench-platform at module-load time (e.g. for static
    # analysis or tooling without the SDK installed).
    from rcabench_platform.v3.sdk.evaluation.v2 import AgentRCAOutput

    state = _State()
    instruction = str(
        config.get("continuation_instruction") or _DEFAULT_INSTRUCTION
    )

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        try:
            output = AgentRCAOutput.model_validate(args)
        except Exception as exc:  # pydantic ValidationError + anything weird
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "agent_contract_validation_failed",
                                "detail": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
                is_error=True,
            )
        if not output.root_causes:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "root_causes must be non-empty — "
                                "submit at least one RootCauseClaim with "
                                "service + fault_kind + >=1 evidence."
                            }
                        ),
                    )
                ],
                is_error=True,
            )
        state.submitted = True
        # Serialize via the model so the wire payload is always
        # contract-conformant (alias ``from`` survives, enums become their
        # string values). Eval adapter calls ``AgentRCAOutput.parse_str``
        # on this exact text.
        return ToolTerminate(
            result=ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=output.model_dump_json(by_alias=True),
                    )
                ]
            ),
            reason="rca:final-report-submitted",
        )

    api.register_tool(
        FunctionTool(
            name="submit_final_report",
            description=(
                "Submit the final root-cause analysis as the official "
                "rcabench-platform AgentRCAOutput. This is the only "
                "sanctioned way to terminate. Service names must match "
                "strings present in the data; fault_kind must be one of "
                "the enum values listed in the agent contract above."
            ),
            parameters=_AGENT_RCA_OUTPUT_SCHEMA,
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


# Inlined (no $defs / $ref) JSON schema mirroring AgentRCAOutput, so any
# LLM tool-call backend that doesn't resolve refs still works. Kept
# manually in sync with the upstream Pydantic models — Pydantic remains
# the source of truth at validation time, this is just the wire schema
# the model sees.
_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["metric", "trace", "log"],
        },
        "sql": {
            "type": "string",
            "description": (
                "DuckDB SQL the platform can re-execute against the case "
                "parquets to verify the claim."
            ),
        },
        "claim": {
            "type": "string",
            "description": (
                "<=20-word natural-language assertion the SQL rows back."
            ),
        },
    },
    "required": ["kind", "sql", "claim"],
}

_FAULT_KIND_ENUM: list[str] = [
    "pod_failure",
    "pod_unavailable",
    "network_delay",
    "network_loss",
    "network_partition",
    "network_corrupt",
    "network_duplicate",
    "network_bandwidth_limit",
    "http_aborted",
    "http_slow",
    "http_payload_modified",
    "http_response_status_modified",
    "cpu_stress",
    "jvm_thread_cpu_stress",
    "mem_stress",
    "jvm_heap_stress",
    "jvm_gc_pressure",
    "jvm_method_exception",
    "jvm_jdbc_exception",
    "jvm_method_latency",
    "jvm_jdbc_latency",
    "jvm_method_mutated",
    "dns_resolution_failed",
    "dns_resolution_wrong",
    "clock_skew",
    "unknown",
]

_AGENT_RCA_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root_causes": {
            "type": "array",
            "minItems": 1,
            "description": (
                "One entry per distinct root cause. Do NOT collapse "
                "multiple distinct faults into a single entry."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": (
                            "Canonical service_name as it appears in the "
                            "data — must match strings present in the "
                            "parquets / views; do not invent."
                        ),
                    },
                    "fault_kind": {
                        "type": "string",
                        "enum": _FAULT_KIND_ENUM,
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "items": _EVIDENCE_SCHEMA,
                    },
                },
                "required": ["service", "fault_kind", "evidence"],
            },
        },
        "propagation": {
            "type": "array",
            "description": (
                "Fault-impact chain edges — FROM the failing service "
                "TOWARD the user-visible alarm tier. Not the request-call "
                "direction. Synthetic generators (loadgenerator, locust, "
                "wrk2, dsb-wrk2, k6) are NOT services."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": _EVIDENCE_SCHEMA,
                    },
                },
                "required": ["from", "to"],
            },
        },
    },
    "required": ["root_causes"],
}


__all__ = ["MANIFEST", "install"]
