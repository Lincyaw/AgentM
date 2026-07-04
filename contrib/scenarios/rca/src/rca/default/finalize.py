"""Termination protocol for the RCA orchestrator.

The orchestrator submits its final answer via ``submit_final_report``. The
schema is the fpg ``ModelRCAOutput`` contract bound to this scenario's
``fpg_profile.toml``:

* ``nodes[]`` — anomalous service or link states, with ``subject`` such as
  ``svc:checkout`` or ``link:checkout->payment``, a profile-bound
  ``predicate``, interval, and evidence.
* ``edges[]`` — directed causal links from cause node id to effect node id.
* ``root_causes[]`` — explicit root-cause node ids, ordered by confidence.

We validate by handing raw tool args to the profile-bound Pydantic model.
A failed validation returns ``is_error=True`` so the model can retry; a
successful validation returns :class:`ToolTerminate` so the loop exits
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
from typing import Any, Final

from loguru import logger
from pydantic import ConfigDict

from agentm.core.abi import (
    DecideTurnActionEvent,
    ExtensionAPI,
    Inject,
    LoopAction,
    ModelEndTurn,
    Stop,
    TextContent,
    UserMessage,
)
from agentm.core.abi import (
    FunctionTool,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from rca.fpg_schema import (
    FpgOutputConfig,
    model_output_model,
    model_output_tool_schema,
    resolve_profile_path,
)


class FinalizeConfig(FpgOutputConfig):
    model_config = ConfigDict(extra="forbid")

    continuation_instruction: str | None = None

MANIFEST = ExtensionManifest(
    name="finalize",
    description=(
        "Termination protocol: the orchestrator must call submit_final_report "
        "to end the investigation. Otherwise the loop continues."
    ),
    registers=("tool:submit_final_report",),
    config_schema=FinalizeConfig,
)

_DEFAULT_INSTRUCTION = (
    "Your last assistant message had no tool_call. Prose-only turns are "
    "rejected. You MUST emit exactly one tool_call now.\n\n"
    "Choose one of two paths:\n"
    "  (A) If you believe you have a confirmed root cause backed by "
    "evidence — call `submit_final_report` now to end the investigation.\n"
    "  (B) Otherwise — call any other registered investigation tool to "
    "continue: dispatch a worker, run a SQL query, "
    "update or remove a hypothesis, etc.\n\n"
    "Do not respond with prose alone. Do not say `Let me ...` without "
    "calling the tool you just named. The only way out of this loop is to "
    "call `submit_final_report` (path A) or run more investigation tools "
    "(path B)."
)

@dataclass(slots=True)
class _State:
    submitted: bool = False

def install(api: ExtensionAPI, config: FinalizeConfig) -> None:
    profile_path = resolve_profile_path(config.profile_path, scenario_dir=api.scenario_dir)
    output_model = model_output_model(profile_path)
    tool_parameters = model_output_tool_schema(profile_path)

    state = _State()
    instruction = config.continuation_instruction or _DEFAULT_INSTRUCTION

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        if _is_empty_no_root_report(args):
            state.submitted = True
            return ToolTerminate(
                result=ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(args, ensure_ascii=False),
                        )
                    ]
                ),
                reason="rca:final-report-submitted",
            )
        try:
            output = output_model.model_validate(args)
        except Exception as exc:  # pydantic ValidationError + anything weird
            logger.warning("fpg model output validation failed: {}", exc)
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "fpg_model_output_validation_failed",
                                "detail": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
                is_error=True,
            )
        state.submitted = True
        # Serialize via the model so the wire payload is always
        # contract-conformant. Eval adapter parses this exact text with
        # fpg.ModelRCAOutput.
        return ToolTerminate(
            result=ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=output.model_dump_json(),
                    )
                ]
            ),
            reason="rca:final-report-submitted",
        )

    api.register_tool(
        FunctionTool(
            name="submit_final_report",
            description=(
                "Submit the final root-cause analysis as fpg ModelRCAOutput. "
                "This is the only sanctioned way to terminate. Subjects are "
                "profile entity refs such as svc:<service> or "
                "link:<source_service>-><target_service>; predicates must use the "
                "vocabulary listed in the agent contract; root_causes must "
                "reference node ids from nodes[]."
            ),
            parameters=tool_parameters,
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


def _is_empty_no_root_report(args: dict[str, Any]) -> bool:
    return (
        args.get("nodes") == []
        and args.get("edges") == []
        and args.get("root_causes") == []
    )

__all__: Final = ["MANIFEST", "install"]
