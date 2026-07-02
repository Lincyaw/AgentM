"""Termination protocol for RCA sub-agent workers.

Companion to :mod:`agentm_rca.finalize` (which terminates the
orchestrator). Workers in this scenario kept burning their full budget on
``query_sql`` calls without ever emitting a final assistant text turn —
``wait_subagent`` then surfaced ``final_text: null`` to the orchestrator,
which dispatched another worker. The fundamental issue: with only
investigation tools available, the model never decides "I'm done; time to
write a summary."

This extension installs two things:

* ``return_response`` tool — the worker's analogue of the orchestrator's
  ``submit_final_report``. Accepts a freeform ``text`` field and returns
  :class:`ToolTerminate` so the child loop exits cleanly. The text is
  JSON-encoded into the tool result for ``_extract_response`` to recover.
* ``decide_turn_action`` handler — a budget-aware injector. Empirically,
  open-ended adversarial personas (e.g., the current ``critic``) never call
  ``return_response`` on their own and just keep querying until
  force-stopped. When the worker has consumed ``warn_threshold`` of its
  turn budget without submitting, this handler injects a strong reminder;
  at ``force_threshold`` the reminder becomes a hard demand.

Scenarios that need a structured worker submission can override the tool
by registering one with the same name and a richer schema *after* this
extension installs (later registration wins).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel

from agentm.core.abi import (
    DecideTurnActionEvent,
    ExtensionAPI,
    Inject,
    LoopAction,
    TextContent,
    UserMessage,
)
from agentm.core.abi import (
    FunctionTool,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest

_DEFAULT_WARN_THRESHOLD = 0.6
_DEFAULT_FORCE_THRESHOLD = 0.85

class WorkerFinalizeConfig(BaseModel):
    warn_threshold: float = _DEFAULT_WARN_THRESHOLD
    force_threshold: float = _DEFAULT_FORCE_THRESHOLD

MANIFEST = ExtensionManifest(
    name="worker_finalize",
    description=(
        "Termination protocol for sub-agent workers: register the "
        "return_response tool so the worker can exit cleanly with its "
        "structured findings. Without it, workers loop on investigation "
        "tools until their budget runs out and emit no summary."
    ),
    registers=("tool:return_response",),
    config_schema=WorkerFinalizeConfig,
)

@dataclass(slots=True)
class _State:
    submitted: bool = False
    warned: bool = False
    forced: bool = False

def install(api: ExtensionAPI, config: WorkerFinalizeConfig) -> None:
    state = _State()
    warn_threshold = config.warn_threshold
    force_threshold = config.force_threshold

    async def _submit(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": (
                                    "return_response.text must be a "
                                    "non-empty string"
                                )
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
                        text=json.dumps(
                            {"status": "submitted", "text": text},
                            ensure_ascii=False,
                        ),
                    )
                ]
            ),
            reason="subagent:response-submitted",
        )

    def _on_decide_turn_action(event: DecideTurnActionEvent) -> LoopAction | None:
        if state.submitted:
            return None
        max_turns = api.session.get_loop_config().max_turns
        if max_turns is None or max_turns <= 0:
            return None
        # ``turn_index`` in TurnObservation is the turn that just
        # finished, so the next turn the model gets is ``turn_index+1``.
        # Compare against fractions of ``max_turns`` to gate the inject.
        consumed = event.observation.turn_index + 1
        ratio = consumed / max_turns
        if ratio < warn_threshold:
            return None
        remaining = max_turns - consumed
        if ratio >= force_threshold and not state.forced:
            state.forced = True
            text = (
                f"BUDGET CRITICAL: {consumed}/{max_turns} turns consumed "
                f"({remaining} remaining). You MUST call "
                "`return_response(text=...)` on your very next turn with "
                "whatever findings you have so far. Investigation is over. "
                "If you keep running other tools instead, the harness will "
                "force-stop you with no chance to summarize and your work "
                "will be discarded."
            )
        elif not state.warned:
            state.warned = True
            text = (
                f"BUDGET WARNING: {consumed}/{max_turns} turns consumed "
                f"({remaining} remaining). Wrap up your investigation NOW "
                "and call `return_response(text=...)` with your structured "
                "findings before you exhaust the turn budget. Anything you "
                "haven't summarized when the budget hits zero is lost."
            )
        else:
            return None
        return Inject(
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(type="text", text=text)],
                    timestamp=time.time(),
                ),
            ]
        )

    api.on("decide_turn_action", _on_decide_turn_action)

    api.register_tool(
        FunctionTool(
            name="return_response",
            description=(
                "Submit your final response to the dispatcher and end the "
                "task. This is the only sanctioned way to terminate. The "
                "``text`` you pass is delivered verbatim to the orchestrator "
                "as your structured findings — write the full report, not a "
                "one-line confirmation. After this call, the worker session "
                "exits; you cannot run more tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Your full structured findings — the report "
                            "your persona prompt asked for. Plain text, "
                            "no JSON wrapping needed."
                        ),
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            fn=_submit,
        )
    )

__all__: Final = ["MANIFEST", "install"]
