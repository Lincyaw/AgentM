"""``submit_plan`` extractor tool — Pydantic-backed.

The extractor child commits a basic-block partition of the new-turn
window before emitting events. The plan is informational — the
structural invariant (every internal event is a true branch point) is
enforced on the emitted events when the child calls
``finalize_extraction``. Carrying the plan as its own tool keeps
generation-order discipline: the model commits to a partition before
token-generating events.

This module is **not** an atom — the merged :mod:`atom` imports
:func:`build_submit_plan_tool` to mint a stateful :class:`FunctionTool`
over the per-firing :class:`ExtractionState` at install time.
"""

from __future__ import annotations

from typing import Any, Literal

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from pydantic import BaseModel, ConfigDict, Field

from .._tool_decorator import harness_tool
from .._witness_errors import format_witness_error
from ._state_echo import state_echo
from .state import ExtractionState

SUBMIT_PLAN_TOOL_NAME = "submit_plan"


class BlockPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turns: list[int] = Field(
        description=(
            "Contiguous trajectory indices that form this block. A "
            "linear block (one target, no branching) typically spans "
            "several turns; a branch-point block (hyp/dec/concl) is "
            "usually one turn. Every turn in the new-turn window "
            "appears in at least one block. Blocks are normally "
            "disjoint, with one specific exception: a choice-point "
            "branch block (a hyp formed by the agent's targeting "
            "choice at a tool_call turn) may share its single turn "
            "with the first turn of the linear block that executes "
            "the choice — see the passthrough worked example in the "
            "prompt's 'The block plan' section."
        ),
    )
    kind: Literal["linear", "branch"] = Field(
        description=(
            "'linear': the agent is probing one target without "
            "branching — emit ONE act + ONE evid covering all of "
            "the block's turns. 'branch': a reasoning move "
            "(hyp/dec/concl) — emit ONE atomic event."
        ),
    )
    note: str = Field(
        description=(
            "One short sentence naming the target/intent of the "
            "block. Linear: which service / file / data class is "
            "being probed. Branch: which kind of move (hyp/dec/"
            "concl) and what it's about."
        ),
    )


class SubmitPlanArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_plan: list[BlockPlanEntry] = Field(
        description=(
            "Partition the new-turn window into contiguous basic "
            "blocks (see Principle 3 of the extractor prompt). Each "
            "turn in the window appears in at least one block. The "
            "plan is informational — the auditable invariant (every "
            "internal event must be a true branch point, not a "
            "passthrough) is enforced on the emitted events when "
            "you call finalize_extraction. Plan-structure enforcement "
            "(e.g. no two adjacent linear blocks) is intentionally OFF "
            "in v19; concentrate the discipline on the events you "
            "actually emit."
        ),
    )


def build_submit_plan_tool(state: ExtractionState) -> FunctionTool:
    """Mint a :class:`FunctionTool` closing over ``state``."""

    @harness_tool(SUBMIT_PLAN_TOOL_NAME)
    async def _submit_plan(args: SubmitPlanArgs, _ctx: Any) -> ToolResult:
        """Commit the firing's basic-block plan. Call this ONCE before any node/edge edits. The plan partitions the new-turn window into contiguous blocks (linear vs branch); it's informational, not enforced as a structural constraint. The auditable invariant lives on the emitted events: every internal event must be a true branch point (in-degree > 1 OR out-degree > 1) — checked when you call finalize_extraction."""
        plan_payload = [entry.model_dump() for entry in args.block_plan]
        err = state.commit_plan(plan_payload)
        if err is not None:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=err,
                            attempt=f"submit_plan with {len(plan_payload)} block(s)",
                            state_echo=state_echo(state),
                            options=[
                                "re-call submit_plan with a corrected block_plan "
                                "(every entry needs non-empty turns, kind in "
                                "{'linear','branch'}, and a non-empty note)",
                            ],
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f'{{"ok": true, "blocks": {len(state.block_plan)}}}',
                )
            ],
        )

    return _submit_plan


__all__ = [
    "SUBMIT_PLAN_TOOL_NAME",
    "BlockPlanEntry",
    "SubmitPlanArgs",
    "build_submit_plan_tool",
]
