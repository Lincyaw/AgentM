"""``upsert_node`` extractor tool — Pydantic-backed.

Insert or replace one event node in the persistent folded graph (this
firing's pending ops + every prior firing). The id resolves against the
*current folded view*; refs are NOT carried on the node — edges are
their own thing (see :mod:`upsert_edge`).

This module is **not** an atom — the merged :mod:`atom` imports
:func:`build_upsert_node_tool` to mint a stateful :class:`FunctionTool`
over the per-firing :class:`ExtractionState`.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from pydantic import BaseModel, ConfigDict, Field

from .._tool_decorator import harness_tool
from .._witness_errors import format_witness_error
from ._state_echo import state_echo
from .state import ExtractionState

UPSERT_NODE_TOOL_NAME = "upsert_node"

# Use a Literal so the generated JSON schema carries the closed-set
# enum verbatim — pydantic_to_openai_tool_schema strips title keys for
# parity with the legacy hand-written ``enum: [...]`` constants.
_EventKindLiteral = Literal["task", "hyp", "evid", "act", "dec", "concl"]


class UpsertNodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        description=(
            "Node id. If this id already exists in the current graph "
            "(this firing or any prior firing), the node is updated "
            "in place. Otherwise the id must equal the next available "
            "id (max existing id + 1) or a node id you deleted earlier "
            "in this firing (re-use after delete is the merge-duplicate "
            "path). Gaps over still-live ids are rejected."
        ),
    )
    kind: _EventKindLiteral = Field(
        description=(
            "Closed-set event kind classified by ACTION SIGNATURE, "
            "not by what the agent says it is doing."
        ),
    )
    summary: str = Field(
        description=(
            "Natural-language paragraph describing this event, with "
            "LENGTH PROPORTIONAL TO source_turns COUNT. A "
            "single-turn branch event (task / hyp / dec / concl) is "
            "one focused sentence with the concrete claim. A linear "
            "act or evid that COALESCES N consecutive turns must be "
            "a paragraph that walks through what happened across "
            "those N turns: roughly one short sentence per covered "
            "turn. Name every distinct tool_call's concrete "
            "parameters verbatim (services, time windows, query "
            "filters, file paths, error codes, span/log/metric "
            "names) and quote the key numbers each result returned."
        ),
    )
    source_turns: list[int] = Field(
        description=(
            "Trajectory indices this event was extracted from. "
            "Non-empty and contiguous ([first, first+1, ..., last] "
            "with no gaps)."
        ),
    )


def _attempt_echo(args: UpsertNodeArgs) -> str:
    summary_preview = args.summary[:40] + ("..." if len(args.summary) > 40 else "")
    return (
        f"upsert_node(id={args.id}, kind={args.kind}, "
        f"source_turns={args.source_turns}, summary={summary_preview!r})"
    )


def build_upsert_node_tool(state: ExtractionState) -> FunctionTool:
    """Mint a :class:`FunctionTool` closing over ``state``."""

    @harness_tool(UPSERT_NODE_TOOL_NAME)
    async def _upsert_node(args: UpsertNodeArgs, _ctx: Any) -> ToolResult:
        """Insert or replace one event node in the current graph. Editing a prior-firing node is supported — the id resolves against the folded view (this firing + every prior firing). Recorded as a NodeUpsert op."""
        raw = args.model_dump()
        result = state.apply_node_upsert(raw)
        if isinstance(result, str):
            # The validator's message already names the offending field;
            # we route it through format_witness_error so the LLM sees the
            # standard three-section template every error funnels through.
            options = [
                "re-call upsert_node with id = max(existing ids) + 1 to append a fresh node",
                "re-call upsert_node with an id already present in the folded graph to edit in place",
                "delete_node(<id>) first if you intended to replace and re-use that id",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=_attempt_echo(args),
                            state_echo=state_echo(state),
                            options=options,
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        )

    return _upsert_node


__all__ = [
    "UPSERT_NODE_TOOL_NAME",
    "UpsertNodeArgs",
    "build_upsert_node_tool",
]
