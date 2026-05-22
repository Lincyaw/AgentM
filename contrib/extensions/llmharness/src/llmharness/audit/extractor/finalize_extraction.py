"""``finalize_extraction`` terminator tool — Pydantic-backed.

The extractor child terminates by calling ``finalize_extraction()`` with
no payload. The handler runs :meth:`ExtractionState.finalize`, which:

* drains pending events / edges into the public ``events`` / ``edges``
  tuples;
* runs the cross-graph degree check — rejects firings whose internal
  events sit on a (in_deg=1, out_deg=1) passthrough chain.

On finalize failure the firing stays alive: the model may issue more
``upsert_*`` / ``delete_*`` edits to promote passthrough nodes to true
branch points, then re-call ``finalize_extraction``. The error message
funnels through :func:`format_witness_error` and names the concrete
in/out neighbours of every passthrough event so the LLM gets a
self-correcting directive rather than the v18 generic "merge or
promote" wording.

This module is **not** an atom — the merged :mod:`atom` imports
:func:`build_finalize_extraction_tool`.
"""

from __future__ import annotations

import re
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from pydantic import BaseModel, ConfigDict

from .._tool_decorator import harness_tool
from .._witness_errors import format_witness_error
from ._state_echo import chain_neighbours, state_echo
from .state import ExtractionState

FINALIZE_EXTRACTION_TOOL_NAME = "finalize_extraction"
FINALIZE_EXTRACTION_REASON = "llmharness:finalize_extraction"


class FinalizeExtractionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


_PASSTHROUGH_LINE = re.compile(
    r"event\[(\d+)\] kind=([^\s']+) '([^']*)': in=1, out=1"
)


def _passthrough_options(state: ExtractionState, raw_error: str) -> list[str]:
    """Build concrete tool-call options naming the passthrough's neighbours.

    Parses the validator's diagnostic for the passthrough ids, then
    looks up each event's in/out neighbour in the pending edge list and
    emits a recipe targeting THAT pair. When the validator's text shape
    drifts we fall back to a single generic option so the helper still
    has something actionable to render.
    """
    options: list[str] = []
    for match in _PASSTHROUGH_LINE.finditer(raw_error):
        passthrough_id = int(match.group(1))
        in_id, out_id = chain_neighbours(state, passthrough_id)
        if in_id is None or out_id is None:
            continue
        # next id to use when merging act+evid into a single node
        existing_ids = sorted(ev.id for ev in state._events_pending)
        next_free = (existing_ids[-1] + 1) if existing_ids else 1
        options.append(
            f"delete_node({passthrough_id}) then upsert_node(id={next_free}, "
            f"kind=..., source_turns=[...]) — merge the in-neighbour "
            f"({in_id}) and the passthrough into one basic block"
        )
        options.append(
            f"upsert_edge(src={next_free}, dst={passthrough_id}, kind=..., "
            f"reason=..., cited_quote=... or cited_entities=[...]) "
            f"— emit a NEW event that refs back to {passthrough_id}, "
            f"giving it out-degree 2"
        )
    options.append("reset_extraction() — last resort; drops pending state")
    # Cap at four so the prompt stays scannable.
    return options[:4]


def build_finalize_extraction_tool(state: ExtractionState) -> FunctionTool:
    """Mint a :class:`FunctionTool` closing over ``state``."""

    @harness_tool(FINALIZE_EXTRACTION_TOOL_NAME, terminates=True)
    async def _finalize(
        _args: FinalizeExtractionArgs, _ctx: Any
    ) -> ToolTerminate | ToolResult:
        """Terminate the extractor firing. Call this AFTER you have emitted every node/edge with upsert_node / upsert_edge (and any merges via delete_node / delete_edge). The handler runs the cross-graph degree check — passthrough events (in=1, out=1) are rejected and the firing stays alive so you can append more edits and re-call. On success the loop terminates."""
        err = state.finalize()
        if err is None:
            digest = (
                f'{{"ok": true, "events": {len(state.events)}, '
                f'"edges": {len(state.edges)}, '
                f'"dropped": {len(state.dropped_edges)}}}'
            )
            return ToolTerminate(
                result=ToolResult(content=[TextContent(type="text", text=digest)]),
                reason=FINALIZE_EXTRACTION_REASON,
            )
        # Degree-check rejection — name the offending neighbours.
        symptom = err.split("\n", 1)[0]
        options = _passthrough_options(state, err)
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=format_witness_error(
                        symptom=symptom,
                        attempt="finalize_extraction()",
                        state_echo=state_echo(state),
                        options=options,
                    ),
                )
            ],
            is_error=True,
        )

    return _finalize


__all__ = [
    "FINALIZE_EXTRACTION_REASON",
    "FINALIZE_EXTRACTION_TOOL_NAME",
    "FinalizeExtractionArgs",
    "build_finalize_extraction_tool",
]
