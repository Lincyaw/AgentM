"""``finalize_extraction`` terminator tool — Pydantic-backed.

V4 (2026-05-24): finalize is now a one-shot commit on a
witness-valid, id-monotonic graph. The previous degree hard-reject is
gone — :meth:`ExtractionState.finalize` always succeeds when the
pending graph passed witness + id checks. After a successful
finalize the handler asks the state for a SOFT degree warning
(``compute_degree_warning``) and appends it as an advisory to the
success ``tool_result`` text. The model sees the hint but the firing
terminates either way.

This module is **not** an atom — the merged :mod:`atom` imports
:func:`build_finalize_extraction_tool`.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from pydantic import BaseModel, ConfigDict

from llmharness.runtime.decorator import harness_tool

from ..state import ExtractionState

FINALIZE_EXTRACTION_TOOL_NAME = "finalize_extraction"
FINALIZE_EXTRACTION_REASON = "llmharness:finalize_extraction"


class FinalizeExtractionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_finalize_extraction_tool(state: ExtractionState) -> FunctionTool:
    """Mint a :class:`FunctionTool` closing over ``state``."""

    @harness_tool(FINALIZE_EXTRACTION_TOOL_NAME, terminates=True)
    async def _finalize(_args: FinalizeExtractionArgs, _ctx: Any) -> ToolTerminate | ToolResult:
        """Terminate the extractor firing. Call this AFTER emitting every node/edge with upsert_node / upsert_edge (and any merges via delete_node / delete_edge). The handler commits the witness-valid graph and ends the firing. A soft warning about chain-link nodes may be appended to the success result — it's a hint for the NEXT firing, NOT a rejection of this one."""
        err = state.finalize()
        if err is not None:
            # Only failure path: caller mis-used the tool (already
            # finalized). Surface verbatim so the model can detect
            # and stop calling further tools.
            return ToolResult(
                content=[TextContent(type="text", text=err)],
                is_error=True,
            )
        digest = (
            f'{{"ok": true, "events": {len(state.events)}, '
            f'"edges": {len(state.edges)}, '
            f'"dropped": {len(state.dropped_edges)}}}'
        )
        warning = state.compute_degree_warning()
        text = "Graph committed. Note: " + warning + "\n\n" + digest if warning else digest
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text=text)]),
            reason=FINALIZE_EXTRACTION_REASON,
        )

    return _finalize


__all__ = [
    "FINALIZE_EXTRACTION_REASON",
    "FINALIZE_EXTRACTION_TOOL_NAME",
    "FinalizeExtractionArgs",
    "build_finalize_extraction_tool",
]
