"""``upsert_edge`` extractor tool — Pydantic-backed.

Insert or replace one witness-bearing edge keyed by ``(src, dst, kind)``.
Both endpoint nodes must already exist in the current folded graph
(this firing or any prior firing). Witness rules:

* ``kind='data'`` requires non-empty ``cited_entities``;
* ``kind='ref'`` requires ``cited_quote`` to appear (case+whitespace
  normalised substring) in BOTH the src and dst nodes' source_turns
  text.

This module is **not** an atom.
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

UPSERT_EDGE_TOOL_NAME = "upsert_edge"

_EdgeKindLiteral = Literal["data", "ref"]


class UpsertEdgeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: int = Field(
        description=(
            "Source node id — must exist in the current graph (this "
            "firing or any prior firing)."
        ),
    )
    dst: int = Field(
        description=(
            "Destination node id — must exist in the current graph "
            "(this firing or any prior firing)."
        ),
    )
    kind: _EdgeKindLiteral = Field(
        description=(
            "Edge kind. 'ref' for textual-witness edges (cited_quote); "
            "'data' for entity-witness edges (cited_entities)."
        ),
    )
    reason: str = Field(
        description="One short sentence explaining the causal connection.",
    )
    cited_entities: list[str] = Field(
        description=(
            "Concrete identifiers (service names, span ids, file "
            "paths, error codes, etc.) shared by src and dst. "
            "Required non-empty when kind='data'; each entity must "
            "appear (case+whitespace normalized substring) in at "
            "least one of the src or dst node's source_turns text. "
            "Pass [] when kind='ref'."
        ),
    )
    cited_quote: str = Field(
        description=(
            "Verbatim substring of BOTH the src node's source_turns "
            "text AND the dst node's source_turns text "
            "(case+whitespace normalized). Required when kind='ref'. "
            "Paraphrasing or reformatting is rejected at op-build "
            "time. Pass \"\" when kind='data'."
        ),
    )


def _attempt_echo(args: UpsertEdgeArgs) -> str:
    if args.kind == "data":
        return (
            f"upsert_edge(src={args.src}, dst={args.dst}, kind=data, "
            f"cited_entities={args.cited_entities!r})"
        )
    quote_preview = args.cited_quote[:60] + ("..." if len(args.cited_quote) > 60 else "")
    return (
        f"upsert_edge(src={args.src}, dst={args.dst}, kind=ref, "
        f"cited_quote={quote_preview!r})"
    )


def build_upsert_edge_tool(state: ExtractionState) -> FunctionTool:
    @harness_tool(UPSERT_EDGE_TOOL_NAME)
    async def _upsert_edge(args: UpsertEdgeArgs, _ctx: Any) -> ToolResult:
        """Insert or replace one witness-bearing edge keyed by (src, dst, kind). Both endpoint nodes must already exist in the current graph (this firing or any prior firing). kind='data' requires non-empty cited_entities; kind='ref' requires cited_quote to appear in BOTH endpoint nodes' source_turns text."""
        raw = args.model_dump()
        result = state.apply_edge_upsert(raw)
        if isinstance(result, str):
            nodes, _edges = state._folded_view()
            existing = sorted(nodes.keys())[:8]
            if args.kind == "ref":
                options = [
                    "re-call upsert_edge with a cited_quote that is a verbatim "
                    "substring of BOTH endpoints' source_turns text (after "
                    "case+whitespace normalisation)",
                    "switch to kind='data' if no shared literal quote exists, "
                    "and pass cited_entities=[<shared identifier>, ...]",
                    f"verify src/dst are existing node ids — folded view contains {existing}",
                ]
            else:
                options = [
                    "re-call upsert_edge with non-empty cited_entities — each "
                    "entry must be a concrete identifier present in at least one "
                    "endpoint's source_turns text",
                    "switch to kind='ref' with cited_quote if a verbatim shared "
                    "substring exists in BOTH endpoints",
                    f"verify src/dst are existing node ids — folded view contains {existing}",
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

    return _upsert_edge


__all__ = ["UPSERT_EDGE_TOOL_NAME", "UpsertEdgeArgs", "build_upsert_edge_tool"]
