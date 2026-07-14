"""Read-side tools over a built ``TrajectoryIndex`` — the single query surface.

These builders are the ONE set of index-query tools. Both atoms register them:
``atom.py`` (which builds the index live from the running session) and
``query_atom.py`` (which loads a persisted ``index.json`` and only reads it).
They live here — a plain library module, not an atom — so both atoms can share
them without an atom-to-atom import (§11).

Every builder reads the index from ``api.get_service(INDEX_SERVICE_KEY)``; the
mounting atom is responsible for having put a ``TrajectoryIndex`` there.
"""

from __future__ import annotations

from typing import Final

from agentm.core.abi import ExtensionAPI, FunctionTool, TextContent, ToolResult
from pydantic import BaseModel, Field

from .ir.index import TrajectoryIndex
from .pass1_nodes.serialize import JsonValue

INDEX_SERVICE_KEY: Final = "trajectory_index.index"


def build_search_tool(api: ExtensionAPI) -> FunctionTool:
    class SearchParams(BaseModel):
        query: str = Field(description="Search query (symbol name, concept, or keyword)")
        kinds: list[str] | None = Field(default=None, description="Filter by symbol kinds")
        limit: int = Field(default=10, description="Max results")

    async def _handle(args: dict[str, JsonValue]) -> ToolResult:
        params = SearchParams.model_validate(args)
        index = api.get_service(INDEX_SERVICE_KEY)
        assert isinstance(index, TrajectoryIndex)

        kind_filter = set(params.kinds) if params.kinds else None

        results = index.search(
            params.query,
            kinds=kind_filter,
            limit=params.limit,
            include_references=True,
            include_related=True,
        )
        if not results:
            return ToolResult(content=[TextContent(type="text", text="No symbols found.")])

        lines: list[str] = []
        for r in results:
            sym = r.symbol
            lines.append(
                f"- **{sym.canonical_name}** ({sym.kind}, score={r.score:.2f})  id={sym.id}"
            )
            if sym.summary:
                lines.append(f"  {sym.summary}")
            if r.references:
                refs = ", ".join(f"step {ref.step_id}:{ref.kind}" for ref in r.references[:3])
                lines.append(f"  references: {refs}")
            if r.related:
                rels = ", ".join(
                    f"{rel.symbol.canonical_name}({rel.score:.2f})" for rel in r.related[:3]
                )
                lines.append(f"  related: {rels}")

        return ToolResult(content=[TextContent(type="text", text="\n".join(lines))])

    return FunctionTool(
        name="search_symbols",
        description="Search for symbols in the trajectory semantic index.",
        parameters=SearchParams,
        fn=_handle,
    )


def build_context_tool(api: ExtensionAPI) -> FunctionTool:
    class ContextParams(BaseModel):
        symbol_id: str = Field(description="Symbol ID to get context for")

    async def _handle(args: dict[str, JsonValue]) -> ToolResult:
        params = ContextParams.model_validate(args)
        index = api.get_service(INDEX_SERVICE_KEY)
        assert isinstance(index, TrajectoryIndex)

        try:
            ctx = index.get_context(params.symbol_id)
        except KeyError:
            return ToolResult(
                content=[TextContent(type="text", text=f"Symbol not found: {params.symbol_id}")],
                is_error=True,
            )

        lines: list[str] = []
        sym = ctx.symbol
        lines.append(f"# {sym.canonical_name} ({sym.kind})")
        if sym.summary:
            lines.append(f"\n{sym.summary}")
        if sym.aliases:
            lines.append(f"\nAliases: {', '.join(sorted(sym.aliases))}")

        if ctx.definition:
            d = ctx.definition
            lines.append(f'\n## Definition\nStep {d.step_id}: "{d.text}" ({d.kind})')

        if ctx.timeline:
            lines.append("\n## Timeline")
            for item in ctx.timeline[:15]:
                lines.append(
                    f"- [{item.step.role}] step {item.step.step_id}: "
                    f'"{item.reference.text}" ({item.reference.kind})'
                )

        if ctx.related:
            lines.append("\n## Related symbols")
            for rel in ctx.related[:10]:
                rel_types = ", ".join(r.type for r in rel.relations[:3])
                lines.append(
                    f"- {rel.symbol.canonical_name} ({rel.symbol.kind}) "
                    f"— {rel_types} (score={rel.score:.2f})"
                )

        return ToolResult(content=[TextContent(type="text", text="\n".join(lines))])

    return FunctionTool(
        name="get_symbol_context",
        description=(
            "Get full context for a specific symbol: definition, timeline, "
            "related symbols, and surrounding trajectory snippets."
        ),
        parameters=ContextParams,
        fn=_handle,
    )


def build_insights_tool(api: ExtensionAPI) -> FunctionTool:
    """One read tool over everything the index's analysis passes concluded.

    ``index_trajectory`` *builds* the index (runs the passes); this *reads*
    their conclusions back as one ranked feed of possible issues — the whole
    point of the passes. Nothing else surfaces them, so without this the
    grounding flags, claim statuses and constraint gaps stay buried in the
    index. Each line carries the step ids to go inspect; the agent decides
    what is real (these are leads, not verdicts).
    """

    async def _handle(args: dict[str, JsonValue]) -> ToolResult:
        index = api.get_service(INDEX_SERVICE_KEY)
        assert isinstance(index, TrajectoryIndex)

        out: list[str] = []

        # Pass 3 grounding warnings (already severity-sorted).
        warns = index.warnings()
        if warns:
            out.append("## Grounding flags (named/used but not tool-backed)")
            for w in warns:
                steps = f"  [steps: {', '.join(w.step_ids)}]" if w.step_ids else ""
                out.append(f"- {w.kind} — {w.symbol_name!r}: {w.detail}{steps}")

        # Pass 3.5 value contradictions (dependency risk).
        contradicted = [d for d in index.get_dependencies() if d.risk == "contradicted"]
        if contradicted:
            out.append("\n## Value contradictions (used a value a tool contradicts)")
            for d in contradicted:
                sym = index.symbols.get(d.symbol_id)
                name = sym.canonical_name if sym else d.symbol_id
                out.append(f"- {name!r}: def step {d.def_step_id} → use step {d.use_step_id}")

        # Pass 3 claim status: conflicted dominates, then unsourced.
        bad_claims = [f for f in index.claim_findings if f.status in ("conflicted", "unsourced")]
        bad_claims.sort(key=lambda f: (f.status != "conflicted", f.step_id))
        if bad_claims:
            out.append("\n## Claims without support")
            for f in bad_claims:
                c = index.claims.get(f.claim_id)
                text = (c.text[:120] if c else f.claim_id)
                out.append(f"- {f.status} — step {f.step_id}: {text!r}")

        # Pass 3 constraint layer: violated / omitted (already filtered).
        attn = index.constraint_attention()
        if attn:
            out.append("\n## Constraint gaps (answer fails or never verified a requirement)")
            for a in attn:
                step = f"  [step: {a['step_id']}]" if a.get("step_id") else ""
                out.append(f"- {a['kind']} — {a['summary']}{step}")

        if not out:
            text = (
                "No insights surfaced: grounding, claims, and constraints look "
                "clean — or those passes have not run yet."
            )
        else:
            text = (
                "Possible issues the index derived (leads to inspect, not verdicts):\n\n"
                + "\n".join(out)
            )
        return ToolResult(content=[TextContent(type="text", text=text)])

    return FunctionTool(
        name="get_insights",
        description=(
            "Review everything the trajectory index's analysis passes concluded: "
            "grounding flags (fabricated/ungrounded names), value contradictions, "
            "unsupported claims, and unmet task constraints — one ranked feed "
            "of possible issues, each with the step ids to inspect."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        fn=_handle,
    )


def register_query_tools(api: ExtensionAPI) -> None:
    """Register the three read tools (search / context / insights) on ``api``."""
    api.register_tool(build_search_tool(api))
    api.register_tool(build_context_tool(api))
    api.register_tool(build_insights_tool(api))
