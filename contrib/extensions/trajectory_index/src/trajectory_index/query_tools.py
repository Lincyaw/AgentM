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

from typing import Any, Final

from agentm.core.abi import ExtensionAPI, FunctionTool, TextContent, ToolResult
from pydantic import BaseModel, Field

from .ir.index import TrajectoryIndex
from .ir.models import normalize_name
from .pass1_nodes.serialize import JsonValue

INDEX_SERVICE_KEY: Final = "trajectory_index.index"


def _resolve_symbol(index: TrajectoryIndex, key: str) -> str | None:
    """Resolve a symbol reference that may be an id OR a name/alias → id.

    The auditor naturally passes an entity NAME to get_symbol_context, but the
    tool historically required the internal id and dead-ended on "Symbol not
    found" — forcing a search_symbols detour. Accept both: id first, then an
    exact name/alias match (whitespace/case-insensitive via normalize_name).
    """
    if key in index.symbols:
        return key
    norm = normalize_name(key)
    if not norm:
        return None
    for sym in index.symbols.values():
        if normalize_name(sym.canonical_name) == norm:
            return sym.id
        if any(normalize_name(a) == norm for a in sym.aliases):
            return sym.id
    return None


def _grounding_timeline(index: TrajectoryIndex, symbol_id: str) -> str:
    """Compact occurrence timeline for a flagged symbol, inlined into
    get_insights so the auditor need not follow up with get_symbol_context.

    Each occurrence as ``step<id>:<kind>``; a trailing count of tool-backed
    occurrences makes the grounding verdict self-contained (0 tool-backed = the
    entity was never produced by a tool).
    """
    try:
        ctx = index.get_context(symbol_id)
    except KeyError:
        return ""
    items = ctx.timeline[:8]
    if not items:
        return ""
    parts = [f"step{it.step.step_id}:{it.reference.kind}" for it in items]
    n_tool = sum(1 for it in ctx.timeline if it.reference.grounded)
    more = "" if len(ctx.timeline) <= 8 else f" +{len(ctx.timeline) - 8} more"
    return ", ".join(parts) + more + f"  ({n_tool} tool-backed)"


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
        symbol_id: str = Field(
            description="Symbol id (sym_…) OR the entity name/alias — both work"
        )

    async def _handle(args: dict[str, JsonValue]) -> ToolResult:
        params = ContextParams.model_validate(args)
        index = api.get_service(INDEX_SERVICE_KEY)
        assert isinstance(index, TrajectoryIndex)

        resolved = _resolve_symbol(index, params.symbol_id)
        if resolved is None:
            return ToolResult(
                content=[TextContent(type="text", text=(
                    f"Symbol not found: {params.symbol_id!r} — no id or exact "
                    "name/alias matches. Use search_symbols to discover it."
                ))],
                is_error=True,
            )
        ctx = index.get_context(resolved)

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
        attn = index.constraint_attention()

        # --- 1. Contradictions: lead with these, but the polarity is the
        #        MODEL's call, not a code fact. Code verifies the witness quote
        #        is verbatim-present (SCHEMA §2.3); whether it actually
        #        contradicts the claim is a model judgment that can be wrong
        #        (§2.6 model-polarity over-alarm), so frame as a lead to
        #        confirm, never "highest signal". ---
        edges_by_claim: dict[str, list[Any]] = {}
        for e in index.edges.values():
            edges_by_claim.setdefault(e.src, []).append(e)

        contra: list[str] = []
        for f in index.claim_findings:
            if f.status != "conflicted":
                continue
            c = index.claims.get(f.claim_id)
            if not c:
                continue
            confs = [e for e in edges_by_claim.get(f.claim_id, []) if e.kind == "conflicts"]
            quote = confs[0].quote[:120] if confs else ""
            dst = confs[0].dst if confs else "?"
            contra.append(
                f"- claim conflicted — step {f.step_id}: {c.text[:110]!r}\n"
                f"    vs observation step {dst}: {quote!r}"
            )
        for d in index.get_dependencies():
            if d.risk == "contradicted":
                sym = index.symbols.get(d.symbol_id)
                name = sym.canonical_name if sym else d.symbol_id
                contra.append(
                    f"- value contradicted — {name!r}: def step {d.def_step_id} → use step {d.use_step_id}"
                )
        for a in attn:
            if a["kind"] == "constraint_violated":
                step = f"  [step {a['step_id']}]" if a.get("step_id") else ""
                contra.append(f"- constraint violated — {a['summary']}{step}")
        if contra:
            out.append(
                "## Contradictions (witness quote verbatim-verified; the "
                "conflict itself is the model's judgment — confirm by reading "
                "both steps before trusting)"
            )
            out += contra

        # --- 1b. Agent self-contradictions (claim↔claim, SCHEMA §2.8):
        #        the agent asserting incompatible things about one entity, with
        #        NO environment evidence needed — belief revision / flip-flops.
        #        Propositional tier: a monotone advisory (model NLI, no verbatim
        #        certificate), so it is a lead to confirm, never a verdict. ---
        self_contra: list[str] = []
        for e in index.edges.values():
            if e.kind != "self_contradicts":
                continue
            ca = index.claims.get(e.src)
            cb = index.claims.get(e.dst)
            if not ca or not cb:
                continue
            self_contra.append(
                f"- self-contradiction — step {ca.step_id}: {ca.text[:100]!r}\n"
                f"    vs step {cb.step_id}: {cb.text[:100]!r}"
            )
        if self_contra:
            out.append(
                "\n## Agent self-contradictions (advisory — the agent vs "
                "itself, model-judged; confirm by reading both spans)"
            )
            out += self_contra

        # --- 2. Grounding flags (named/used but not tool-backed). Self-
        #        contained: each carries its id AND occurrence timeline, so the
        #        auditor confirms grounding here instead of a search_symbols +
        #        get_symbol_context round-trip per flag. ---
        warns = index.warnings()
        if warns:
            out.append(
                "\n## Grounding flags (named/used but not tool-backed — id + "
                "occurrence timeline inline; no follow-up lookup needed)"
            )
            for i, w in enumerate(warns):
                steps = f"  [steps: {', '.join(w.step_ids)}]" if w.step_ids else ""
                out.append(f"- {w.kind} — {w.symbol_name!r} (id={w.symbol_id}): {w.detail}{steps}")
                # inline the timeline for the top flags (bounds length on
                # flag-heavy trajectories; the rest still carry id + steps)
                if i < 15:
                    tl = _grounding_timeline(index, w.symbol_id)
                    if tl:
                        out.append(f"    seen: {tl}")

        # --- 3. Constraint gaps: committed without verifying (omitted) ---
        omitted = [a for a in attn if a["kind"] == "constraint_omitted"]
        if omitted:
            out.append("\n## Constraint gaps (committed without verifying a requirement)")
            for a in omitted:
                step = f"  [step {a['step_id']}]" if a.get("step_id") else ""
                out.append(f"- {a['summary']}{step}")

        # --- 4. Unsupported claims: LOW signal, de-emphasized + capped. A claim
        #        with no linked observation is mostly baseline — agent narration
        #        or evidence the trajectory never recovered — not a finding. ---
        unsourced = [f for f in index.claim_findings if f.status == "unsourced"]
        if unsourced:
            out.append(
                f"\n## Unsupported claims ({len(unsourced)} — low signal: assertions "
                "with no linked observation; often narration or unrecovered evidence)"
            )
            for f in unsourced[:6]:
                c = index.claims.get(f.claim_id)
                out.append(f"- step {f.step_id}: {(c.text[:100] if c else f.claim_id)!r}")
            if len(unsourced) > 6:
                out.append(f"- … +{len(unsourced) - 6} more")

        # --- 5. Claim analysis ABSENCE (SCHEMA §3 claims_empty): Pass 1
        #        extracted zero claims (terse/search-heavy trajectory, or an
        #        extraction failure). Absence of the claim-status /
        #        unsupported-commitment signal is NOT cleanliness — say so, so a
        #        reader falls back to grounding + constraints instead of reading
        #        silence as clean. Key on index.claims: claim_findings is empty
        #        whenever the fold did not run, even on a claim-rich index. ---
        if not index.claims:
            out.append(
                "\n## Claim analysis absent\n"
                "- Pass 1 extracted 0 assertions from this trajectory "
                "(terse/search-heavy, or an extraction failure). Claim-status "
                "and unsupported-commitment signals are UNAVAILABLE here — not "
                "clean. Rely on grounding flags + constraints, and read "
                "commitments directly from the spans."
            )

        if not out:
            text = (
                "No insights surfaced: grounding, claims, and constraints look "
                "clean — or those passes have not run yet."
            )
        else:
            text = (
                "Possible issues the index derived (most actionable first; "
                "leads to inspect, not verdicts):\n\n" + "\n".join(out)
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
