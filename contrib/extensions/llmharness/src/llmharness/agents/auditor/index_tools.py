"""Auditor index query tools — let the auditor explore the trajectory and symbol table.

Instead of dumping the full context index as JSON in the system prompt,
these tools let the auditor query specifics on demand: read a turn,
search entities, check coverage, get a symbol's reference timeline.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agentm.core.abi import ExtensionAPI, FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel, Field


class _IndexState:
    """Holds the pre-built index data for tool closures."""

    def __init__(
        self,
        trajectory: list[dict[str, Any]],
        symbols: list[dict[str, Any]],
        references: list[dict[str, Any]],
        context_index: dict[str, Any] | None = None,
    ) -> None:
        self.trajectory = trajectory
        self.symbols = symbols
        self.references = references
        self.context_index = context_index or {}
        self._context_turns_by_index: dict[int, dict[str, Any]] = {}
        for turn in self.context_index.get("turns", []):
            if isinstance(turn, dict) and isinstance(turn.get("turn_index"), int):
                self._context_turns_by_index[turn["turn_index"]] = turn
        self._sym_by_id: dict[str, dict[str, Any]] = {}
        self._sym_by_name: dict[str, dict[str, Any]] = {}
        for s in symbols:
            sym_id = s.get("id") or s.get("name", "")
            self._sym_by_id[sym_id] = s
            name = s.get("name", "")
            if name:
                self._sym_by_name[name.lower()] = s
            for alias in s.get("aliases", []):
                if isinstance(alias, str):
                    self._sym_by_name[alias.lower()] = s
        self._refs_by_sym: dict[str, list[dict[str, Any]]] = {}
        for r in references:
            self._refs_by_sym.setdefault(r.get("symbol_id", ""), []).append(r)


def _text_result(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _build_get_turn_tool(state: _IndexState) -> FunctionTool:
    class Args(BaseModel):
        turn_index: int = Field(
            description="0-based turn index to read (see list_turns for the range)",
        )
        full: bool = Field(
            default=False,
            description=(
                "false (default): each content block is a preview clipped to a "
                "few hundred chars, with an explicit '[truncated N chars ...]' "
                "marker where content was cut. true: return the turn's blocks "
                "in full, no clipping. Re-reading the same turn without "
                "full=true always returns the same clipped preview — when a "
                "preview is marked truncated, retry with full=true instead."
            ),
        )

    def _clip(text: str, limit: int, full: bool) -> str:
        if full or len(text) <= limit:
            return text
        return (
            text[:limit]
            + f"\n[truncated {len(text) - limit} chars — call get_turn again "
            "with full=true for the complete content]"
        )

    async def handler(args: dict[str, Any]) -> ToolResult:
        parsed = Args.model_validate(args)
        idx = parsed.turn_index
        full = parsed.full
        if idx < 0 or idx >= len(state.trajectory):
            return _text_result(f"Turn {idx} out of range (0-{len(state.trajectory)-1})")
        turn = state.trajectory[idx]
        role = turn.get("role", "?")
        content = turn.get("content", [])
        # Compact by default: role + clipped content blocks; full=true lifts
        # every clip so the auditor can always reach complete evidence.
        blocks: list[str] = []
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    btype = b.get("type", "")
                    text = b.get("text", "")
                    name = b.get("name", "")
                    if btype == "tool_call" and name:
                        args_str = json.dumps(b.get("arguments", {}), ensure_ascii=False)
                        blocks.append(f"[tool_call: {name}] {_clip(args_str, 500, full)}")
                    elif btype == "tool_result":
                        sub = b.get("content", [])
                        sub_text = ""
                        if isinstance(sub, list):
                            for s in sub:
                                if isinstance(s, dict):
                                    sub_text += s.get("text", "")
                        blocks.append(f"[tool_result] {_clip(sub_text, 1000, full)}")
                    elif text:
                        blocks.append(_clip(text, 1500, full))
        else:
            blocks.append(_clip(str(content), 1500, full))
        return _text_result(f"Turn {idx} ({role}):\n" + "\n".join(blocks))

    return FunctionTool(
        name="get_turn",
        description=(
            "Read the content of a specific trajectory turn. "
            "Use this to verify what the agent actually did at a given turn "
            "before making claims about the agent's behavior. "
            "By default blocks are clipped previews with explicit truncation "
            "markers; pass full=true to read a turn's complete content."
        ),
        parameters=Args,
        fn=handler,
    )


def _build_search_entities_tool(state: _IndexState) -> FunctionTool:
    class Args(BaseModel):
        query: str = Field(description="Search query (entity name, kind, or keyword)")
        kind: str | None = Field(default=None, description="Filter by kind: service, metric, tool, skill, api, etc.")

    async def handler(args: dict[str, Any]) -> ToolResult:
        parsed = Args.model_validate(args)
        q = parsed.query.lower()
        results: list[dict[str, Any]] = []
        for s in state.symbols:
            name = s.get("name", "")
            kind = s.get("kind", "")
            aliases = s.get("aliases", [])
            all_names = [name.lower()] + [a.lower() for a in aliases if isinstance(a, str)]
            if parsed.kind and kind != parsed.kind:
                continue
            if q in name.lower() or any(q in a for a in all_names) or q in kind:
                ref_count = len(state._refs_by_sym.get(s.get("id") or s.get("name", ""), []))
                ref_kinds = Counter(
                    r.get("kind", "?") for r in state._refs_by_sym.get(s.get("id") or s.get("name", ""), [])
                )
                results.append({
                    "name": name,
                    "kind": kind,
                    "summary": s.get("summary", ""),
                    "aliases": aliases,
                    "ref_count": ref_count,
                    "ref_kinds": dict(ref_kinds),
                })
        if not results:
            return _text_result(f"No entities matching '{parsed.query}'" + (f" (kind={parsed.kind})" if parsed.kind else ""))
        return _text_result(json.dumps(results[:20], ensure_ascii=False, indent=2))

    return FunctionTool(
        name="search_entities",
        description=(
            "Search the symbol table for entities by name or kind. "
            "Returns matching entities with reference counts and kinds."
        ),
        parameters=Args,
        fn=handler,
    )


def _build_get_entity_timeline_tool(state: _IndexState) -> FunctionTool:
    class Args(BaseModel):
        name: str = Field(description="Entity name (or alias) to look up")

    async def handler(args: dict[str, Any]) -> ToolResult:
        parsed = Args.model_validate(args)
        sym = state._sym_by_name.get(parsed.name.lower())
        if sym is None:
            for entity in state.context_index.get("entities", []):
                if not isinstance(entity, dict):
                    continue
                names = [str(entity.get("name", ""))]
                aliases = entity.get("aliases")
                if isinstance(aliases, list):
                    names.extend(str(alias) for alias in aliases)
                if parsed.name.lower() not in {name.lower() for name in names}:
                    continue
                refs: list[dict[str, Any]] = []
                raw_turns = entity.get("turns")
                turns = raw_turns if isinstance(raw_turns, list) else []
                for turn_index in turns:
                    turn = state._context_turns_by_index.get(turn_index)
                    refs.append(
                        {
                            "turn": turn_index,
                            "kind": "context_index",
                            "text": str((turn or {}).get("summary", ""))[:100],
                        }
                    )
                return _text_result(json.dumps({"entity": entity, "references": refs}, ensure_ascii=False, indent=2))
            return _text_result(f"Entity '{parsed.name}' not found in the symbol table.")
        sym_id = sym["id"]
        refs = state._refs_by_sym.get(sym_id, [])
        if not refs:
            return _text_result(
                f"Entity '{sym['name']}' ({sym.get('kind','?')}) exists but has no references in the trajectory."
            )
        timeline: list[dict[str, Any]] = []
        for r in sorted(refs, key=lambda x: x.get("step_id", 0)):
            timeline.append({
                "turn": r.get("step_id"),
                "kind": r.get("kind", "?"),
                "text": r.get("text", "")[:100],
            })
        header = {
            "name": sym["name"],
            "kind": sym.get("kind", "?"),
            "summary": sym.get("summary", ""),
            "aliases": sym.get("aliases", []),
            "total_refs": len(refs),
        }
        return _text_result(json.dumps({"entity": header, "references": timeline}, ensure_ascii=False, indent=2))

    return FunctionTool(
        name="get_entity_timeline",
        description=(
            "Get the full reference timeline for a named entity: "
            "which turns reference it, with what kind (tool_input, tool_output, mention)."
        ),
        parameters=Args,
        fn=handler,
    )


def _build_list_entities_tool(state: _IndexState) -> FunctionTool:
    class Args(BaseModel):
        kind: str | None = Field(default=None, description="Filter by entity kind (e.g. 'service', 'metric', 'tool')")

    async def handler(args: dict[str, Any]) -> ToolResult:
        parsed = Args.model_validate(args)
        entities: list[dict[str, Any]] = []
        if state.symbols:
            for s in state.symbols:
                if parsed.kind and s.get("kind") != parsed.kind:
                    continue
                refs = state._refs_by_sym.get(s.get("id") or s.get("name", ""), [])
                ref_kinds = Counter(r.get("kind", "?") for r in refs)
                entities.append({
                    "name": s["name"],
                    "kind": s.get("kind", "?"),
                    "refs": len(refs),
                    "ref_kinds": dict(ref_kinds),
                })
        else:
            for entity in state.context_index.get("entities", []):
                if not isinstance(entity, dict):
                    continue
                kind = str(entity.get("type", "?"))
                if parsed.kind and kind != parsed.kind:
                    continue
                turns = entity.get("turns")
                ref_count = len(turns) if isinstance(turns, list) else 0
                entities.append({
                    "name": entity.get("name", ""),
                    "kind": kind,
                    "refs": ref_count,
                    "ref_kinds": {"context_index": ref_count},
                })
        entities.sort(key=lambda x: (-x["refs"], x["name"]))
        kind_counts = Counter(e["kind"] for e in entities)
        return _text_result(json.dumps({
            "total": len(entities),
            "by_kind": dict(kind_counts),
            "entities": entities,
        }, ensure_ascii=False, indent=2))

    return FunctionTool(
        name="list_entities",
        description=(
            "List all entities in the symbol table with reference counts and kinds. "
            "Optionally filter by entity kind. Use this to see what the index contains "
            "before drilling into specifics with search_entities or get_entity_timeline."
        ),
        parameters=Args,
        fn=handler,
    )


def _build_list_attention_hints_tool(state: _IndexState) -> FunctionTool:
    class Args(BaseModel):
        kind: str | None = Field(default=None, description="Optional hint kind filter")
        limit: int = Field(default=8, description="Maximum number of hints to return")

    async def handler(args: dict[str, Any]) -> ToolResult:
        parsed = Args.model_validate(args)
        hints = state.context_index.get("attention_hints", [])
        selected: list[dict[str, Any]] = []
        for hint in hints:
            if not isinstance(hint, dict):
                continue
            if parsed.kind and hint.get("kind") != parsed.kind:
                continue
            selected.append(hint)
            if len(selected) >= max(1, parsed.limit):
                break
        if not selected:
            return _text_result("No attention hints available" + (f" for kind={parsed.kind}" if parsed.kind else ""))
        return _text_result(json.dumps(selected, ensure_ascii=False, indent=2))

    return FunctionTool(
        name="list_attention_hints",
        description=(
            "List context-index attention hints such as competing observations, "
            "weak candidate signals, and local signals on disappeared entities. "
            "Use this to choose which concrete evidence to verify with get_turn."
        ),
        parameters=Args,
        fn=handler,
    )


def _build_list_turns_tool(state: _IndexState) -> FunctionTool:
    class Args(BaseModel):
        start: int = Field(default=0, description="Start turn index (inclusive)")
        end: int | None = Field(default=None, description="End turn index (exclusive); omit for all remaining")

    async def handler(args: dict[str, Any]) -> ToolResult:
        parsed = Args.model_validate(args)
        lo = max(0, parsed.start)
        hi = parsed.end if parsed.end is not None else len(state.trajectory)
        hi = min(hi, len(state.trajectory))
        lines: list[str] = [f"Trajectory: {len(state.trajectory)} turns total (showing {lo}-{hi-1})"]
        for i in range(lo, hi):
            turn = state.trajectory[i]
            role = turn.get("role", "?")
            content = turn.get("content", [])
            summary = ""
            tool_name = ""
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "tool_call" or b.get("name"):
                            tool_name = b.get("name", "")
                        elif b.get("type") == "tool_result":
                            sub = b.get("content", [])
                            if isinstance(sub, list):
                                for s in sub:
                                    if isinstance(s, dict) and s.get("text"):
                                        summary = s["text"][:80]
                                        break
                        elif b.get("text") and not summary:
                            summary = b["text"][:80]
            tag = f" → {tool_name}" if tool_name else ""
            lines.append(f"  [{i}] {role}{tag}: {summary}")
        return _text_result("\n".join(lines))

    return FunctionTool(
        name="list_turns",
        description=(
            "List trajectory turns with a compact summary of each. "
            "Use this first to get an overview of what the agent did, "
            "then drill into specific turns with get_turn."
        ),
        parameters=Args,
        fn=handler,
    )


def _build_list_claims_tool(state: _IndexState) -> FunctionTool | None:
    """Build list_claims tool from claim_structure in context_index."""
    ci = state.context_index or {}
    claim_structure = ci.get("claim_structure")
    if not claim_structure:
        return None

    class Args(BaseModel):
        pass

    async def handler(args: dict[str, Any]) -> ToolResult:
        span_roles = claim_structure.get("span_roles", {})
        commitment_points = claim_structure.get("commitment_points", [])

        lines: list[str] = []
        lines.append("Claim structure analysis (from Level 2):")
        lines.append("")

        commit_spans = sorted(
            int(k) for k, v in span_roles.items()
            if v in ("commit", "verify", "finalize")
        )
        explore_spans = sorted(
            int(k) for k, v in span_roles.items()
            if v == "explore"
        )
        lines.append(f"Commitment spans: {commit_spans}")
        lines.append(f"Exploration spans: {explore_spans}")
        lines.append("")

        if commitment_points:
            lines.append("Commitment points (potential errors):")
            for cp in commitment_points:
                grounded = cp.get("grounded", False)
                entity = cp.get("entity", "")
                reason = cp.get("reason", "")
                lines.append(
                    f"  span {cp['span']}: entity='{entity}' "
                    f"grounded={grounded} — {reason}"
                )

        return _text_result("\n".join(lines))

    return FunctionTool(
        name="list_claims",
        description=(
            "List the claim structure from Level 2 analysis: which spans are "
            "commitment points (decisions/conclusions/finalizations) vs "
            "exploration (search/retrieval). Focus error localization on "
            "commitment spans — exploration spans are almost never errors."
        ),
        parameters=Args,
        fn=handler,
    )


MANIFEST = ExtensionManifest(
    name="auditor_index_tools",
    description="Register trajectory and index query tools for the auditor.",
    registers=(
        "tool:list_turns",
        "tool:get_turn",
        "tool:search_entities",
        "tool:get_entity_timeline",
        "tool:list_entities",
        "tool:list_attention_hints",
        "tool:list_claims",
    ),
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    trajectory = config.get("trajectory", [])
    symbols = config.get("symbols", [])
    references = config.get("references", [])
    context_index = config.get("context_index")
    state = _IndexState(
        trajectory,
        symbols,
        references,
        context_index if isinstance(context_index, dict) else None,
    )
    api.register_tool(_build_list_turns_tool(state))
    api.register_tool(_build_get_turn_tool(state))
    api.register_tool(_build_search_entities_tool(state))
    api.register_tool(_build_get_entity_timeline_tool(state))
    api.register_tool(_build_list_entities_tool(state))
    claims_tool = _build_list_claims_tool(state)
    if claims_tool is not None:
        api.register_tool(claims_tool)
    api.register_tool(_build_list_attention_hints_tool(state))
