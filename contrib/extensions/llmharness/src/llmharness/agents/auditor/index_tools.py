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
    ) -> None:
        self.trajectory = trajectory
        self.symbols = symbols
        self.references = references
        self._sym_by_id: dict[str, dict[str, Any]] = {s["id"]: s for s in symbols}
        self._sym_by_name: dict[str, dict[str, Any]] = {}
        for s in symbols:
            self._sym_by_name[s["name"].lower()] = s
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
        turn_index: int = Field(description="0-based turn index to read")

    async def handler(args: dict[str, Any]) -> ToolResult:
        parsed = Args.model_validate(args)
        idx = parsed.turn_index
        if idx < 0 or idx >= len(state.trajectory):
            return _text_result(f"Turn {idx} out of range (0-{len(state.trajectory)-1})")
        turn = state.trajectory[idx]
        role = turn.get("role", "?")
        content = turn.get("content", [])
        # Compact: show role + content blocks
        blocks: list[str] = []
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    btype = b.get("type", "")
                    text = b.get("text", "")
                    name = b.get("name", "")
                    if btype == "tool_call" and name:
                        args_str = json.dumps(b.get("arguments", {}), ensure_ascii=False)
                        blocks.append(f"[tool_call: {name}] {args_str[:500]}")
                    elif btype == "tool_result":
                        sub = b.get("content", [])
                        sub_text = ""
                        if isinstance(sub, list):
                            for s in sub:
                                if isinstance(s, dict):
                                    sub_text += s.get("text", "")[:800]
                        blocks.append(f"[tool_result] {sub_text[:1000]}")
                    elif text:
                        blocks.append(text[:1500])
        else:
            blocks.append(str(content)[:1500])
        return _text_result(f"Turn {idx} ({role}):\n" + "\n".join(blocks))

    return FunctionTool(
        name="get_turn",
        description=(
            "Read the content of a specific trajectory turn. "
            "Use this to verify what the agent actually did at a given turn "
            "before making claims about the agent's behavior."
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
                ref_count = len(state._refs_by_sym.get(s["id"], []))
                ref_kinds = Counter(
                    r.get("kind", "?") for r in state._refs_by_sym.get(s["id"], [])
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
        for s in state.symbols:
            if parsed.kind and s.get("kind") != parsed.kind:
                continue
            refs = state._refs_by_sym.get(s["id"], [])
            ref_kinds = Counter(r.get("kind", "?") for r in refs)
            entities.append({
                "name": s["name"],
                "kind": s.get("kind", "?"),
                "refs": len(refs),
                "ref_kinds": dict(ref_kinds),
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


MANIFEST = ExtensionManifest(
    name="auditor_index_tools",
    description="Register trajectory and index query tools for the auditor.",
    registers=(
        "tool:list_turns",
        "tool:get_turn",
        "tool:search_entities",
        "tool:get_entity_timeline",
        "tool:list_entities",
    ),
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    trajectory = config.get("trajectory", [])
    symbols = config.get("symbols", [])
    references = config.get("references", [])
    state = _IndexState(trajectory, symbols, references)
    api.register_tool(_build_list_turns_tool(state))
    api.register_tool(_build_get_turn_tool(state))
    api.register_tool(_build_search_entities_tool(state))
    api.register_tool(_build_get_entity_timeline_tool(state))
    api.register_tool(_build_list_entities_tool(state))
