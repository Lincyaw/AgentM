"""Trajectory semantic index atom.

Registers tools for the main agent to build and query a semantic index
over its own trajectory. Extraction is delegated to a child agent
(entity_extractor scenario) that runs on a small/fast model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import Any, Final

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    ExtensionAPI,
    FunctionTool,
    LoopConfig,
    SessionEntry,
    TextContent,
    ToolCallBlock,
    ToolResult,
    ToolResultBlock,
)
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, Field

from .agents import extractor_scenario
from .agents.entity_extractor.schema import ExtractionResult
from .data import JsonValue, ProviderSpec
from .index import (
    Step,
    StepRole,
    TrajectoryIndex,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_SERVICE_KEY: Final = "trajectory_index.index"

_ROLE_MAP: Final = {
    "user": StepRole.USER,
    "assistant": StepRole.ASSISTANT,
    "tool_result": StepRole.TOOL_RESULT,
    "system": StepRole.SYSTEM,
}


class TrajectoryIndexConfig(BaseModel):
    model: str = Field(
        default="qwen",
        description="config.toml model profile name for the extraction agent",
    )
    vocabulary: str = Field(
        default="default",
        description="Vocabulary name for symbol kind validation (default, coding, research)",
    )
    resolve_aliases: bool = Field(
        default=True,
        description="Run Pass 2 name resolution (merge same-entity surface forms) "
        "before the def-use layer. Best-effort: a model failure degrades to the "
        "deterministic Pass 1+3 without merging.",
    )
    resolve_model: str = Field(
        default="",
        description="Model profile for the Pass 2 same-entity judgment. Empty "
        "reuses the extraction model.",
    )


MANIFEST = ExtensionManifest(
    name="trajectory_index",
    description=(
        "Build and query a semantic symbol-reference index over the current "
        "session trajectory using a small extraction model."
    ),
    registers=(
        "tool:index_trajectory",
        "tool:search_symbols",
        "tool:get_symbol_context",
    ),
    config_schema=TrajectoryIndexConfig,
)


# ---------------------------------------------------------------------------
# Session branch → clean messages (for extraction) + Steps (for index)
# ---------------------------------------------------------------------------


def _branch_to_clean_messages(branch: list[SessionEntry]) -> list[dict[str, JsonValue]]:
    """Convert session branch entries to the clean message format for extraction."""
    from .data import clean_trace_messages

    raw: list[dict[str, JsonValue]] = []
    for entry in branch:
        if entry.type == "message":
            raw.append(
                {
                    "id": entry.id,
                    "payload": _agentmsg_to_payload(entry.payload),
                }
            )
    return clean_trace_messages(raw)


def _agentmsg_to_payload(msg: AgentMessage) -> dict[str, JsonValue]:
    """Serialize an AgentMessage to the trace payload dict shape."""
    blocks: list[JsonValue] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCallBlock):
            blocks.append(
                {
                    "type": "tool_call",
                    "name": block.name,
                    "arguments": block.arguments,
                }
            )
        elif isinstance(block, ToolResultBlock):
            sub_blocks: list[JsonValue] = []
            for sub in block.content:
                if isinstance(sub, TextContent):
                    sub_blocks.append({"type": "text", "text": sub.text})
            tr: dict[str, JsonValue] = {"type": "tool_result", "content": sub_blocks}
            if not block.deterministic:
                tr["deterministic"] = False
            blocks.append(tr)
    return {"role": msg.role, "content": blocks}


def _clean_messages_to_steps(
    messages: list[dict[str, JsonValue]],
    run_id: str,
    start_index: int = 0,
) -> list[Step]:
    """Convert clean messages to index Steps (for populating the in-memory index)."""
    steps: list[Step] = []
    for i, msg in enumerate(messages, start=start_index):
        msg_role = msg.get("role", "")
        role = _ROLE_MAP.get(str(msg_role), StepRole.USER)
        parts: list[str] = []
        tool_name: str | None = None

        content_blocks = msg.get("content", [])
        if not isinstance(content_blocks, list):
            continue
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type", ""))
            if btype == "text":
                text = block.get("text", "")
                parts.append(str(text))
            elif btype == "tool_call":
                name = block.get("name")
                tool_name = str(name) if name is not None else None
                parts.append(f"[tool_call: {tool_name}]")
            elif btype == "tool_result":
                sub_content = block.get("content", [])
                if not isinstance(sub_content, list):
                    continue
                for sub in sub_content:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        parts.append(str(sub.get("text", "")))

        content = "\n".join(parts)
        if not content.strip():
            continue

        msg_id = msg.get("id")
        steps.append(
            Step(
                run_id=run_id,
                step_id=str(msg_id) if msg_id is not None else f"s{i}",
                index=i,
                role=role,
                content=content,
                tool_name=tool_name,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Child session orchestration
# ---------------------------------------------------------------------------


_MAX_RETRIES: Final = 3
_RETRY_DELAY: Final = 5.0


def _try_parse_response(
    messages: list[AgentMessage],
    vocabulary: str = "default",
) -> tuple[ExtractionResult | None, str | None]:
    """Try to parse an ExtractionResult from assistant messages."""
    from .data import _try_parse_response as _parse

    return _parse(messages, vocabulary)


def build_extraction_config(
    *,
    cwd: str,
    model: str | None = None,
    provider: ProviderSpec | None = None,
    vocabulary: str = "default",
    parent_session_id: str | None = None,
) -> AgentSessionConfig:
    """Build the ``AgentSessionConfig`` for an extractor child session."""
    return AgentSessionConfig(
        cwd=cwd,
        model=model,
        provider=provider,
        scenario=extractor_scenario(),
        purpose="trajectory_symbol_extractor",
        loop_config=LoopConfig(max_turns=1),
        lineage={
            "kind": "trajectory_index_extraction",
            "parent_session_id": parent_session_id or "",
        },
        atom_config_overrides={
            "trajectory_extractor_context": {"vocabulary": vocabulary},
        },
    )


def _agentmsg_to_extraction_dict(
    msg: AgentMessage, index: int,
) -> dict[str, JsonValue]:
    """Serialize one AgentMessage to the extraction input format."""
    from .data import _truncate_block

    payload = _agentmsg_to_payload(msg)
    role = payload.get("role", "")
    content = payload.get("content", [])
    if not isinstance(content, list) or not content:
        return {}
    blocks: list[JsonValue] = [_truncate_block(b) for b in content if isinstance(b, dict)]
    return {"id": str(index), "role": role, "content": blocks}


_IDENT_CHAR = re.compile(r"[A-Za-z0-9_.\-/]")


def _at_word_boundary(text: str, start: int, end: int) -> bool:
    """Check that the match at text[start:end] is not inside a longer identifier."""
    if start > 0 and _IDENT_CHAR.match(text[start - 1]):
        return False
    return not (end < len(text) and _IDENT_CHAR.match(text[end]))


def _mark_known_symbols(text: str, known_names: list[str]) -> str:
    """Wrap occurrences of known symbol names with [[...]] in the text.

    Only marks at word boundaries — a known symbol inside a longer
    identifier is not marked (the longer form may be a different entity).
    """
    if not known_names:
        return text
    for name in sorted(known_names, key=len, reverse=True):
        if not name or len(name) < 2:
            continue
        lower = text.lower()
        search = name.lower()
        result_parts: list[str] = []
        pos = 0
        while pos < len(text):
            idx = lower.find(search, pos)
            if idx < 0:
                result_parts.append(text[pos:])
                break
            end = idx + len(name)
            # Skip if already inside [[...]]
            if idx >= 2 and text[idx - 2:idx] == "[[":
                result_parts.append(text[pos:end])
                pos = end
                continue
            # Skip if inside a longer identifier
            if not _at_word_boundary(text, idx, end):
                result_parts.append(text[pos:end])
                pos = end
                continue
            result_parts.append(text[pos:idx])
            result_parts.append(f"[[{text[idx:end]}]]")
            pos = end
        text = "".join(result_parts)
    return text


def _format_message_compact(
    msg: dict[str, Any],
    known_names: list[str] | None = None,
) -> str:
    """Format one serialized message as compact text for the extraction prompt.

    If ``known_names`` is provided, occurrences of those names in the text
    are wrapped with ``[[...]]`` to mark them as already extracted.
    """
    mid = msg.get("id", "")
    role = msg.get("role", "")
    blocks = msg.get("content", [])
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "tool_call":
            name = block.get("name", "")
            args = block.get("arguments", block.get("input", {}))
            arg_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            parts.append(f"[tool_call: [[{name}]]]\n{arg_str}" if known_names and name.lower() in {n.lower() for n in known_names} else f"[tool_call: {name}]\n{arg_str}")
        elif btype == "tool_result":
            sub = block.get("content", [])
            if isinstance(sub, list):
                for s in sub:
                    if isinstance(s, dict):
                        parts.append(str(s.get("text", "")))
            else:
                parts.append(str(sub))
    body = "\n".join(p for p in parts if p)
    if known_names:
        body = _mark_known_symbols(body, known_names)
    return f"[{mid}|{role}]\n{body}"


def build_extraction_prompt(
    messages: list[AgentMessage],
    *,
    registry: list[dict[str, Any]] | None = None,
    message_id_start: int = 0,
) -> str:
    """Build the extractor's input prompt from trajectory messages.

    Known symbols from the registry are marked inline with ``[[name]]``
    in the message text. No separate known_symbols section is needed.
    """
    serialized = [
        d for i, m in enumerate(messages, start=message_id_start)
        if (d := _agentmsg_to_extraction_dict(m, i))
    ]
    known_names = [str(e.get("name", "")) for e in registry] if registry else None
    formatted = "\n\n".join(
        text for msg in serialized
        if (text := _format_message_compact(msg, known_names))
    )
    return formatted


async def run_extraction_session(
    config: AgentSessionConfig,
    prompt: str,
    *,
    spawn: Any = None,
    vocabulary: str = "default",
) -> ExtractionResult | None:
    """Run the extractor child and parse the result.

    ``spawn`` creates a child session from a config. Defaults to
    ``AgentSession.create``; callers inside an atom pass
    ``api.spawn_child_session``.
    """
    from agentm.core.runtime import AgentSession

    if spawn is None:
        spawn = AgentSession.create

    for attempt in range(_MAX_RETRIES):
        try:
            child = await spawn(config)
            try:
                child_msgs: list[AgentMessage] = await child.prompt(prompt)
                result, error = _try_parse_response(child_msgs, vocabulary)
                if result:
                    return result
                if error:
                    logger.warning("extraction parse failed: {}", error)
                    retry_child = await spawn(config)
                    try:
                        retry_prompt = (
                            f"Your previous output failed validation:\n{error}\n\n"
                            f"Here is the input again:\n{prompt}\n\n"
                            "Fix the errors and output valid JSON only."
                        )
                        retry_msgs = await retry_child.prompt(retry_prompt)
                        result, _ = _try_parse_response(retry_msgs, vocabulary)
                        if result:
                            return result
                    finally:
                        with contextlib.suppress(Exception):
                            await retry_child.shutdown()
                return None
            finally:
                with contextlib.suppress(Exception):
                    await child.shutdown()
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                logger.warning("extraction child failed, retry {}/{}", attempt + 1, _MAX_RETRIES - 1)
                await asyncio.sleep(_RETRY_DELAY * (2**attempt))
            else:
                logger.exception("extraction child failed after all retries")
    return None


async def run_extraction(
    api: ExtensionAPI,
    messages: list[AgentMessage],
    *,
    registry: list[dict[str, Any]] | None = None,
    message_id_start: int = 0,
    vocabulary: str = "default",
    model: str | None = None,
) -> ExtractionResult | None:
    """Extraction entry point for use inside an atom (has ``ExtensionAPI``)."""
    config = build_extraction_config(
        cwd=api.cwd,
        model=model,
        vocabulary=vocabulary,
        parent_session_id=api.session_id,
    )
    prompt = build_extraction_prompt(
        messages, registry=registry, message_id_start=message_id_start,
    )
    return await run_extraction_session(
        config, prompt, spawn=api.spawn_child_session, vocabulary=vocabulary,
    )


# ---------------------------------------------------------------------------
# Populate index from extraction result
# ---------------------------------------------------------------------------


def _serialize_for_index(messages: list[AgentMessage]) -> list[dict[str, JsonValue]]:
    """Convert AgentMessage list to the dict format populate_from_extraction expects."""
    return [
        d for i, m in enumerate(messages)
        if (d := _agentmsg_to_extraction_dict(m, i))
    ]


def _populate_index(
    index: TrajectoryIndex,
    result: ExtractionResult,
    steps: list[Step],
    messages: list[dict[str, JsonValue]],
) -> None:
    # Steps are already added by the caller; delegate symbol + reference
    # population to the public method on TrajectoryIndex.
    # We pass steps_by_id so the index can resolve turn_id → Step.
    # Note: populate_from_extraction handles add_step internally when
    # called from eval; the atom pre-adds steps, so we call the lower-level
    # upsert + reference path directly.
    from .data import _build_references

    steps_by_id = {s.step_id: s for s in steps}

    for ext_sym in result.symbols:
        index.upsert_symbol(
            name=ext_sym.name,
            kind=ext_sym.kind.lower(),
            aliases=ext_sym.aliases,
            entity_class=ext_sym.entity_class,
        )

    refs, _rels = _build_references(index.registry_snapshot(), messages)
    for ref in refs:
        sym = index.resolve_symbol_by_name(ref.symbol_name)
        if not sym:
            continue
        step = steps_by_id.get(ref.turn_id)
        if not step:
            continue
        index.add_reference(
            symbol=sym,
            step=step,
            text=ref.text,
            kind=ref.kind,
            start=ref.start,
            confidence=0.8,
        )


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------


def _build_index_tool(api: ExtensionAPI, cfg: TrajectoryIndexConfig) -> FunctionTool:
    async def _handle(args: dict[str, JsonValue]) -> ToolResult:
        index = api.get_service(INDEX_SERVICE_KEY)
        assert isinstance(index, TrajectoryIndex)
        run_id = api.session_id

        all_messages = api.session.get_messages()
        if not all_messages:
            return ToolResult(content=[TextContent(type="text", text="No messages to index.")])

        watermark = index.indexed_message_count
        new_messages = all_messages[watermark:]
        if not new_messages:
            stats = index.stats(run_id)
            return ToolResult(content=[TextContent(
                type="text",
                text=(
                    f"No new messages. Index has "
                    f"{stats.symbol_count} symbols, "
                    f"{stats.reference_count} references, "
                    f"{stats.relation_count} relations."
                ),
            )])

        registry = index.registry_snapshot() if watermark > 0 else None
        result = await run_extraction(
            api,
            new_messages,
            registry=registry,
            message_id_start=watermark,
            vocabulary=cfg.vocabulary,
            model=cfg.model,
        )
        if result is None:
            return ToolResult(
                content=[TextContent(type="text", text="Extraction failed — check logs.")],
                is_error=True,
            )

        index.populate_from_extraction(
            result,
            _serialize_for_index(new_messages),
            run_id=run_id,
        )
        index.indexed_message_count = len(all_messages)

        # Model-judgment passes (best-effort — any model failure leaves the
        # deterministic layer intact). Each is an independent local judgment.
        merged = coref = 0
        model = cfg.resolve_model or cfg.model
        sf = api.spawn_child_session  # §11: atom passes its own factory
        if cfg.resolve_aliases:
            from .adjudicate import resolve_aliases, resolve_references

            try:
                groups = await resolve_aliases(index, model=model, apply=False, session_factory=sf)
                if groups:
                    index.apply_alias_merges(groups)
                    merged = sum(len(g) for g in groups) - len(groups)
                coref = await resolve_references(index, model=model, apply=False, session_factory=sf)
            except Exception:
                logger.warning("Pass 2 (alias/coref) failed, degrading to Pass 1+3", exc_info=True)

        # Pass 3 (dataflow): def-use + grounding over the full run (deterministic).
        index.build_dependencies()

        # Pass 3.5 — value fidelity (independent of alias resolution).
        try:
            from .adjudicate import compare_values

            await compare_values(index, model=model, apply=True, session_factory=sf)
        except Exception:
            logger.warning("Pass 3.5 (value fidelity) failed, skipping", exc_info=True)

        stats = index.stats(run_id)
        deps = index.get_dependencies()
        ungrounded = sum(1 for d in deps if d.risk == "ungrounded")
        contradicted = sum(1 for d in deps if d.risk == "contradicted")
        merged_note = f", merged {merged} aliases" if merged else ""
        coref_note = f", resolved {coref} anaphors" if coref else ""
        flags = []
        if ungrounded:
            flags.append(f"{ungrounded} fabricated-name")
        if contradicted:
            flags.append(f"{contradicted} wrong-value")
        flag_note = f", {' + '.join(flags)} candidate(s)" if flags else ""
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Indexed {len(new_messages)} new messages "
                        f"(total {stats.step_count} steps) -> "
                        f"{stats.symbol_count} symbols, "
                        f"{stats.reference_count} references, "
                        f"{stats.relation_count} relations, "
                        f"{stats.dependency_count} dependencies"
                        f"{merged_note}{coref_note}{flag_note}."
                    ),
                )
            ]
        )

    return FunctionTool(
        name="index_trajectory",
        description=(
            "Build or update the semantic index over the current session's "
            "trajectory. Incrementally extracts symbols, references, and relations "
            "from new messages (Pass 1), resolves same-entity aliases (Pass 2), and "
            "builds the def-use / grounding layer that flags fabricated names — "
            "structured identifiers the model used but no tool ever produced (Pass 3)."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        fn=_handle,
    )


def _build_search_tool(api: ExtensionAPI) -> FunctionTool:
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


def _build_context_tool(api: ExtensionAPI) -> FunctionTool:
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


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def install(api: ExtensionAPI, config: TrajectoryIndexConfig) -> None:
    index = TrajectoryIndex()
    api.set_service(INDEX_SERVICE_KEY, index)

    api.register_tool(_build_index_tool(api, config))
    api.register_tool(_build_search_tool(api))
    api.register_tool(_build_context_tool(api))
