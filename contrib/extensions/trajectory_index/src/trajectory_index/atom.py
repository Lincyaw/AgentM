"""Trajectory semantic index atom.

Registers tools for the main agent to build and query a semantic index
over its own trajectory. Extraction is delegated to a child agent
(entity_extractor scenario) that runs on a small/fast model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Final

from agentm.core.abi import (
    AgentMessage,
    ExtensionAPI,
    FunctionTool,
    LoopConfig,
    SessionEntry,
    TextContent,
    ToolCallBlock,
    ToolResult,
    ToolResultBlock,
)
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, Field

from .agents import extractor_scenario
from .agents.entity_extractor.schema import ExtractionResult
from .data import JsonValue, ProviderSpec, resolve_provider
from .index import (
    Step,
    StepRole,
    Symbol,
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
# Provider resolution
# ---------------------------------------------------------------------------


def _resolve_provider_safe(model_name: str) -> ProviderSpec | None:
    try:
        return resolve_provider(model_name)
    except RuntimeError:
        logger.warning(f"could not resolve model profile {model_name!r}")
        return None


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
            blocks.append({"type": "tool_result", "content": sub_blocks})
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


async def run_extraction(
    api: ExtensionAPI,
    messages: list[dict[str, JsonValue]],
    provider: ProviderSpec | None,
    registry: list[dict[str, Any]] | None = None,
    message_id_start: int = 0,
    vocabulary: str = "default",
) -> ExtractionResult | None:
    """Public extraction entry point: index a trajectory via a child session.

    Consumed by this atom's own turn-end handler and by external harnesses
    (e.g. llmharness) that want a one-shot extraction over a serialized
    trajectory. Vocabulary selection is explicit — no module globals or
    environment variables.
    """
    from .data import _reindex_messages

    reindexed = _reindex_messages(messages, start=message_id_start)

    if registry:
        prompt_data: dict[str, Any] = {
            "known_symbols": registry,
            "messages": reindexed,
        }
        prompt = json.dumps(prompt_data, ensure_ascii=False, indent=2)
    else:
        prompt = json.dumps(reindexed, ensure_ascii=False, indent=2)

    scenario = extractor_scenario()
    config = AgentSessionConfig(
        cwd=api.cwd,
        provider=provider,
        scenario=scenario,
        purpose="trajectory_symbol_extractor",
        loop_config=LoopConfig(max_turns=1),
        lineage={
            "kind": "trajectory_index_extraction",
            "parent_session_id": api.session_id,
        },
        atom_config_overrides={
            "trajectory_extractor_context": {"vocabulary": vocabulary},
        },
    )

    for attempt in range(_MAX_RETRIES):
        try:
            child = await api.spawn_child_session(config)
            try:
                child_msgs: list[AgentMessage] = await child.prompt(prompt)
                result, error = _try_parse_response(child_msgs, vocabulary)
                if result:
                    return result
                if error:
                    logger.warning(f"extraction parse failed: {error}")
                    retry_child = await api.spawn_child_session(config)
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
                logger.warning(f"extraction child failed, retry {attempt + 1}/{_MAX_RETRIES - 1}")
                await asyncio.sleep(_RETRY_DELAY * (2**attempt))
            else:
                logger.exception("extraction child failed after all retries")
    return None


# ---------------------------------------------------------------------------
# Populate index from extraction result
# ---------------------------------------------------------------------------


def _populate_index(
    index: TrajectoryIndex,
    result: ExtractionResult,
    steps: list[Step],
    messages: list[dict[str, JsonValue]],
) -> None:
    from .data import _build_references

    steps_by_id = {s.step_id: s for s in steps}

    for ext_sym in result.symbols:
        index.upsert_symbol(
            name=ext_sym.name,
            kind=ext_sym.kind.lower(),
            summary=ext_sym.summary,
            aliases=ext_sym.aliases,
        )

    def _resolve(name: str) -> Symbol | None:
        return index.resolve_symbol_by_name(name)

    refs, _rels = _build_references(index.registry_snapshot(), messages)
    for ref in refs:
        sym = _resolve(ref.symbol_name)
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

        branch = api.session.get_branch()
        clean_msgs = _branch_to_clean_messages(branch)
        if not clean_msgs:
            return ToolResult(content=[TextContent(type="text", text="No messages to index.")])

        watermark = index.indexed_message_count
        new_msgs = clean_msgs[watermark:]
        if not new_msgs:
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

        from .data import _reindex_messages

        prompt_msgs = _reindex_messages(new_msgs, start=watermark)
        new_steps = _clean_messages_to_steps(prompt_msgs, run_id, start_index=watermark)
        for step in new_steps:
            index.add_step(step)

        registry = index.registry_snapshot() if watermark > 0 else None
        provider = _resolve_provider_safe(cfg.model)
        result = await run_extraction(
            api,
            new_msgs,
            provider,
            registry=registry,
            message_id_start=watermark,
            vocabulary=cfg.vocabulary,
        )
        if result is None:
            return ToolResult(
                content=[TextContent(type="text", text="Extraction failed — check logs.")],
                is_error=True,
            )

        _populate_index(index, result, new_steps, prompt_msgs)
        index.indexed_message_count = len(clean_msgs)
        stats = index.stats(run_id)
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Indexed {len(new_msgs)} new messages "
                        f"(total {stats.step_count} steps) -> "
                        f"{stats.symbol_count} symbols, "
                        f"{stats.reference_count} references, "
                        f"{stats.relation_count} relations."
                    ),
                )
            ]
        )

    return FunctionTool(
        name="index_trajectory",
        description=(
            "Build or update the semantic index over the current session's "
            "trajectory. Incrementally extracts symbols, references, and "
            "relations from new messages since the last indexing."
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
