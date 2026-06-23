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
    AssistantMessage,
    ExtensionAPI,
    FunctionTool,
    LoopConfig,
    TextContent,
    ToolCallBlock,
    ToolResult,
)
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, Field

from .agents import extractor_scenario
from .agents.entity_extractor.schema import ReportEntitiesParams
from .index import (
    Entity,
    EntityKind,
    MentionType,
    RelationType,
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


MANIFEST = ExtensionManifest(
    name="trajectory_index",
    description=(
        "Build and query a semantic entity-relation index over the current "
        "session trajectory using a small extraction model."
    ),
    registers=(
        "tool:index_trajectory",
        "tool:search_entities",
        "tool:get_entity_context",
    ),
    config_schema=TrajectoryIndexConfig,
)


# ---------------------------------------------------------------------------
# Provider resolution (same pattern as llmharness)
# ---------------------------------------------------------------------------


def _resolve_provider(model_name: str) -> tuple[str, dict[str, Any]] | None:
    from agentm.ai import DEFAULT_PROVIDER_DESCRIPTORS
    from agentm.core.lib import resolve_model_profile

    profile = resolve_model_profile(model_name)
    if profile is None:
        logger.warning(f"model profile {model_name!r} not found in config.toml")
        return None
    for desc in DEFAULT_PROVIDER_DESCRIPTORS:
        if desc.id == profile.provider and desc.extension_module is not None:
            return (desc.extension_module, dict(profile.to_build_config()))
    logger.warning(f"provider {profile.provider!r} has no extension module")
    return None


# ---------------------------------------------------------------------------
# Message → Step conversion
# ---------------------------------------------------------------------------


def _messages_to_steps(
    messages: list[AgentMessage],
    run_id: str,
) -> list[Step]:
    steps: list[Step] = []
    for i, msg in enumerate(messages):
        role = _ROLE_MAP.get(msg.role, StepRole.USER)
        parts: list[str] = []
        tool_name: str | None = None

        for block in msg.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ToolCallBlock):
                tool_name = block.name
                parts.append(f"[tool_call: {block.name}({json.dumps(block.arguments, ensure_ascii=False, default=str)[:500]})]")
            elif hasattr(block, "text"):
                parts.append(str(block.text))
            elif hasattr(block, "content"):
                for sub in block.content:
                    if hasattr(sub, "text"):
                        parts.append(str(sub.text))

        content = "\n".join(parts)
        if not content.strip():
            continue

        steps.append(Step(
            run_id=run_id,
            step_id=f"s{i}",
            index=i,
            role=role,
            content=content,
            tool_name=tool_name,
            timestamp=msg.timestamp if hasattr(msg, "timestamp") else None,
        ))
    return steps


# ---------------------------------------------------------------------------
# Child session orchestration
# ---------------------------------------------------------------------------


_MAX_RETRIES: Final = 3
_RETRY_DELAY: Final = 5.0

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)


def _extract_json_from_text(messages: list[AgentMessage]) -> dict[str, Any] | None:
    """Extract a JSON object from the last assistant text response."""
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if not isinstance(block, TextContent):
                continue
            text = block.text.strip()
            # Try the raw text as JSON first
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            # Try extracting from ```json ... ``` fences
            m = _JSON_BLOCK_RE.search(text)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
            # Try finding the first { ... } substring
            start = text.find("{")
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(text[start : i + 1])
                                if isinstance(obj, dict):
                                    return obj
                            except json.JSONDecodeError:
                                pass
                            break
    return None


async def _run_extraction(
    api: ExtensionAPI,
    steps: list[Step],
    provider: tuple[str, dict[str, Any]] | None,
) -> ReportEntitiesParams | None:
    steps_payload = [
        {
            "index": s.index,
            "role": s.role.value,
            "content": s.content[:2000],
            "tool_name": s.tool_name,
        }
        for s in steps
    ]
    prompt = json.dumps(steps_payload, ensure_ascii=False, indent=2)

    scenario = extractor_scenario()
    config = AgentSessionConfig(
        cwd=api.cwd,
        provider=provider,
        scenario=scenario,
        purpose="trajectory_entity_extractor",
        loop_config=LoopConfig(max_turns=1),
        lineage={
            "kind": "trajectory_index_extraction",
            "parent_session_id": api.session_id,
        },
    )

    for attempt in range(_MAX_RETRIES):
        try:
            child = await api.spawn_child_session(config)
            try:
                child_msgs: list[AgentMessage] = await child.prompt(prompt)
                obj = _extract_json_from_text(child_msgs)
                if obj:
                    return ReportEntitiesParams.model_validate(obj)
                logger.warning("extraction child returned no parseable JSON")
                return None
            finally:
                with contextlib.suppress(Exception):
                    await child.shutdown()
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                logger.warning(
                    f"extraction child failed, retry {attempt + 1}/{_MAX_RETRIES - 1}"
                )
                await asyncio.sleep(_RETRY_DELAY * (2 ** attempt))
            else:
                logger.exception("extraction child failed after all retries")
    return None


# ---------------------------------------------------------------------------
# Populate index from extraction result
# ---------------------------------------------------------------------------


def _kind_from_str(s: str) -> EntityKind:
    try:
        return EntityKind(s.lower())
    except ValueError:
        return EntityKind.UNKNOWN


def _mention_type_from_str(s: str) -> MentionType:
    try:
        return MentionType(s.lower())
    except ValueError:
        return MentionType.UNKNOWN


def _relation_type_from_str(s: str) -> RelationType:
    try:
        return RelationType(s.lower())
    except ValueError:
        return RelationType.CO_MENTIONED


def _populate_index(
    index: TrajectoryIndex,
    result: ReportEntitiesParams,
    steps: list[Step],
) -> None:
    steps_by_index = {s.index: s for s in steps}

    entity_map: dict[str, Entity] = {}
    for ext_ent in result.entities:
        entity = index.upsert_entity(
            name=ext_ent.name,
            kind=_kind_from_str(ext_ent.kind),
            summary=ext_ent.summary,
            aliases=ext_ent.aliases,
        )
        entity_map[ext_ent.name] = entity

    for ext_men in result.mentions:
        ent = entity_map.get(ext_men.entity_name)
        if not ent:
            continue
        step = steps_by_index.get(ext_men.step_index)
        if not step:
            continue
        start = step.content.find(ext_men.text)
        if start < 0:
            start = 0
        index.add_mention(
            entity=ent,
            step=step,
            text=ext_men.text,
            mention_type=_mention_type_from_str(ext_men.mention_type),
            start=start,
            end=start + len(ext_men.text) if start > 0 else len(ext_men.text),
            confidence=0.9,
        )

    for ext_rel in result.relations:
        from_ent = entity_map.get(ext_rel.from_entity)
        to_ent = entity_map.get(ext_rel.to_entity)
        if not from_ent or not to_ent:
            continue
        step = steps_by_index.get(ext_rel.step_index)
        if not step:
            continue
        index.add_relation(
            from_entity=from_ent,
            to_entity=to_ent,
            rel_type=_relation_type_from_str(ext_rel.relation_type),
            step=step,
            confidence=0.85,
        )


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------


def _build_index_tool(api: ExtensionAPI, cfg: TrajectoryIndexConfig) -> FunctionTool:
    async def _handle(args: dict[str, Any]) -> ToolResult:
        index = api.get_service(INDEX_SERVICE_KEY)
        assert isinstance(index, TrajectoryIndex)
        messages = api.session.get_messages()
        run_id = api.session_id

        steps = _messages_to_steps(messages, run_id)
        if not steps:
            return ToolResult(content=[TextContent(type="text", text="No messages to index.")])

        for step in steps:
            index.add_step(step)

        provider = _resolve_provider(cfg.model)
        result = await _run_extraction(api, steps, provider)
        if result is None:
            return ToolResult(
                content=[TextContent(type="text", text="Extraction failed — check logs.")],
                is_error=True,
            )

        _populate_index(index, result, steps)
        stats = index.stats(run_id)
        return ToolResult(content=[TextContent(
            type="text",
            text=(
                f"Indexed {stats.step_count} steps -> "
                f"{stats.entity_count} entities, "
                f"{stats.mention_count} mentions, "
                f"{stats.relation_count} relations."
            ),
        )])

    return FunctionTool(
        name="index_trajectory",
        description=(
            "Build a semantic index over the current session's trajectory. "
            "Extracts entities, mentions, and relations from all messages."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        fn=_handle,
    )


def _build_search_tool(api: ExtensionAPI) -> FunctionTool:
    class SearchParams(BaseModel):
        query: str = Field(description="Search query (entity name, concept, or keyword)")
        kinds: list[str] | None = Field(default=None, description="Filter by entity kinds")
        limit: int = Field(default=10, description="Max results")

    async def _handle(args: dict[str, Any]) -> ToolResult:
        params = SearchParams.model_validate(args)
        index = api.get_service(INDEX_SERVICE_KEY)
        assert isinstance(index, TrajectoryIndex)

        kind_filter = None
        if params.kinds:
            kind_filter = {_kind_from_str(k) for k in params.kinds}

        results = index.search(
            params.query, kinds=kind_filter, limit=params.limit,
            include_mentions=True, include_related=True,
        )
        if not results:
            return ToolResult(content=[TextContent(type="text", text="No entities found.")])

        lines: list[str] = []
        for r in results:
            ent = r.entity
            lines.append(
                f"- **{ent.canonical_name}** ({ent.kind.value}, score={r.score:.2f})"
                f"  id={ent.id}"
            )
            if ent.summary:
                lines.append(f"  {ent.summary}")
            if r.mentions:
                refs = ", ".join(
                    f"step {m.step_id}:{m.mention_type.value}" for m in r.mentions[:3]
                )
                lines.append(f"  mentions: {refs}")
            if r.related:
                rels = ", ".join(
                    f"{rel.entity.canonical_name}({rel.score:.2f})" for rel in r.related[:3]
                )
                lines.append(f"  related: {rels}")

        return ToolResult(content=[TextContent(type="text", text="\n".join(lines))])

    return FunctionTool(
        name="search_entities",
        description="Search for entities in the trajectory semantic index.",
        parameters=SearchParams,
        fn=_handle,
    )


def _build_context_tool(api: ExtensionAPI) -> FunctionTool:
    class ContextParams(BaseModel):
        entity_id: str = Field(description="Entity ID to get context for")

    async def _handle(args: dict[str, Any]) -> ToolResult:
        params = ContextParams.model_validate(args)
        index = api.get_service(INDEX_SERVICE_KEY)
        assert isinstance(index, TrajectoryIndex)

        try:
            ctx = index.get_context(params.entity_id)
        except KeyError:
            return ToolResult(
                content=[TextContent(type="text", text=f"Entity not found: {params.entity_id}")],
                is_error=True,
            )

        lines: list[str] = []
        ent = ctx.entity
        lines.append(f"# {ent.canonical_name} ({ent.kind.value})")
        if ent.summary:
            lines.append(f"\n{ent.summary}")
        if ent.aliases:
            lines.append(f"\nAliases: {', '.join(sorted(ent.aliases))}")

        if ctx.definition:
            d = ctx.definition
            lines.append(f"\n## Definition\nStep {d.step_id}: \"{d.text}\" ({d.mention_type.value})")

        if ctx.timeline:
            lines.append("\n## Timeline")
            for item in ctx.timeline[:15]:
                lines.append(
                    f"- [{item.step.role.value}] step {item.step.step_id}: "
                    f"\"{item.mention.text}\" ({item.mention.mention_type.value})"
                )

        if ctx.related:
            lines.append("\n## Related entities")
            for rel in ctx.related[:10]:
                rel_types = ", ".join(r.type.value for r in rel.relations[:3])
                lines.append(
                    f"- {rel.entity.canonical_name} ({rel.entity.kind.value}) "
                    f"— {rel_types} (score={rel.score:.2f})"
                )

        return ToolResult(content=[TextContent(type="text", text="\n".join(lines))])

    return FunctionTool(
        name="get_entity_context",
        description=(
            "Get full context for a specific entity: definition, timeline, "
            "related entities, and surrounding trajectory snippets."
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
