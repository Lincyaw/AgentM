"""Trajectory semantic index atom.

Registers tools for the main agent to build and query a semantic index
over its own trajectory. Extraction is delegated to a child agent
(entity_extractor scenario) that runs on a small/fast model.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Final

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    ExtensionAPI,
    FunctionTool,
    LoopConfig,
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
from .ir.index import TrajectoryIndex
from .pass1_nodes.serialize import JsonValue
from .query_tools import INDEX_SERVICE_KEY, register_query_tools

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


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
    value_fidelity: bool = Field(
        default=True,
        description="Run Pass 3.5 value fidelity check: for each def-use edge "
        "with a grounded binding, ask the model whether the agent's usage "
        "matches what the tool provided. Flags 'contradicted' risk. One LLM "
        "call over all targets.",
    )
    analyze_claims: bool = Field(
        default=False,
        description="Run the claim/constraint adjudication passes after def-use: "
        "claim↔observation edges + status fold (supported/unsourced/conflicted) "
        "and task-constraint satisfaction (verified/violated/omitted). Off by "
        "default because each is a per-claim oracle sweep (LLM cost scales with "
        "claim count); enable for post-hoc audit indexing where the checks are "
        "consumed. Best-effort: any failure leaves the deterministic layers intact.",
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
        "tool:get_insights",
    ),
    config_schema=TrajectoryIndexConfig,
)


# ---------------------------------------------------------------------------
# AgentMessage → extraction/serialization shapes
# ---------------------------------------------------------------------------


def _agentmsg_to_payload(msg: AgentMessage) -> dict[str, JsonValue]:
    """Serialize an AgentMessage to the trace payload dict shape."""
    blocks: list[JsonValue] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCallBlock):
            tc: dict[str, JsonValue] = {
                "type": "tool_call",
                "name": block.name,
                "arguments": block.arguments,
            }
            if block.id:
                tc["id"] = block.id
            blocks.append(tc)
        elif isinstance(block, ToolResultBlock):
            sub_blocks: list[JsonValue] = []
            for sub in block.content:
                if isinstance(sub, TextContent):
                    sub_blocks.append({"type": "text", "text": sub.text})
            tr: dict[str, JsonValue] = {"type": "tool_result", "content": sub_blocks}
            if block.tool_call_id:
                tr["tool_call_id"] = block.tool_call_id
            if getattr(block, "is_error", False):
                tr["is_error"] = True
            if not block.deterministic:
                tr["deterministic"] = False
            blocks.append(tr)
    return {"role": msg.role, "content": blocks}


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
    from .pass1_nodes.serialize import _try_parse_response as _parse

    return _parse(messages, vocabulary)


def build_extraction_config(
    *,
    cwd: str,
    model: str | None = None,
    provider: tuple[str, dict[str, Any]] | None = None,
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
    payload = _agentmsg_to_payload(msg)
    role = payload.get("role", "")
    content = payload.get("content", [])
    if not isinstance(content, list) or not content:
        return {}
    blocks: list[JsonValue] = [b for b in content if isinstance(b, dict)]
    return {"id": str(index), "role": role, "content": blocks}


def _format_message_compact(
    msg: dict[str, Any],
    registry: list[dict[str, Any]] | None = None,
) -> str:
    """Format one serialized message as compact text for the extraction prompt.

    Previously-extracted symbols are marked inline with their original tag
    (e.g. ``⟦sym kind=file|codec.py⟧``) so the model sees what was already
    found, in context.
    """
    from .pass1_nodes.serialize import view_body_with_map

    mid = msg.get("id", "")
    role = msg.get("role", "")
    body, _ = view_body_with_map(msg)
    if registry:
        body = _mark_extracted(body, registry)
    return f"[{mid}|{role}]\n{body}"


def _mark_extracted(text: str, registry: list[dict[str, Any]]) -> str:
    """Wrap occurrences of previously-extracted symbols with their original tag.

    ``⟦sym kind=file|codec.py⟧`` — not a generic ``⟦known|…⟧``.
    """
    import re

    from .pass1_nodes.markup import CLOSE, OPEN

    names: list[tuple[str, str]] = []
    for entry in registry:
        kind = str(entry.get("kind", "unknown"))
        name = str(entry.get("name", ""))
        if name:
            names.append((name, kind))
        for alias in entry.get("aliases", []):
            alias = str(alias)
            if alias:
                names.append((alias, kind))
    names.sort(key=lambda x: -len(x[0]))

    for name, kind in names:
        tag = f"{OPEN}sym kind={kind}|"
        escaped = re.escape(name)
        boundary = r"a-zA-Z0-9_.\-" + ("/" if "/" in name else "")
        pattern = re.compile(rf"(?<!\|)(?<![{boundary}]){escaped}(?![{boundary}])", re.IGNORECASE)
        text = pattern.sub(f"{tag}{name}{CLOSE}", text)
    return text


def build_extraction_prompt(
    messages: list[AgentMessage],
    *,
    registry: list[dict[str, Any]] | None = None,
    message_id_start: int = 0,
) -> str:
    """Build the extractor's input prompt from trajectory messages.

    Previously-extracted symbols are marked inline with their original tag
    (e.g. ``⟦sym kind=file|codec.py⟧``).
    """
    serialized = [
        d for i, m in enumerate(messages, start=message_id_start)
        if (d := _agentmsg_to_extraction_dict(m, i))
    ]
    formatted = "\n\n".join(
        text for msg in serialized
        if (text := _format_message_compact(msg, registry))
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

    ``spawn`` creates a child session from a config. Required (§11: this
    module never imports ``agentm.core.runtime``). The atom passes
    ``api.spawn_child_session``; offline callers pass ``AgentSession.create``
    imported on their own side — same convention as ``adjudicate._ask_model``.
    """
    if spawn is None:
        raise ValueError("spawn is required (pass AgentSession.create for offline use)")

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


def _first_user_text(messages: list[AgentMessage]) -> str:
    """The first user message's text — the task question for constraint analysis.

    Constraint nodes come from Pass 1's extraction of the task text; the
    question only seeds commit detection, so a best-effort join is enough.
    """
    for m in messages:
        if m.role == "user":
            text = " ".join(b.text for b in m.content if isinstance(b, TextContent)).strip()
            if text:
                return text
    return ""


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
        merged = 0
        model = cfg.resolve_model or cfg.model
        sf = api.spawn_child_session  # §11: atom passes its own factory
        if cfg.resolve_aliases:
            from .pass2_edges.identity import resolve_aliases

            try:
                groups = await resolve_aliases(index, model=model, apply=False, session_factory=sf)
                if groups:
                    index.apply_alias_merges(groups)
                    merged = sum(len(g) for g in groups) - len(groups)
            except Exception:
                logger.warning("Pass 2 (alias) failed, degrading to Pass 1+3", exc_info=True)

        # Pass 3 (dataflow): def-use + grounding over the full run (deterministic).
        index.build_dependencies()

        # Pass 3.5 — value fidelity (independent of alias resolution).
        if cfg.value_fidelity:
            try:
                from .pass3_folds.grounding import compare_values

                await compare_values(index, model=model, apply=True, session_factory=sf)
            except Exception:
                logger.warning("Pass 3.5 (value fidelity) failed, skipping", exc_info=True)

        # Pass 4 — claim/constraint adjudication (opt-in; per-claim oracle sweep).
        # Wholesale-replaces this run's edges/claim_findings/constraint_findings,
        # so it reasons over the full run each call, not just the new delta.
        #
        # Pipeline: commit(1 LLM) → unified evidence(P LLM) → fold(code) → intent(1 LLM)
        # The unified evidence sweep merges claim and constraint evidence into
        # one call per partition (was 2P before the merge).
        if cfg.analyze_claims:
            from .pass2_edges.claims import build_claim_edges
            from .pass2_edges.intent_alignment import build_intent_alignment_edges
            from .pass3_folds.claim_status import fold_claim_statuses
            from .pass3_folds.constraints import analyze_constraints

            constraints = list(index.constraints.values())
            question = _first_user_text(all_messages)

            # E1: commit detection — runs only when constraints exist;
            # the binding and step_id feed the merged evidence sweep so
            # the model knows the candidate answer.
            _commit_obj = None
            commit_binding = ""
            commit_step_id_str = ""
            precomputed: dict[str, Any] | None = None
            if constraints:
                from .ir.diagnostics import Diagnostics as _Diag
                from .pass3_folds.constraints import _detect_commit

                _diag = _Diag()
                _all_steps = sorted(
                    (s for s in index.steps.values()
                     if not run_id or s.run_id == run_id),
                    key=lambda s: s.index,
                )
                try:
                    _commit_obj = await _detect_commit(
                        index, _all_steps, question=question,
                        model=model, session_factory=sf, diag=_diag,
                    )
                    if _commit_obj is not None:
                        commit_binding = _commit_obj.binding
                        commit_step_id_str = _commit_obj.step.step_id
                except Exception:
                    logger.warning("Pass 4 (commit detection) failed, skipping constraints", exc_info=True)
                    constraints = []

            # Unified evidence sweep — claims + constraints in one partition sweep
            try:
                edge_result = await build_claim_edges(
                    index, run_id=run_id, model=model, session_factory=sf,
                    constraints=constraints if commit_binding else None,
                    commit_binding=commit_binding,
                    commit_step_id=commit_step_id_str,
                )
                fold_claim_statuses(index, edge_result, run_id=run_id)
                if edge_result.constraint_results:
                    precomputed = edge_result.constraint_results
            except Exception:
                logger.warning("Pass 4 (evidence/status) failed, skipping", exc_info=True)

            # Constraint findings — uses pre-computed commit + evidence
            if constraints:
                try:
                    await analyze_constraints(
                        index, run_id=run_id, question=question,
                        model=model, session_factory=sf,
                        precomputed_evidence=precomputed,
                        precomputed_commit=_commit_obj,
                    )
                except Exception:
                    logger.warning("Pass 4 (constraint analysis) failed, skipping", exc_info=True)

            try:
                await build_intent_alignment_edges(
                    index, run_id=run_id, model=model, session_factory=sf,
                )
            except Exception:
                logger.warning("Pass 4 (intent alignment) failed, skipping", exc_info=True)

        stats = index.stats(run_id)
        deps = index.get_dependencies()
        ungrounded = sum(1 for d in deps if d.risk == "ungrounded")
        contradicted = sum(1 for d in deps if d.risk == "contradicted")
        merged_note = f", merged {merged} aliases" if merged else ""
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
                        f"{merged_note}{flag_note}."
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



# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def install(api: ExtensionAPI, config: TrajectoryIndexConfig) -> None:
    index = TrajectoryIndex()
    api.set_service(INDEX_SERVICE_KEY, index)

    api.register_tool(_build_index_tool(api, config))
    register_query_tools(api)  # search_symbols / get_symbol_context / get_insights
