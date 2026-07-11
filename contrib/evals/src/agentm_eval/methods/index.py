"""Trajectory-index method adapter.

Single entry point for running trajectory_index extraction offline.
Delegates to the atom's composable API and the library's public
``TrajectoryIndex.populate_from_extraction``.

Usage::

    from agentm_eval.methods.index import extract_symbols, build_index

    # Whole-trajectory extraction
    chunks = await extract_symbols(messages, model="azure-gpt")

    # Chunked incremental extraction
    chunks = await extract_symbols(messages, model="azure-gpt", chunk_size=(2, 5))

    # Build a TrajectoryIndex from chunks (runs all 3 passes)
    index = await build_index(chunks)
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from agentm.core.abi import AgentMessage
from loguru import logger

from trajectory_index.agents.entity_extractor.schema import ExtractionResult
from trajectory_index.data import _symbol_aliases


# ---------------------------------------------------------------------------
# Core extraction — delegates to atom API
# ---------------------------------------------------------------------------


def _prescan_structural(
    messages: list[AgentMessage],
    registry: list[dict[str, Any]] | None,
    seen: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Code-level structural extraction before LLM.

    Scans tool_call blocks for tool names and SQL for table names.
    Returns (updated_registry, structural_symbols) where structural_symbols
    are ExtractedSymbol-like objects to merge into the final result.
    """
    from trajectory_index.atom import _agentmsg_to_extraction_dict
    from trajectory_index.data import extract_structural_symbols
    from trajectory_index.index import normalize_name

    serialized = [
        d for i, m in enumerate(messages)
        if (d := _agentmsg_to_extraction_dict(m, i))
    ]
    structural = extract_structural_symbols(serialized)
    if not structural:
        return (list(registry) if registry else [], [])

    if seen is None:
        seen = set()
        if registry:
            for entry in registry:
                seen.add(normalize_name(str(entry.get("name", ""))))

    updated = list(registry) if registry else []
    new_syms = []
    for sym in structural:
        norm = normalize_name(sym.name)
        if norm not in seen:
            seen.add(norm)
            updated.append({"name": sym.name, "kind": sym.kind})
            new_syms.append(sym)

    return updated, new_syms


async def _run_one(
    messages: list[AgentMessage],
    *,
    model: str | None,
    vocabulary: str,
    registry: list[dict[str, Any]] | None,
    message_id_start: int,
    cwd: str | None,
) -> ExtractionResult | None:
    from trajectory_index.atom import (
        build_extraction_config,
        build_extraction_prompt,
        run_extraction_session,
    )

    config = build_extraction_config(
        cwd=cwd or os.getcwd(),
        model=model,
        vocabulary=vocabulary,
    )
    prompt = build_extraction_prompt(
        messages,
        registry=registry,
        message_id_start=message_id_start,
    )
    return await run_extraction_session(config, prompt, vocabulary=vocabulary)


# ---------------------------------------------------------------------------
# Chunking (operates on serialized dicts internally)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MessageChunk:
    start: int
    messages: list[AgentMessage]


def _chunk_messages(
    messages: list[AgentMessage],
    size_range: tuple[int, int],
) -> list[_MessageChunk]:
    import random

    from agentm.core.abi import ToolResultMessage

    lo, hi = size_range
    if len(messages) <= lo:
        return [_MessageChunk(start=0, messages=messages)]

    chunks: list[_MessageChunk] = []
    start = 0
    while start < len(messages):
        chunk_size = random.randint(lo, hi)
        end = start + chunk_size
        if end >= len(messages):
            chunks.append(_MessageChunk(start=start, messages=messages[start:]))
            break
        while end < len(messages) and isinstance(messages[end], ToolResultMessage):
            end += 1
        chunks.append(_MessageChunk(start=start, messages=messages[start:end]))
        start = end
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractedChunk:
    run_id: str
    messages: list[AgentMessage]
    result: ExtractionResult


type OnChunkCallback = Callable[[ExtractedChunk, list[AgentMessage]], None]


async def extract_symbols(
    messages: list[AgentMessage],
    *,
    model: str | None = None,
    vocabulary: str = "default",
    chunk_size: tuple[int, int] | None = None,
    on_chunk: OnChunkCallback | None = None,
    run_id: str = "",
    cwd: str | None = None,
) -> list[ExtractedChunk]:
    """Extract symbols from trajectory messages.

    Accepts typed ``AgentMessage`` objects. When ``chunk_size`` is None,
    runs a single extraction. When set (e.g. ``(2, 5)``), splits into
    chunks with registry accumulation.
    """
    from trajectory_index.index import normalize_name

    if chunk_size is None:
        result = await _run_one(
            messages, model=model, vocabulary=vocabulary,
            registry=None, message_id_start=0, cwd=cwd,
        )
        if result is None:
            return []
        extracted = ExtractedChunk(run_id=run_id, messages=messages, result=result)
        if on_chunk:
            on_chunk(extracted, messages)
        return [extracted]

    chunks = _chunk_messages(messages, chunk_size)
    registry: list[dict[str, Any]] = []
    seen: set[str] = set()
    results: list[ExtractedChunk] = []

    for i, chunk in enumerate(chunks):
        # Code-level prescan: extract tool names, SQL tables, etc.
        chunk_registry, structural = _prescan_structural(
            chunk.messages, registry if registry else None, seen,
        )
        if structural:
            logger.info(
                "chunk {}/{}: prescan found {} structural symbols",
                i + 1, len(chunks), len(structural),
            )

        try:
            result = await _run_one(
                chunk.messages, model=model, vocabulary=vocabulary,
                registry=chunk_registry if chunk_registry else None,
                message_id_start=chunk.start, cwd=cwd,
            )
        except Exception:
            logger.exception("chunk {}/{} extraction failed", i + 1, len(chunks))
            continue
        if result is None:
            logger.warning("chunk {}/{} no parseable result", i + 1, len(chunks))
            continue

        # Merge structural symbols into LLM result (deduped)
        if structural:
            from trajectory_index.agents.entity_extractor.schema import ExtractedSymbol
            llm_names = {normalize_name(s.name) for s in result.symbols}
            for ss in structural:
                if normalize_name(ss.name) not in llm_names:
                    result.symbols.append(ExtractedSymbol(
                        name=ss.name, kind=ss.kind, entity_class="identifier",
                    ))

        extracted = ExtractedChunk(run_id=run_id, messages=chunk.messages, result=result)
        results.append(extracted)

        if on_chunk:
            on_chunk(extracted, chunk.messages)

        for sym in result.symbols:
            norm = normalize_name(sym.name)
            if norm not in seen:
                seen.add(norm)
                entry: dict[str, Any] = {"name": sym.name, "kind": sym.kind}
                if sym.summary:
                    entry["summary"] = sym.summary
                if sym.aliases:
                    entry["aliases"] = sym.aliases
                registry.append(entry)

    return results


# ---------------------------------------------------------------------------
# Index building — delegates to TrajectoryIndex.populate_from_extraction
# ---------------------------------------------------------------------------

_SPAN_SYMBOL_RE: Final = re.compile(r"^(?:span\s+)?s\d+$", re.IGNORECASE)


def _span_namespace(run_id: str, sym: dict[str, Any]) -> str:
    """Namespace function for span-ID symbols (telbench-style)."""
    if not run_id or str(sym.get("kind", "")).lower() != "file":
        return ""
    names = [str(sym.get("name", "")), *_symbol_aliases(sym)]
    return run_id if any(_SPAN_SYMBOL_RE.match(str(name).strip()) for name in names) else ""


def _to_index_dicts(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Serialize AgentMessage list to the dict format populate_from_extraction expects."""
    from trajectory_index.atom import _agentmsg_to_extraction_dict

    return [
        d for i, m in enumerate(messages)
        if (d := _agentmsg_to_extraction_dict(m, i))
    ]


async def resolve_index(
    index: Any,
    *,
    model: str | None = None,
) -> None:
    """Run Pass 2 (alias + coreference) and Pass 3 + 3.5 (dataflow + value fidelity).

    Best-effort: any LLM pass that fails is skipped; the deterministic
    layers remain intact.
    """
    from agentm.core.runtime import AgentSession
    from trajectory_index.adjudicate import (
        compare_values,
        resolve_aliases,
        resolve_references,
    )

    sf = AgentSession.create

    try:
        groups = await resolve_aliases(index, model=model, apply=True, session_factory=sf)
        logger.info("Pass 2a: merged {} alias groups", len(groups))
    except Exception:
        logger.warning("Pass 2a (alias resolution) failed, skipping", exc_info=True)

    try:
        n = await resolve_references(index, model=model, apply=True, session_factory=sf)
        logger.info("Pass 2b: resolved {} anaphors", n)
    except Exception:
        logger.warning("Pass 2b (coreference) failed, skipping", exc_info=True)

    index.build_dependencies()

    try:
        results = await compare_values(index, model=model, apply=True, session_factory=sf)
        contradicted = sum(1 for _, o in results if o == "contradict")
        logger.info("Pass 3.5: {} edges judged, {} contradicted", len(results), contradicted)
    except Exception:
        logger.warning("Pass 3.5 (value fidelity) failed, skipping", exc_info=True)


async def build_index(
    chunks: list[ExtractedChunk],
    *,
    namespace_fn: Callable[[str, dict[str, Any]], str] | None = _span_namespace,
    model: str | None = None,
    resolve: bool = True,
) -> Any:
    """Build a TrajectoryIndex from extraction chunks.

    Runs all three passes:
      Pass 1: populate symbols + references from extraction results
      Pass 2: alias resolution + coreference (LLM, best-effort)
      Pass 3: def-use / grounding (deterministic)
      Pass 3.5: value fidelity comparison (LLM, best-effort)
    """
    from trajectory_index.index import TrajectoryIndex

    index = TrajectoryIndex()
    for chunk in chunks:
        index.populate_from_extraction(
            chunk.result,
            _to_index_dicts(chunk.messages),
            run_id=chunk.run_id,
            namespace_fn=namespace_fn,
        )

    if resolve:
        await resolve_index(index, model=model)
    else:
        index.build_dependencies()

    return index
