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


def _match_known_symbol(
    name: str,
    registry: list[dict[str, Any]],
    min_len: int = 3,
    min_ratio: float = 0.6,
) -> dict[str, Any] | None:
    """Find an existing registry entry by exact normalized match or substring.

    Conservative: only matches when one name contains the other (after
    normalization) AND the shorter name is at least ``min_ratio`` of the
    longer name's length. This prevents short substrings (e.g. ``abn``)
    from absorbing unrelated longer names.
    """
    from trajectory_index.index import normalize_name

    norm = normalize_name(name)
    if len(norm) < min_len:
        return None

    def _substr_ok(a: str, b: str) -> bool:
        """Check substring containment at token boundaries with length guard."""
        if a in b:
            shorter, longer = a, b
        elif b in a:
            shorter, longer = b, a
        else:
            return False
        if len(shorter) < min_len or len(shorter) / len(longer) < min_ratio:
            return False
        # Must match at a token boundary (start/end of string or after _./-)
        idx = longer.find(shorter)
        at_start = idx == 0 or longer[idx - 1] in "_.-/"
        at_end = idx + len(shorter) == len(longer) or longer[idx + len(shorter)] in "_.-/"
        return at_start and at_end

    for entry in registry:
        canonical = str(entry.get("name", ""))
        entry_norm = normalize_name(canonical)
        if not entry_norm or entry_norm == norm:
            continue

        if _substr_ok(norm, entry_norm):
            return entry

        for alias in entry.get("aliases", []):
            alias_norm = normalize_name(str(alias))
            if alias_norm == norm:
                return entry
            if _substr_ok(norm, alias_norm):
                return entry

    return None


def _prescan_structural(
    messages: list[AgentMessage],
    registry: list[dict[str, Any]] | None,
    seen: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Code-level structural extraction + fuzzy alias expansion.

    1. Extracts tool names and SQL table names from structured blocks.
    2. Fuzzy-matches new names against existing registry to expand aliases
       (increases recall without creating duplicate symbols).

    Returns (updated_registry, structural_symbols).
    """
    from trajectory_index.atom import _agentmsg_to_extraction_dict
    from trajectory_index.data import extract_structural_symbols
    from trajectory_index.index import normalize_name

    serialized = [
        d for i, m in enumerate(messages)
        if (d := _agentmsg_to_extraction_dict(m, i, truncate=False))
    ]
    structural = extract_structural_symbols(serialized)

    if seen is None:
        seen = set()
        if registry:
            for entry in registry:
                seen.add(normalize_name(str(entry.get("name", ""))))
                for alias in entry.get("aliases", []):
                    seen.add(normalize_name(str(alias)))

    updated = list(registry) if registry else []
    new_syms = []

    for sym in structural:
        norm = normalize_name(sym.name)
        if norm in seen:
            continue

        # Try fuzzy match against existing symbols
        match = _match_known_symbol(sym.name, updated)
        if match:
            # Add as alias to existing symbol
            aliases = match.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            if sym.name not in aliases:
                aliases.append(sym.name)
                match["aliases"] = aliases
            seen.add(norm)
            logger.debug("prescan: '{}' → alias of '{}'", sym.name, match.get("name"))
        else:
            # New symbol
            seen.add(norm)
            updated.append({"name": sym.name, "kind": sym.kind})
            new_syms.append(sym)

    return updated, new_syms


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    result: ExtractionResult | None
    session_id: str | None = None


async def _run_one(
    messages: list[AgentMessage],
    *,
    model: str | None,
    vocabulary: str,
    registry: list[dict[str, Any]] | None,
    message_id_start: int,
    cwd: str | None,
) -> ExtractionOutcome:
    from agentm.core.runtime import AgentSession

    from trajectory_index.atom import (
        build_extraction_config,
        build_extraction_prompt,
        run_extraction_session,
    )

    _captured_sid: list[str] = []
    _orig_create = AgentSession.create

    async def _capturing_spawn(config: Any) -> Any:
        session = await _orig_create(config)
        _captured_sid.append(session.session_id)
        return session

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
    result = await run_extraction_session(
        config, prompt, vocabulary=vocabulary, spawn=_capturing_spawn,
    )
    sid = _captured_sid[0] if _captured_sid else None
    return ExtractionOutcome(result=result, session_id=sid)


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
    message_id_start: int = 0


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
        outcome = await _run_one(
            messages, model=model, vocabulary=vocabulary,
            registry=None, message_id_start=0, cwd=cwd,
        )
        if outcome.result is None:
            return []
        extracted = ExtractedChunk(run_id=run_id, messages=messages, result=outcome.result)
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
            outcome = await _run_one(
                chunk.messages, model=model, vocabulary=vocabulary,
                registry=chunk_registry if chunk_registry else None,
                message_id_start=chunk.start, cwd=cwd,
            )
        except Exception:
            logger.exception("chunk {}/{} extraction failed", i + 1, len(chunks))
            continue
        result = outcome.result
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

        extracted = ExtractedChunk(
            run_id=run_id, messages=chunk.messages, result=result,
            message_id_start=chunk.start,
        )
        results.append(extracted)

        if on_chunk:
            on_chunk(extracted, chunk.messages)

        for sym in result.symbols:
            norm = normalize_name(sym.name)
            if norm in seen:
                continue
            # Fuzzy match: absorb as alias if close to existing symbol
            match = _match_known_symbol(sym.name, registry)
            if match:
                aliases = match.get("aliases", [])
                if not isinstance(aliases, list):
                    aliases = []
                if sym.name not in aliases:
                    aliases.append(sym.name)
                    match["aliases"] = aliases
                seen.add(norm)
                logger.debug("extract: '{}' → alias of '{}'", sym.name, match.get("name"))
            else:
                seen.add(norm)
                entry: dict[str, Any] = {"name": sym.name, "kind": sym.kind}
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
        if (d := _agentmsg_to_extraction_dict(m, i, truncate=False))
    ]


async def resolve_index(
    index: Any,
    *,
    model: str | None = None,
) -> None:
    """Run Pass 2 (alias resolution) and Pass 3 (dataflow).

    Best-effort: Pass 2 LLM failure is skipped; Pass 3 is deterministic.
    """
    from agentm.core.runtime import AgentSession
    from trajectory_index.adjudicate import resolve_aliases

    sf = AgentSession.create

    try:
        groups = await resolve_aliases(index, model=model, apply=True, session_factory=sf)
        logger.info("Pass 2: merged {} alias groups", len(groups))
    except Exception:
        logger.warning("Pass 2 (alias resolution) failed, skipping", exc_info=True)

    index.build_dependencies()


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
            chunk.messages,
            run_id=chunk.run_id,
            namespace_fn=namespace_fn,
            message_id_start=chunk.message_id_start,
        )

    if resolve:
        await resolve_index(index, model=model)
    else:
        index.build_dependencies()

    return index
