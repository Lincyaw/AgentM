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


# ---------------------------------------------------------------------------
# Core extraction — delegates to atom API
# ---------------------------------------------------------------------------




@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    result: ExtractionResult | None
    session_id: str | None = None


async def run_extraction(
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
class MessageChunk:
    start: int
    messages: list[AgentMessage]


def chunk_messages(
    messages: list[AgentMessage],
    size_range: tuple[int, int],
) -> list[MessageChunk]:
    import random

    from agentm.core.abi import ToolResultMessage

    lo, hi = size_range
    if len(messages) <= lo:
        return [MessageChunk(start=0, messages=messages)]

    chunks: list[MessageChunk] = []
    start = 0
    while start < len(messages):
        chunk_size = random.randint(lo, hi)
        end = start + chunk_size
        if end >= len(messages):
            chunks.append(MessageChunk(start=start, messages=messages[start:]))
            break
        while end < len(messages) and isinstance(messages[end], ToolResultMessage):
            end += 1
        chunks.append(MessageChunk(start=start, messages=messages[start:end]))
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


type OnChunkCallback = Callable[[ExtractedChunk, list[AgentMessage], list[dict[str, Any]]], None]


async def extract_symbols(
    messages: list[AgentMessage],
    *,
    model: str | None = None,
    vocabulary: str = "default",
    chunk_size: tuple[int, int] | None = None,
    on_chunk: OnChunkCallback | None = None,
    run_id: str = "",
    cwd: str | None = None,
    namespace_fn: Callable[[str, dict[str, Any]], str] | None = None,
) -> tuple[list[ExtractedChunk], Any]:
    """Extract symbols from trajectory messages.

    Uses a ``TrajectoryIndex`` as the single registry: each chunk's
    extraction result is populated into the index immediately, and the
    next chunk gets ``index.registry_snapshot()`` as its known-symbol
    context. This mirrors the atom's online flow.

    Returns ``(chunks, index)``.
    """
    from trajectory_index.ir.index import TrajectoryIndex

    if namespace_fn is None:
        namespace_fn = _span_namespace
    index = TrajectoryIndex()

    if chunk_size is None:
        outcome = await run_extraction(
            messages, model=model, vocabulary=vocabulary,
            registry=None, message_id_start=0, cwd=cwd,
        )
        if outcome.result is None:
            return [], index
        index.populate_from_extraction(
            outcome.result, messages,
            run_id=run_id, namespace_fn=namespace_fn,
        )
        extracted = ExtractedChunk(run_id=run_id, messages=messages, result=outcome.result)
        if on_chunk:
            on_chunk(extracted, messages, index.registry_snapshot())
        return [extracted], index

    chunks = chunk_messages(messages, chunk_size)
    results: list[ExtractedChunk] = []

    for i, chunk in enumerate(chunks):
        registry = index.registry_snapshot() or None

        try:
            outcome = await run_extraction(
                chunk.messages, model=model, vocabulary=vocabulary,
                registry=registry,
                message_id_start=chunk.start, cwd=cwd,
            )
        except Exception:
            logger.exception("chunk {}/{} extraction failed", i + 1, len(chunks))
            continue
        result = outcome.result
        if result is None:
            logger.warning("chunk {}/{} no parseable result", i + 1, len(chunks))
            continue

        index.populate_from_extraction(
            result, chunk.messages,
            run_id=run_id,
            namespace_fn=namespace_fn,
            message_id_start=chunk.start,
        )

        extracted = ExtractedChunk(
            run_id=run_id, messages=chunk.messages, result=result,
            message_id_start=chunk.start,
        )
        results.append(extracted)

        if on_chunk:
            on_chunk(extracted, chunk.messages, index.registry_snapshot())

    return results, index


# ---------------------------------------------------------------------------
# Index building — delegates to TrajectoryIndex.populate_from_extraction
# ---------------------------------------------------------------------------

_SPAN_SYMBOL_RE: Final = re.compile(r"^(?:span\s+)?s\d+$", re.IGNORECASE)


def _symbol_aliases(symbol: dict[str, Any]) -> list[str]:
    aliases = symbol.get("aliases", [])
    if not isinstance(aliases, list):
        return []
    return [str(alias) for alias in aliases]


def _span_namespace(run_id: str, sym: dict[str, Any]) -> str:
    """Namespace function for span-ID symbols (telbench-style)."""
    if not run_id or str(sym.get("kind", "")).lower() != "file":
        return ""
    names = [str(sym.get("name", "")), *_symbol_aliases(sym)]
    return run_id if any(_SPAN_SYMBOL_RE.match(str(name).strip()) for name in names) else ""



def _first_user_text_from_index(index: Any) -> str:
    """Extract the first user step's text from an already-populated index."""
    from trajectory_index.ir.models import StepRole
    for step in sorted(index.steps.values(), key=lambda s: s.index):
        if step.role == StepRole.USER and step.content.strip():
            return step.content
    return ""


async def resolve_index(
    index: Any,
    *,
    model: str | None = None,
    run_id: str = "",
) -> None:
    """Run Pass 2 (alias resolution), Pass 3 (dataflow), and Pass 4 (claims/constraints).

    Best-effort: each pass failure is logged and skipped.
    """
    from agentm.core.runtime import AgentSession
    from trajectory_index.pass2_edges.identity import resolve_aliases

    sf = AgentSession.create

    try:
        groups = await resolve_aliases(index, model=model, apply=True, session_factory=sf)
        logger.info("Pass 2: merged {} alias groups", len(groups))
    except Exception:
        logger.warning("Pass 2 (alias resolution) failed, skipping", exc_info=True)

    index.build_dependencies()

    # Pass 4: claim/constraint adjudication.
    from trajectory_index.pass2_edges.claims import build_claim_edges
    from trajectory_index.pass2_edges.intent_alignment import build_intent_alignment_edges
    from trajectory_index.pass3_folds.claim_status import fold_claim_statuses
    from trajectory_index.pass3_folds.constraints import analyze_constraints

    constraints = list(index.constraints.values())
    question = _first_user_text_from_index(index)

    _commit_obj = None
    commit_binding = ""
    commit_step_id_str = ""
    precomputed: dict[str, Any] | None = None

    if constraints:
        from trajectory_index.ir.diagnostics import Diagnostics as _Diag
        from trajectory_index.pass3_folds.constraints import _detect_commit

        _all_steps = sorted(
            (s for s in index.steps.values()
             if not run_id or s.run_id == run_id),
            key=lambda s: s.index,
        )
        try:
            _commit_obj = await _detect_commit(
                index, _all_steps, question=question,
                model=model, session_factory=sf, diag=_Diag(),
            )
            if _commit_obj is not None:
                commit_binding = _commit_obj.binding
                commit_step_id_str = _commit_obj.step.step_id
        except Exception:
            logger.warning("Pass 4 (commit detection) failed, skipping constraints", exc_info=True)
            constraints = []

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


async def build_index(
    chunks_or_index: list[ExtractedChunk] | Any,
    *,
    namespace_fn: Callable[[str, dict[str, Any]], str] | None = _span_namespace,
    model: str | None = None,
    resolve: bool = True,
    run_id: str = "",
) -> Any:
    """Build or finalize a TrajectoryIndex.

    Accepts either an already-populated index (from ``extract_symbols``)
    or a list of chunks to populate from scratch. Then runs Pass 2
    (alias resolution), Pass 3 (def-use / grounding), and Pass 4
    (claims / constraints / intent alignment).
    """
    from trajectory_index.ir.index import TrajectoryIndex

    if isinstance(chunks_or_index, TrajectoryIndex):
        index = chunks_or_index
    else:
        index = TrajectoryIndex()
        for chunk in chunks_or_index:
            index.populate_from_extraction(
                chunk.result,
                chunk.messages,
                run_id=chunk.run_id,
                namespace_fn=namespace_fn,
                message_id_start=chunk.message_id_start,
            )

    if resolve:
        await resolve_index(index, model=model, run_id=run_id)
    else:
        index.build_dependencies()

    # Pass 3: value flow (constraint checks need LLM).
    try:
        from agentm.core.runtime import AgentSession
        from trajectory_index.pass3_folds.value_flow import build_value_flow

        vf = await build_value_flow(
            index, model=model, session_factory=AgentSession.create,
        )
        index._value_flow = vf  # type: ignore[attr-defined]
        logger.info(
            "value flow: {} timelines, {} iterations, {} constraint checks",
            len(vf.get("value_timelines", [])),
            len(vf.get("iterations", [])),
            len(vf.get("constraint_checks", [])),
        )
    except Exception:
        logger.warning("value flow pass failed, skipping", exc_info=True)

    return index
