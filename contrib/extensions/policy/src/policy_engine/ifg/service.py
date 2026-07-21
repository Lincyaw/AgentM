"""High-level IFG backfill and realtime append services."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Sequence

from agentm.core.abi.trajectory import Turn, TurnCheckpoint
from sqlalchemy.engine import Connection

from policy_engine.repository_index import RepositoryIndex

from .extract import extract_ifg_from_tool_events
from .normalize import (
    preferred_tool_rows,
    tool_event_from_policy_row,
    tool_events_from_turns,
)
from .project import (
    build_ifg_graph,
    build_ifg_symbols,
    resolve_path_candidates,
    unique_action_symbol_edges,
    unique_actions,
    unique_file_edges,
    unique_file_symbol_edges,
    unique_graph_edges,
    unique_nodes,
    unique_path_candidates,
    unique_source_units,
    unique_symbol_symbol_edges,
    unique_symbol_mentions,
    unique_symbols,
)
from .repository import (
    read_atomic_rows,
    replace_derived_projection,
    write_atomic_rows,
    write_session_summary,
)
from .schema import IFG_EXTRACTOR_VERSION, delete_ifg_session, ensure_ifg_schema
from .types import IfgBackfillResult, IfgExtractionRows, IfgToolEvent


def backfill_ifg_from_policy_events(
    conn: Connection,
    session_id: str,
    *,
    replace: bool = True,
    extractor_version: str = IFG_EXTRACTOR_VERSION,
) -> IfgBackfillResult:
    ensure_ifg_schema(conn)
    deleted = (
        delete_ifg_session(conn, session_id, extractor_version=extractor_version)
        if replace
        else 0
    )
    source_rows = preferred_tool_rows(conn, session_id)
    source_events = tuple(tool_event_from_policy_row(row) for row in source_rows)
    return persist_ifg_tool_events(
        conn,
        source_events,
        session_id=session_id,
        extractor_version=extractor_version,
        deleted=deleted,
        update_summary=True,
    )


def backfill_ifg_from_trajectory_turns(
    conn: Connection,
    session_id: str,
    turns: Iterable[Turn | TurnCheckpoint],
    *,
    cwd: str | None = None,
    replace: bool = True,
    extractor_version: str = IFG_EXTRACTOR_VERSION,
) -> IfgBackfillResult:
    ensure_ifg_schema(conn)
    deleted = (
        delete_ifg_session(conn, session_id, extractor_version=extractor_version)
        if replace
        else 0
    )
    source_events = tool_events_from_turns(turns, session_id=session_id, cwd=cwd)
    return persist_ifg_tool_events(
        conn,
        source_events,
        session_id=session_id,
        extractor_version=extractor_version,
        deleted=deleted,
        update_summary=True,
    )


def persist_ifg_tool_events(
    conn: Connection,
    events: Sequence[IfgToolEvent],
    *,
    session_id: str | None = None,
    extractor_version: str = IFG_EXTRACTOR_VERSION,
    deleted: int = 0,
    update_summary: bool = False,
    repository_index: RepositoryIndex | None = None,
    rebuild_projection: bool = True,
) -> IfgBackfillResult:
    ensure_ifg_schema(conn)
    source_events = tuple(events)
    resolved_session_id = session_id or _events_session_id(source_events)
    extracted = extract_ifg_from_tool_events(
        source_events,
        extractor_version=extractor_version,
        repository_index=repository_index,
    )
    atomic_rows = _dedupe_atomic_rows(extracted)
    now = time.time()

    write_atomic_rows(
        conn,
        session_id=resolved_session_id,
        events=source_events,
        rows=atomic_rows,
        extractor_version=extractor_version,
        now=now,
    )

    primary_projection: IfgExtractionRows | None = None
    if rebuild_projection:
        sessions = _event_sessions(source_events, fallback=resolved_session_id)
        for rebuild_session_id in sessions:
            projection = rebuild_ifg_projection(
                conn,
                rebuild_session_id,
                extractor_version=extractor_version,
                now=now,
                repository_index=repository_index,
            )
            if rebuild_session_id == resolved_session_id:
                primary_projection = projection
    if primary_projection is None:
        primary_projection = (
            _projection_from_atomic_rows(
                atomic_rows,
                extractor_version=extractor_version,
                repository_index=repository_index,
            )
            if rebuild_projection
            else atomic_rows
        )
    result = _backfill_result(
        session_id=resolved_session_id,
        extractor_version=extractor_version,
        source_events=len(source_events),
        projection=primary_projection,
        errors=len(extracted.errors),
        deleted=deleted,
    )
    summary = {
        "session_id": result.session_id,
        "extractor_version": result.extractor_version,
        "source_events": result.source_events,
        "graph_nodes": result.graph_nodes,
        "graph_edges": result.graph_edges,
        "actions": result.actions,
        "files": result.files,
        "file_edges": result.file_edges,
        "path_candidates": result.path_candidates,
        "unresolved_path_candidates": result.unresolved_path_candidates,
        "source_units": result.source_units,
        "symbols": result.symbols,
        "action_symbol_edges": result.action_symbol_edges,
        "file_symbol_edges": result.file_symbol_edges,
        "symbol_symbol_edges": result.symbol_symbol_edges,
        "symbol_mentions": result.symbol_mentions,
        "unresolved_symbol_mentions": result.unresolved_symbol_mentions,
        "errors": result.errors,
        "deleted": result.deleted,
        "action_kinds": result.action_kinds,
    }
    if update_summary:
        write_session_summary(
            conn,
            session_id=resolved_session_id,
            extractor_version=extractor_version,
            summary=summary,
            now=now,
        )
    return result


def rebuild_ifg_projection(
    conn: Connection,
    session_id: str,
    *,
    extractor_version: str = IFG_EXTRACTOR_VERSION,
    now: float | None = None,
    repository_index: RepositoryIndex | None = None,
) -> IfgExtractionRows:
    """Rebuild the complete derived graph once from durable atomic rows."""

    ensure_ifg_schema(conn)
    effective_now = time.time() if now is None else now
    atomic_rows = read_atomic_rows(
        conn,
        session_id=session_id,
        extractor_version=extractor_version,
    )
    projection = _projection_from_atomic_rows(
        atomic_rows,
        extractor_version=extractor_version,
        repository_index=repository_index,
    )
    replace_derived_projection(
        conn,
        session_id=session_id,
        extractor_version=extractor_version,
        projection=projection,
        now=effective_now,
    )
    return projection


def _dedupe_atomic_rows(rows: IfgExtractionRows) -> IfgExtractionRows:
    return IfgExtractionRows(
        actions=unique_actions(rows.actions),
        file_edges=unique_file_edges(
            tuple(edge for edge in rows.file_edges if edge.is_anchor)
        ),
        source_units=unique_source_units(rows.source_units),
        path_candidates=unique_path_candidates(rows.path_candidates),
        symbol_mentions=unique_symbol_mentions(rows.symbol_mentions),
        errors=rows.errors,
    )


def _projection_from_atomic_rows(
    rows: IfgExtractionRows,
    *,
    extractor_version: str,
    repository_index: RepositoryIndex | None = None,
) -> IfgExtractionRows:
    atomic_rows = _dedupe_atomic_rows(rows)
    resolved_file_edges = resolve_path_candidates(
        anchor_edges=atomic_rows.file_edges,
        candidates=atomic_rows.path_candidates,
        extractor_version=extractor_version,
        repository_index=repository_index,
    )
    symbol_projection = build_ifg_symbols(
        source_units=atomic_rows.source_units,
        symbol_mentions=atomic_rows.symbol_mentions,
        file_edges=resolved_file_edges,
        extractor_version=extractor_version,
        repository_index=repository_index,
    )
    symbols = unique_symbols(symbol_projection.symbols)
    action_symbol_edges = unique_action_symbol_edges(
        symbol_projection.action_symbol_edges
    )
    file_symbol_edges = unique_file_symbol_edges(symbol_projection.file_symbol_edges)
    symbol_symbol_edges = unique_symbol_symbol_edges(
        symbol_projection.symbol_symbol_edges
    )
    graph = build_ifg_graph(
        actions=atomic_rows.actions,
        source_units=atomic_rows.source_units,
        file_edges=resolved_file_edges,
        action_symbol_edges=action_symbol_edges,
        file_symbol_edges=file_symbol_edges,
        symbol_symbol_edges=symbol_symbol_edges,
        symbols=symbols,
        extractor_version=extractor_version,
    )
    return IfgExtractionRows(
        actions=atomic_rows.actions,
        file_edges=resolved_file_edges,
        source_units=atomic_rows.source_units,
        path_candidates=atomic_rows.path_candidates,
        symbol_mentions=atomic_rows.symbol_mentions,
        symbols=symbols,
        action_symbol_edges=action_symbol_edges,
        file_symbol_edges=file_symbol_edges,
        symbol_symbol_edges=symbol_symbol_edges,
        graph_nodes=unique_nodes(graph.graph_nodes),
        graph_edges=unique_graph_edges(graph.graph_edges),
        errors=atomic_rows.errors + symbol_projection.errors,
    )


def _backfill_result(
    *,
    session_id: str,
    extractor_version: str,
    source_events: int,
    projection: IfgExtractionRows,
    errors: int,
    deleted: int,
) -> IfgBackfillResult:
    action_kinds = Counter(action.action_kind for action in projection.actions)
    return IfgBackfillResult(
        session_id=session_id,
        extractor_version=extractor_version,
        source_events=source_events,
        graph_nodes=len(projection.graph_nodes),
        graph_edges=len(projection.graph_edges),
        actions=len(projection.actions),
        files=len({edge.path for edge in projection.file_edges}),
        file_edges=len(projection.file_edges),
        path_candidates=len(projection.path_candidates),
        unresolved_path_candidates=_unresolved_path_candidate_count(projection),
        source_units=len(projection.source_units),
        symbols=len(projection.symbols),
        action_symbol_edges=len(projection.action_symbol_edges),
        file_symbol_edges=len(projection.file_symbol_edges),
        symbol_symbol_edges=len(projection.symbol_symbol_edges),
        symbol_mentions=len(projection.symbol_mentions),
        unresolved_symbol_mentions=len(
            {
                edge.metadata.get("mention_id")
                for edge in projection.action_symbol_edges
                if edge.metadata.get("resolution") == "unresolved"
            }
        ),
        errors=errors,
        deleted=deleted,
        action_kinds=dict(action_kinds),
    )


def _unresolved_path_candidate_count(projection: IfgExtractionRows) -> int:
    resolved_ids = {
        edge.metadata.get("candidate_id")
        for edge in projection.file_edges
        if edge.metadata.get("candidate_id")
    }
    return sum(
        candidate.candidate_id not in resolved_ids
        for candidate in projection.path_candidates
    )


def _events_session_id(events: Sequence[IfgToolEvent]) -> str:
    for event in events:
        if event.session_id:
            return event.session_id
    return ""


def _event_sessions(
    events: Sequence[IfgToolEvent], *, fallback: str
) -> tuple[str, ...]:
    sessions = tuple(
        dict.fromkeys(event.session_id for event in events if event.session_id)
    )
    if sessions:
        return sessions
    return (fallback,) if fallback else ()
