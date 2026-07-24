# code-health: ignore-file[AM025] -- repository maps untyped SQLite rows to IFG DTOs
"""SQLite repository for IFG atomic facts and derived projections."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from sqlalchemy.engine import Connection

from .schema import IFG_DERIVED_TABLES
from .types import (
    IfgActionFileEdgeRow,
    IfgActionRow,
    IfgActionSymbolEdgeRow,
    IfgExtractionRows,
    IfgFileSymbolEdgeRow,
    IfgGraphEdgeRow,
    IfgNodeRow,
    IfgPathCandidateRow,
    IfgSourceUnitRow,
    IfgSymbolMentionRow,
    IfgSymbolRow,
    IfgSymbolSymbolEdgeRow,
    IfgToolEvent,
)
from .utils import _aggregate_confidence, _loads, _row_is_error, _stable_id, _to_json


def write_atomic_rows(
    conn: Connection,
    *,
    session_id: str,
    events: Sequence[IfgToolEvent],
    rows: IfgExtractionRows,
    extractor_version: str,
    now: float,
) -> None:
    """Persist source-normalized events and atomic IFG facts."""
    _write_normalized_tool_events(conn, events, extractor_version, now)
    _write_actions(conn, rows.actions, now)
    _write_file_edges(
        conn, tuple(edge for edge in rows.file_edges if edge.is_anchor), now
    )
    _write_source_units(conn, rows.source_units, now)
    _write_path_candidates(conn, rows.path_candidates, now)
    _write_symbol_mentions(conn, rows.symbol_mentions, now)
    _write_errors(conn, session_id, rows.errors, extractor_version, now)


def read_atomic_rows(
    conn: Connection,
    *,
    session_id: str,
    extractor_version: str,
) -> IfgExtractionRows:
    """Load durable atomic facts used to rebuild derived IFG projections."""
    return IfgExtractionRows(
        actions=read_actions(conn, session_id, extractor_version),
        file_edges=read_file_edges(
            conn, session_id, extractor_version, anchors_only=True
        ),
        source_units=read_source_units(conn, session_id, extractor_version),
        path_candidates=read_path_candidates(conn, session_id, extractor_version),
        symbol_mentions=read_symbol_mentions(conn, session_id, extractor_version),
    )


def replace_derived_projection(
    conn: Connection,
    *,
    session_id: str,
    extractor_version: str,
    projection: IfgExtractionRows,
    now: float,
) -> None:
    """Replace derived graph/symbol/file aggregates from a full-session projection."""
    delete_ifg_derived_session(conn, session_id, extractor_version)
    _write_file_edges(conn, projection.file_edges, now)
    _write_symbols(conn, projection.symbols, now)
    _write_action_symbol_edges(conn, projection.action_symbol_edges, now)
    _write_file_symbol_edges(conn, projection.file_symbol_edges, now)
    _write_symbol_symbol_edges(conn, projection.symbol_symbol_edges, now)
    _write_files(conn, session_id, projection.file_edges, extractor_version, now)
    _write_nodes(conn, projection.graph_nodes, now)
    _write_graph_edges(conn, projection.graph_edges, now)


def write_session_summary(
    conn: Connection,
    *,
    session_id: str,
    extractor_version: str,
    summary: Mapping[str, object],
    now: float,
) -> None:
    conn.exec_driver_sql(
        """
        INSERT INTO ifg_session_summary
            (session_id, extractor_version, updated_at, summary_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id, extractor_version) DO UPDATE SET
            updated_at = excluded.updated_at,
            summary_json = excluded.summary_json
        """,
        (session_id, extractor_version, now, _to_json(summary)),
    )


def delete_ifg_derived_session(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> int:
    deleted = 0
    cursor = conn.exec_driver_sql(
        """
        DELETE FROM ifg_action_file_edges
        WHERE session_id = ? AND extractor_version = ? AND is_anchor = 0
        """,
        (session_id, extractor_version),
    )
    deleted += cursor.rowcount
    for table in IFG_DERIVED_TABLES:
        cursor = conn.exec_driver_sql(
            f"DELETE FROM {table} WHERE session_id = ? AND extractor_version = ?",  # noqa: S608
            (session_id, extractor_version),
        )
        deleted += cursor.rowcount
    return deleted


def _write_normalized_tool_events(
    conn: Connection,
    events: Sequence[IfgToolEvent],
    extractor_version: str,
    now: float,
) -> None:
    for event in events:
        normalized_event_id = _stable_id(
            "normalized_event",
            event.session_id,
            str(event.tool_call_id or f"event:{event.event_id}"),
            event.phase,
            str(event.event_id),
            extractor_version,
        )
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_normalized_tool_events
                (normalized_event_id, session_id, event_id, tool_call_id, phase,
                 tool_name, args_json, result_json, processed_json, state_json,
                 cwd, turn, ts, is_error, source, extractor_version,
                 raw_evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(normalized_event_id) DO UPDATE SET
                event_id = excluded.event_id,
                tool_call_id = excluded.tool_call_id,
                phase = excluded.phase,
                tool_name = excluded.tool_name,
                args_json = excluded.args_json,
                result_json = excluded.result_json,
                processed_json = excluded.processed_json,
                state_json = excluded.state_json,
                cwd = excluded.cwd,
                turn = excluded.turn,
                ts = excluded.ts,
                is_error = excluded.is_error,
                source = excluded.source,
                extractor_version = excluded.extractor_version,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                normalized_event_id,
                event.session_id,
                event.event_id,
                event.tool_call_id,
                event.phase,
                event.tool_name,
                _to_json(event.args),
                _to_json(event.result),
                _to_json(event.processed),
                _to_json(event.state),
                event.cwd,
                event.turn,
                event.ts,
                int(_row_is_error(event.result, event.processed)),
                event.source,
                extractor_version,
                _to_json(event.raw_evidence),
                now,
                now,
            ),
        )


def _write_actions(
    conn: Connection,
    actions: Sequence[IfgActionRow],
    now: float,
) -> None:
    for action in actions:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_actions
                (action_id, session_id, turn, event_id, tool_call_id, tool_name,
                 segment_index, command, action_kind, family, template, source,
                 confidence, extractor_version, raw_evidence_json, created_at,
                 updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(action_id) DO UPDATE SET
                turn = excluded.turn,
                event_id = excluded.event_id,
                tool_call_id = excluded.tool_call_id,
                tool_name = excluded.tool_name,
                segment_index = excluded.segment_index,
                command = excluded.command,
                action_kind = excluded.action_kind,
                family = excluded.family,
                template = excluded.template,
                source = excluded.source,
                confidence = excluded.confidence,
                extractor_version = excluded.extractor_version,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                action.action_id,
                action.session_id,
                action.turn,
                action.event_id,
                action.tool_call_id,
                action.tool_name,
                action.segment_index,
                action.command,
                action.action_kind,
                action.family,
                action.template,
                action.source,
                action.confidence,
                action.extractor_version,
                _to_json(action.raw_evidence),
                now,
                now,
            ),
        )


def _write_file_edges(
    conn: Connection,
    edges: Sequence[IfgActionFileEdgeRow],
    now: float,
) -> None:
    for edge in edges:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_action_file_edges
                (edge_id, session_id, action_id, path, relation, turn, event_id,
                 source, confidence, is_anchor, extractor_version, content_hash, before_hash,
                 after_hash, content_state, line_range_json, span_json, metadata_json,
                 raw_evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                path = excluded.path,
                relation = excluded.relation,
                turn = excluded.turn,
                event_id = excluded.event_id,
                source = excluded.source,
                confidence = excluded.confidence,
                is_anchor = excluded.is_anchor,
                extractor_version = excluded.extractor_version,
                content_hash = excluded.content_hash,
                before_hash = excluded.before_hash,
                after_hash = excluded.after_hash,
                content_state = excluded.content_state,
                line_range_json = excluded.line_range_json,
                span_json = excluded.span_json,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json
            """,
            (
                edge.edge_id,
                edge.session_id,
                edge.action_id,
                edge.path,
                edge.relation,
                edge.turn,
                edge.event_id,
                edge.source,
                edge.confidence,
                int(edge.is_anchor),
                edge.extractor_version,
                edge.content_hash,
                edge.before_hash,
                edge.after_hash,
                edge.content_state,
                _to_json(edge.line_range),
                _to_json(edge.span),
                _to_json(edge.metadata),
                _to_json(edge.raw_evidence),
                now,
            ),
        )


def _write_path_candidates(
    conn: Connection,
    candidates: Sequence[IfgPathCandidateRow],
    now: float,
) -> None:
    for candidate in candidates:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_path_candidates
                (candidate_id, session_id, extractor_version, action_id,
                 source_unit_id, path_text, normalized_path, path_kind, relation,
                 turn, event_id, source, confidence, metadata_json,
                 raw_evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                action_id = excluded.action_id,
                source_unit_id = excluded.source_unit_id,
                path_text = excluded.path_text,
                normalized_path = excluded.normalized_path,
                path_kind = excluded.path_kind,
                relation = excluded.relation,
                turn = excluded.turn,
                event_id = excluded.event_id,
                source = excluded.source,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json
            """,
            (
                candidate.candidate_id,
                candidate.session_id,
                candidate.extractor_version,
                candidate.action_id,
                candidate.source_unit_id,
                candidate.path_text,
                candidate.normalized_path,
                candidate.path_kind,
                candidate.relation,
                candidate.turn,
                candidate.event_id,
                candidate.source,
                candidate.confidence,
                _to_json(candidate.metadata),
                _to_json(candidate.raw_evidence),
                now,
            ),
        )


def _write_source_units(
    conn: Connection,
    units: Sequence[IfgSourceUnitRow],
    now: float,
) -> None:
    for unit in units:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_source_units
                (source_unit_id, session_id, extractor_version, action_id, kind,
                 origin, path, relation, turn, event_id, tool_name, language,
                 content_hash, previous_content_hash, result_content_hash,
                 unit_hash, content_state, line_range_json, span_json, content_text,
                 metadata_json, raw_evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_unit_id) DO UPDATE SET
                action_id = excluded.action_id,
                kind = excluded.kind,
                origin = excluded.origin,
                path = excluded.path,
                relation = excluded.relation,
                turn = excluded.turn,
                event_id = excluded.event_id,
                tool_name = excluded.tool_name,
                language = excluded.language,
                content_hash = excluded.content_hash,
                previous_content_hash = excluded.previous_content_hash,
                result_content_hash = excluded.result_content_hash,
                unit_hash = excluded.unit_hash,
                content_state = excluded.content_state,
                line_range_json = excluded.line_range_json,
                span_json = excluded.span_json,
                content_text = excluded.content_text,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                unit.source_unit_id,
                unit.session_id,
                unit.extractor_version,
                unit.action_id,
                unit.kind,
                unit.origin,
                unit.path,
                unit.relation,
                unit.turn,
                unit.event_id,
                unit.tool_name,
                unit.language,
                unit.content_hash,
                unit.previous_content_hash,
                unit.result_content_hash,
                unit.unit_hash,
                unit.content_state,
                _to_json(unit.line_range),
                _to_json(unit.span),
                unit.content_text,
                _to_json(unit.metadata),
                _to_json(unit.raw_evidence),
                now,
                now,
            ),
        )


def _write_symbol_mentions(
    conn: Connection,
    mentions: Sequence[IfgSymbolMentionRow],
    now: float,
) -> None:
    for mention in mentions:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_symbol_mentions
                (mention_id, session_id, extractor_version, action_id,
                 source_unit_id, symbol_text, turn, event_id, source, confidence,
                 path, metadata_json, raw_evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mention_id) DO UPDATE SET
                action_id = excluded.action_id,
                source_unit_id = excluded.source_unit_id,
                symbol_text = excluded.symbol_text,
                turn = excluded.turn,
                event_id = excluded.event_id,
                source = excluded.source,
                confidence = excluded.confidence,
                path = excluded.path,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json
            """,
            (
                mention.mention_id,
                mention.session_id,
                mention.extractor_version,
                mention.action_id,
                mention.source_unit_id,
                mention.symbol_text,
                mention.turn,
                mention.event_id,
                mention.source,
                mention.confidence,
                mention.path,
                _to_json(mention.metadata),
                _to_json(mention.raw_evidence),
                now,
            ),
        )


def _write_symbols(
    conn: Connection,
    symbols: Sequence[IfgSymbolRow],
    now: float,
) -> None:
    for symbol in symbols:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_symbols
                (symbol_id, session_id, extractor_version, kind, qualified_name,
                 path, stable_key, first_seen_turn, last_seen_turn,
                 observation_count, source, confidence, metadata_json,
                 raw_evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol_id) DO UPDATE SET
                kind = excluded.kind,
                qualified_name = excluded.qualified_name,
                path = excluded.path,
                stable_key = excluded.stable_key,
                first_seen_turn = min(ifg_symbols.first_seen_turn, excluded.first_seen_turn),
                last_seen_turn = max(ifg_symbols.last_seen_turn, excluded.last_seen_turn),
                observation_count = excluded.observation_count,
                source = excluded.source,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                symbol.symbol_id,
                symbol.session_id,
                symbol.extractor_version,
                symbol.kind,
                symbol.qualified_name,
                symbol.path,
                symbol.stable_key,
                symbol.first_seen_turn,
                symbol.last_seen_turn,
                symbol.observation_count,
                symbol.source,
                symbol.confidence,
                _to_json(symbol.metadata),
                _to_json(symbol.raw_evidence),
                now,
                now,
            ),
        )


def _write_action_symbol_edges(
    conn: Connection,
    edges: Sequence[IfgActionSymbolEdgeRow],
    now: float,
) -> None:
    for edge in edges:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_action_symbol_edges
                (edge_id, session_id, extractor_version, action_id, symbol_id,
                 relation, turn, event_id, source, confidence, metadata_json,
                 raw_evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                action_id = excluded.action_id,
                symbol_id = excluded.symbol_id,
                relation = excluded.relation,
                turn = excluded.turn,
                event_id = excluded.event_id,
                source = excluded.source,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                edge.edge_id,
                edge.session_id,
                edge.extractor_version,
                edge.action_id,
                edge.symbol_id,
                edge.relation,
                edge.turn,
                edge.event_id,
                edge.source,
                edge.confidence,
                _to_json(edge.metadata),
                _to_json(edge.raw_evidence),
                now,
                now,
            ),
        )


def _write_file_symbol_edges(
    conn: Connection,
    edges: Sequence[IfgFileSymbolEdgeRow],
    now: float,
) -> None:
    for edge in edges:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_file_symbol_edges
                (edge_id, session_id, extractor_version, path, symbol_id, relation,
                 turn, event_id, source, confidence, metadata_json,
                 raw_evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                path = excluded.path,
                symbol_id = excluded.symbol_id,
                relation = excluded.relation,
                turn = excluded.turn,
                event_id = excluded.event_id,
                source = excluded.source,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                edge.edge_id,
                edge.session_id,
                edge.extractor_version,
                edge.path,
                edge.symbol_id,
                edge.relation,
                edge.turn,
                edge.event_id,
                edge.source,
                edge.confidence,
                _to_json(edge.metadata),
                _to_json(edge.raw_evidence),
                now,
                now,
            ),
        )


def _write_symbol_symbol_edges(
    conn: Connection,
    edges: Sequence[IfgSymbolSymbolEdgeRow],
    now: float,
) -> None:
    for edge in edges:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_symbol_symbol_edges
                (edge_id, session_id, extractor_version, from_symbol_id, to_symbol_id,
                 relation, turn, event_id, source, confidence, metadata_json,
                 raw_evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                from_symbol_id = excluded.from_symbol_id,
                to_symbol_id = excluded.to_symbol_id,
                relation = excluded.relation,
                turn = excluded.turn,
                event_id = excluded.event_id,
                source = excluded.source,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                edge.edge_id,
                edge.session_id,
                edge.extractor_version,
                edge.from_symbol_id,
                edge.to_symbol_id,
                edge.relation,
                edge.turn,
                edge.event_id,
                edge.source,
                edge.confidence,
                _to_json(edge.metadata),
                _to_json(edge.raw_evidence),
                now,
                now,
            ),
        )


def _write_nodes(
    conn: Connection,
    nodes: Sequence[IfgNodeRow],
    now: float,
) -> None:
    for node in nodes:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_nodes
                (node_id, session_id, extractor_version, node_type, stable_key,
                 display_name, first_seen_turn, last_seen_turn, observation_count,
                 source, confidence, metadata_json, raw_evidence_json, created_at,
                 updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                node_type = excluded.node_type,
                stable_key = excluded.stable_key,
                display_name = excluded.display_name,
                first_seen_turn = min(ifg_nodes.first_seen_turn, excluded.first_seen_turn),
                last_seen_turn = max(ifg_nodes.last_seen_turn, excluded.last_seen_turn),
                observation_count = excluded.observation_count,
                source = excluded.source,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                node.node_id,
                node.session_id,
                node.extractor_version,
                node.node_type,
                node.stable_key,
                node.display_name,
                node.first_seen_turn,
                node.last_seen_turn,
                node.observation_count,
                node.source,
                node.confidence,
                _to_json(node.metadata),
                _to_json(node.raw_evidence),
                now,
                now,
            ),
        )


def _write_graph_edges(
    conn: Connection,
    edges: Sequence[IfgGraphEdgeRow],
    now: float,
) -> None:
    for edge in edges:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_edges
                (edge_id, session_id, extractor_version, from_node_id, to_node_id,
                 relation, turn, event_id, source, confidence, metadata_json,
                 raw_evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                from_node_id = excluded.from_node_id,
                to_node_id = excluded.to_node_id,
                relation = excluded.relation,
                turn = excluded.turn,
                event_id = excluded.event_id,
                source = excluded.source,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                edge.edge_id,
                edge.session_id,
                edge.extractor_version,
                edge.from_node_id,
                edge.to_node_id,
                edge.relation,
                edge.turn,
                edge.event_id,
                edge.source,
                edge.confidence,
                _to_json(edge.metadata),
                _to_json(edge.raw_evidence),
                now,
                now,
            ),
        )


def _write_files(
    conn: Connection,
    session_id: str,
    file_edges: Sequence[IfgActionFileEdgeRow],
    extractor_version: str,
    now: float,
) -> None:
    by_path: dict[str, list[IfgActionFileEdgeRow]] = {}
    for edge in file_edges:
        by_path.setdefault(edge.path, []).append(edge)
    for path, edges in by_path.items():
        relation_counts = Counter(edge.relation for edge in edges)
        resolution_counts = Counter(
            str(edge.metadata.get("resolution") or "anchor") for edge in edges
        )
        existence_counts = Counter(
            str(edge.metadata.get("existence") or "unknown") for edge in edges
        )
        anchor_count = sum(edge.is_anchor for edge in edges)
        file_metadata = {
            "relation_counts": dict(relation_counts),
            "resolution_counts": dict(resolution_counts),
            "existence_counts": dict(existence_counts),
            "anchor_count": anchor_count,
            "resolved_candidate_count": len(edges) - anchor_count,
        }
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_files
                (session_id, extractor_version, path, first_seen_turn,
                 last_seen_turn, observation_count, source, confidence,
                 metadata_json, raw_evidence_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, extractor_version, path) DO UPDATE SET
                first_seen_turn = min(ifg_files.first_seen_turn, excluded.first_seen_turn),
                last_seen_turn = max(ifg_files.last_seen_turn, excluded.last_seen_turn),
                observation_count = excluded.observation_count,
                source = excluded.source,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                raw_evidence_json = excluded.raw_evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                session_id,
                extractor_version,
                path,
                min(edge.turn for edge in edges),
                max(edge.turn for edge in edges),
                len(edges),
                "action_file_edges",
                _aggregate_confidence(edge.confidence for edge in edges),
                _to_json(file_metadata),
                _to_json(
                    {
                        "edge_ids": [edge.edge_id for edge in edges[:100]],
                        **file_metadata,
                    }
                ),
                now,
            ),
        )


def _write_errors(
    conn: Connection,
    session_id: str,
    errors: Sequence[Mapping[str, object]],
    extractor_version: str,
    now: float,
) -> None:
    for error in errors:
        conn.exec_driver_sql(
            """
            INSERT INTO ifg_extraction_error
                (ts, session_id, turn, event_id, tool_call_id, tool_name,
                 extractor_version, error, raw_evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                session_id,
                error.get("turn"),
                error.get("event_id"),
                error.get("tool_call_id"),
                error.get("tool_name"),
                extractor_version,
                str(error.get("error", "")),
                _to_json(error.get("raw_evidence", {})),
            ),
        )


def read_actions(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> tuple[IfgActionRow, ...]:
    rows = conn.exec_driver_sql(
        """
        SELECT *
        FROM ifg_actions
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY turn, event_id, segment_index
        """,
        (session_id, extractor_version),
    ).mappings()
    return tuple(
        IfgActionRow(
            action_id=str(row["action_id"]),
            session_id=str(row["session_id"]),
            turn=int(row["turn"]),
            event_id=_int_or_none(row["event_id"]),
            tool_call_id=_str_or_none(row["tool_call_id"]),
            tool_name=str(row["tool_name"]),
            segment_index=_int_or_none(row["segment_index"]),
            command=_str_or_none(row["command"]),
            action_kind=str(row["action_kind"]),
            family=str(row["family"]),
            template=_str_or_none(row["template"]),
            source=str(row["source"]),
            confidence=str(row["confidence"]),
            extractor_version=str(row["extractor_version"]),
            raw_evidence=_mapping(row["raw_evidence_json"]),
        )
        for row in rows
    )


def read_file_edges(
    conn: Connection,
    session_id: str,
    extractor_version: str,
    *,
    anchors_only: bool = False,
) -> tuple[IfgActionFileEdgeRow, ...]:
    anchor_clause = " AND is_anchor = 1" if anchors_only else ""
    rows = conn.exec_driver_sql(
        f"""
        SELECT *
        FROM ifg_action_file_edges
        WHERE session_id = ? AND extractor_version = ?{anchor_clause}
        ORDER BY turn, event_id, path
        """,  # noqa: S608
        (session_id, extractor_version),
    ).mappings()
    return tuple(
        IfgActionFileEdgeRow(
            edge_id=str(row["edge_id"]),
            session_id=str(row["session_id"]),
            action_id=str(row["action_id"]),
            path=str(row["path"]),
            relation=str(row["relation"]),
            turn=int(row["turn"]),
            event_id=_int_or_none(row["event_id"]),
            source=str(row["source"]),
            confidence=str(row["confidence"]),
            is_anchor=bool(row["is_anchor"]),
            extractor_version=str(row["extractor_version"]),
            content_hash=_str_or_none(row["content_hash"]),
            before_hash=_str_or_none(row["before_hash"]),
            after_hash=_str_or_none(row["after_hash"]),
            content_state=_str_or_none(row["content_state"]),
            line_range=_mapping(row["line_range_json"]),
            span=_mapping(row["span_json"]),
            metadata=_mapping(row["metadata_json"]),
            raw_evidence=_mapping(row["raw_evidence_json"]),
        )
        for row in rows
    )


def read_path_candidates(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> tuple[IfgPathCandidateRow, ...]:
    rows = conn.exec_driver_sql(
        """
        SELECT *
        FROM ifg_path_candidates
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY turn, event_id, normalized_path
        """,
        (session_id, extractor_version),
    ).mappings()
    return tuple(
        IfgPathCandidateRow(
            candidate_id=str(row["candidate_id"]),
            session_id=str(row["session_id"]),
            extractor_version=str(row["extractor_version"]),
            action_id=str(row["action_id"]),
            source_unit_id=_str_or_none(row["source_unit_id"]),
            path_text=str(row["path_text"]),
            normalized_path=str(row["normalized_path"]),
            path_kind=str(row["path_kind"]),
            relation=str(row["relation"]),
            turn=int(row["turn"]),
            event_id=_int_or_none(row["event_id"]),
            source=str(row["source"]),
            confidence=str(row["confidence"]),
            metadata=_mapping(row["metadata_json"]),
            raw_evidence=_mapping(row["raw_evidence_json"]),
        )
        for row in rows
    )


def read_source_units(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> tuple[IfgSourceUnitRow, ...]:
    rows = conn.exec_driver_sql(
        """
        SELECT *
        FROM ifg_source_units
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY turn, event_id, action_id, kind
        """,
        (session_id, extractor_version),
    ).mappings()
    return tuple(
        IfgSourceUnitRow(
            source_unit_id=str(row["source_unit_id"]),
            session_id=str(row["session_id"]),
            extractor_version=str(row["extractor_version"]),
            action_id=str(row["action_id"]),
            kind=str(row["kind"]),
            origin=str(row["origin"]),
            path=_str_or_none(row["path"]),
            relation=str(row["relation"]),
            turn=int(row["turn"]),
            event_id=_int_or_none(row["event_id"]),
            tool_name=str(row["tool_name"]),
            language=_str_or_none(row["language"]),
            content_hash=_str_or_none(row["content_hash"]),
            previous_content_hash=_str_or_none(row["previous_content_hash"]),
            result_content_hash=_str_or_none(row["result_content_hash"]),
            unit_hash=_str_or_none(row["unit_hash"]),
            content_state=str(row["content_state"]),
            line_range=_mapping(row["line_range_json"]),
            span=_mapping(row["span_json"]),
            content_text=_str_or_none(row["content_text"]),
            metadata=_mapping(row["metadata_json"]),
            raw_evidence=_mapping(row["raw_evidence_json"]),
        )
        for row in rows
    )


def read_symbol_mentions(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> tuple[IfgSymbolMentionRow, ...]:
    rows = conn.exec_driver_sql(
        """
        SELECT *
        FROM ifg_symbol_mentions
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY turn, event_id, symbol_text
        """,
        (session_id, extractor_version),
    ).mappings()
    return tuple(
        IfgSymbolMentionRow(
            mention_id=str(row["mention_id"]),
            session_id=str(row["session_id"]),
            extractor_version=str(row["extractor_version"]),
            action_id=str(row["action_id"]),
            source_unit_id=_str_or_none(row["source_unit_id"]),
            symbol_text=str(row["symbol_text"]),
            turn=int(row["turn"]),
            event_id=_int_or_none(row["event_id"]),
            source=str(row["source"]),
            confidence=str(row["confidence"]),
            path=_str_or_none(row["path"]),
            metadata=_mapping(row["metadata_json"]),
            raw_evidence=_mapping(row["raw_evidence_json"]),
        )
        for row in rows
    )


def _mapping(raw: object) -> Mapping[str, object]:
    value = _loads(raw)
    return value if isinstance(value, Mapping) else {}


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
