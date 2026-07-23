# code-health: ignore-file[AM025] -- policy engine normalizes untyped YAML, SQLite, and runtime event payloads
"""Policy-owned trace view provider for the shared AgentM Textual viewer."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from sqlalchemy.engine import Connection, RowMapping

from policy_engine.ifg.schema import IFG_EXTRACTOR_VERSION
from policy_engine.paths import default_policy_db_path, resolve_policy_db_path
from policy_engine.source_parser import (
    BashRedirect,
    BashSegment,
    parse_bash_segments,
)
from policy_engine.source_semantics import analyze_bash_segment
from policy_engine.trace_indicators import load_policy_indicator_rows

from agentm.trajectory_view import (
    TraceQuery,
    TraceRow,
    TraceSnapshot,
    TraceTableColumn,
    TraceView,
    TraceViewRegistry,
    TraceViewSpec,
    build_trace_view_registry,
    filter_trace_rows,
    run_textual_viewer,
)
from agentm.core.abi.query import TrajectoryQueryStore
from agentm.storage.sql import create_sqlite_engine

_POLICY_TABLES = (
    "event_log",
    "policy_tool_events",
    "policy_file_state",
    "policy_entity_state",
    "policy_context_state",
    "policy_turn_summary",
    "policy_session_summary",
    "policy_eval_error",
)
_IFG_TABLES = (
    "ifg_normalized_tool_events",
    "ifg_actions",
    "ifg_files",
    "ifg_action_file_edges",
    "ifg_source_units",
    "ifg_path_candidates",
    "ifg_symbol_mentions",
    "ifg_symbols",
    "ifg_action_symbol_edges",
    "ifg_file_symbol_edges",
    "ifg_symbol_symbol_edges",
    "ifg_nodes",
    "ifg_edges",
    "ifg_extraction_error",
    "ifg_session_summary",
)

_SUMMARY_CATEGORIES = frozenset({"summary"})
_EFFECT_CATEGORIES = frozenset({"effects"})
_TOOL_CATEGORIES = frozenset({"tools"})
_FILE_CATEGORIES = frozenset({"files"})
_FILE_STREAM_CATEGORIES = frozenset({"file_stream"})
_FILE_TOOL_STREAM_CATEGORIES = frozenset({"file_tool_stream"})
_BASH_FILE_STREAM_CATEGORIES = frozenset({"bash_file_stream"})
_BASH_COMMAND_CATEGORIES = frozenset({"bash_commands"})
_IFG_ACTION_CATEGORIES = frozenset({"ifg_actions"})
_IFG_SOURCE_UNIT_CATEGORIES = frozenset({"ifg_source_units"})
_IFG_FILE_CATEGORIES = frozenset({"ifg_files"})
_IFG_SYMBOL_CATEGORIES = frozenset({"ifg_symbols"})
_ENTITY_CATEGORIES = frozenset({"entities"})
_ERROR_CATEGORIES = frozenset({"errors"})

_FILE_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("name", "File", max_width=60),
    TraceTableColumn("reads", "Reads", max_width=6),
    TraceTableColumn("writes", "Writes", max_width=6),
    TraceTableColumn("refs", "Refs", max_width=5),
    TraceTableColumn("events", "Events", max_width=6),
    TraceTableColumn("latest", "Latest", max_width=34),
    TraceTableColumn("preview", "Preview", max_width=56),
)
_FILE_STREAM_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("name", "File", max_width=58),
    TraceTableColumn("operation", "Operation", max_width=9),
    TraceTableColumn("phase", "Phase", max_width=9),
    TraceTableColumn("file_result", "File Result", max_width=12),
    TraceTableColumn("tool_result", "Tool Result", max_width=18),
    TraceTableColumn("source", "Source", max_width=22),
    TraceTableColumn("tool", "Tool", max_width=16),
    TraceTableColumn("preview", "Preview", max_width=56),
)
_BASH_FILE_STREAM_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("name", "File", max_width=58),
    TraceTableColumn("operation", "Operation", max_width=9),
    TraceTableColumn("evidence", "Evidence", max_width=12),
    TraceTableColumn("phase", "Phase", max_width=9),
    TraceTableColumn("tool_result", "Tool Result", max_width=18),
    TraceTableColumn("source", "Source", max_width=22),
    TraceTableColumn("command", "Command", max_width=72),
)
_BASH_COMMAND_COLUMNS = (
    TraceTableColumn("count", "Count", max_width=7),
    TraceTableColumn("family", "Family", max_width=10),
    TraceTableColumn("command", "Command", max_width=14),
    TraceTableColumn("template", "Template", max_width=82),
    TraceTableColumn("errors", "Tool Err", max_width=8),
    TraceTableColumn("first_turn", "First", max_width=7),
    TraceTableColumn("last_turn", "Last", max_width=7),
    TraceTableColumn("example", "Example", max_width=72),
)
_IFG_ACTION_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("action_kind", "Action", max_width=10),
    TraceTableColumn("family", "Family", max_width=10),
    TraceTableColumn("command", "Command", max_width=14),
    TraceTableColumn("template", "Template", max_width=82),
    TraceTableColumn("confidence", "Conf", max_width=7),
    TraceTableColumn("source", "Source", max_width=22),
    TraceTableColumn("tool", "Tool", max_width=14),
)
_IFG_FILE_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("name", "File", max_width=38),
    TraceTableColumn("reads", "Reads", max_width=6),
    TraceTableColumn("writes", "Writes", max_width=6),
    TraceTableColumn("refs", "Refs", max_width=5),
    TraceTableColumn("evidence", "Evidence", max_width=9),
    TraceTableColumn("existence", "Exists", max_width=11),
    TraceTableColumn("confidence", "Conf", max_width=5),
)
_IFG_SOURCE_UNIT_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("source_kind", "Kind", max_width=18),
    TraceTableColumn("relation", "Rel", max_width=8),
    TraceTableColumn("path", "Path", max_width=58),
    TraceTableColumn("origin", "Origin", max_width=16),
    TraceTableColumn("content_state", "State", max_width=18),
    TraceTableColumn("preview", "Preview", max_width=72),
)
_IFG_SYMBOL_COLUMNS = (
    TraceTableColumn("location", "Loc", max_width=8),
    TraceTableColumn("symbol_kind", "Kind", max_width=10),
    TraceTableColumn("name", "Symbol", max_width=24),
    TraceTableColumn("file", "File", max_width=34),
    TraceTableColumn("validation_state", "Valid", max_width=11),
    TraceTableColumn("confidence", "Conf", max_width=6),
    TraceTableColumn("observations", "Obs", max_width=4),
)

_STRUCTURED_FILE_TOOL_NAMES = frozenset({"read", "write", "edit"})
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]{2,}\b")


@dataclass(frozen=True, slots=True)
class _FileEvent:
    event_id: int
    ts: float
    turn: int
    phase: str
    tool_name: str
    operation: str
    path: str
    source: str
    args_hash: str | None
    file_status: str
    tool_status: str
    tool_exit_code: int | None
    duration_ms: int | None
    error_category: str | None
    error_fingerprint: str | None
    result_content_hash: str | None
    content_hash: str | None
    previous_content_hash: str | None
    policy_tool_event: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _MergedFileEvent:
    event: _FileEvent
    phases: tuple[str, ...]
    sources: tuple[str, ...]
    events: tuple[_FileEvent, ...]

    @property
    def phase_label(self) -> str:
        return "+".join(self.phases)

    @property
    def source_label(self) -> str:
        return "+".join(self.sources)


@dataclass(frozen=True, slots=True)
class PolicyTraceViewProvider:
    """Register policy projection rows into the shared trace viewer."""

    db_path: Path | None = None
    cwd: Path | None = None
    ifg_extractor_version: str = IFG_EXTRACTOR_VERSION

    def trace_view_specs(self) -> Sequence[TraceViewSpec]:
        return (
            TraceViewSpec(
                id="policy",
                title="Policy",
                description="Policy projection overview for this trajectory.",
                shortcut="5",
                build=lambda snapshot, query: self._build_view(
                    "policy",
                    "Policy",
                    snapshot,
                    query,
                    categories=_SUMMARY_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-effects",
                title="Effects",
                description="Persisted policy rule firings.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-effects",
                    "Effects",
                    snapshot,
                    query,
                    categories=_EFFECT_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-ifg-actions",
                title="Actions",
                description="Information-flow actions extracted from tool events.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-ifg-actions",
                    "Actions",
                    snapshot,
                    query,
                    categories=_IFG_ACTION_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-ifg-source-units",
                title="Source Units",
                description="Source fragments linking tool I/O to code symbols.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-ifg-source-units",
                    "Source Units",
                    snapshot,
                    query,
                    categories=_IFG_SOURCE_UNIT_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-ifg-files",
                title="Files",
                description="Information-flow file nodes and relation counts.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-ifg-files",
                    "Files",
                    snapshot,
                    query,
                    categories=_IFG_FILE_CATEGORIES,
                ),
            ),
            TraceViewSpec(
                id="policy-ifg-symbols",
                title="Symbols",
                description="Code symbols and unresolved symbol mentions in the IFG.",
                shortcut="",
                build=lambda snapshot, query: self._build_view(
                    "policy-ifg-symbols",
                    "Symbols",
                    snapshot,
                    query,
                    categories=_IFG_SYMBOL_CATEGORIES,
                ),
            ),
        )

    def _build_view(
        self,
        view_id: str,
        title: str,
        snapshot: TraceSnapshot,
        query: TraceQuery,
        *,
        categories: frozenset[str],
    ) -> TraceView:
        db_path = self.db_path or resolve_policy_db_path(
            session_id=snapshot.session_id,
            cwd=self.cwd,
        )
        rows = load_policy_trace_rows(
            db_path,
            snapshot.session_id,
            categories=categories,
            ifg_extractor_version=self.ifg_extractor_version,
        )
        filtered = filter_trace_rows(rows, query)
        return TraceView(
            id=view_id,
            title=title,
            rows=filtered,
            summary=_summary_text(db_path, rows, filtered),
            empty_text="No policy rows match the current query.",
            columns=_columns_for_categories(categories),
        )


def _columns_for_categories(
    categories: frozenset[str],
) -> tuple[TraceTableColumn, ...]:
    if categories == _FILE_CATEGORIES:
        return _FILE_COLUMNS
    if categories == _FILE_STREAM_CATEGORIES:
        return _FILE_STREAM_COLUMNS
    if categories == _FILE_TOOL_STREAM_CATEGORIES:
        return _FILE_STREAM_COLUMNS
    if categories == _BASH_FILE_STREAM_CATEGORIES:
        return _BASH_FILE_STREAM_COLUMNS
    if categories == _BASH_COMMAND_CATEGORIES:
        return _BASH_COMMAND_COLUMNS
    if categories == _IFG_ACTION_CATEGORIES:
        return _IFG_ACTION_COLUMNS
    if categories == _IFG_SOURCE_UNIT_CATEGORIES:
        return _IFG_SOURCE_UNIT_COLUMNS
    if categories == _IFG_FILE_CATEGORIES:
        return _IFG_FILE_COLUMNS
    if categories == _IFG_SYMBOL_CATEGORIES:
        return _IFG_SYMBOL_COLUMNS
    return ()


def build_policy_trace_view_registry(
    db_path: Path | None = None,
    *,
    cwd: Path | None = None,
) -> TraceViewRegistry:
    """Build AgentM's shared trajectory app registry with policy tabs."""

    return build_trace_view_registry(
        (PolicyTraceViewProvider(db_path=db_path, cwd=cwd),)
    )


def run_policy_trace_viewer(
    query: TrajectoryQueryStore,
    session_id: str,
    *,
    db_path: Path | None = None,
    cwd: Path | None = None,
    follow: bool = False,
) -> None:
    """Run AgentM's shared Textual trace viewer with policy projection tabs."""

    run_textual_viewer(
        query,
        session_id,
        follow=follow,
        registry=build_policy_trace_view_registry(db_path, cwd=cwd),
    )


def load_policy_trace_rows(
    db_path: Path,
    session_id: str,
    *,
    categories: frozenset[str],
    ifg_extractor_version: str = IFG_EXTRACTOR_VERSION,
) -> tuple[TraceRow, ...]:
    if not db_path.exists():
        return (
            _diagnostic_row(
                session_id,
                title="Policy DB Missing",
                preview=str(db_path),
                content=(
                    "No persisted policy database was found.\n\n"
                    f"db_path: {db_path}\n"
                    "Run policy_engine backfill or start a session with the "
                    "policy extension enabled first."
                ),
            ),
        )

    engine = create_sqlite_engine(db_path)
    try:
        with engine.connect() as conn:
            existing = _existing_tables(conn)
            if not existing.intersection((*_POLICY_TABLES, *_IFG_TABLES)):
                return (
                    _diagnostic_row(
                        session_id,
                        title="Policy Tables Missing",
                        preview=str(db_path),
                        content=f"No policy projection tables were found in {db_path}",
                    ),
                )
            counts = _session_counts(
                conn,
                session_id,
                existing,
                ifg_extractor_version=ifg_extractor_version,
            )
            rows: list[TraceRow] = []
            if "summary" in categories:
                rows.extend(
                    _summary_rows(
                        conn,
                        session_id,
                        db_path,
                        existing,
                        counts,
                        ifg_extractor_version=ifg_extractor_version,
                    )
                )
            if "effects" in categories and "event_log" in existing:
                rows.extend(_effect_rows(conn, session_id))
            if "tools" in categories and "policy_tool_events" in existing:
                rows.extend(_tool_rows(conn, session_id))
            if "files" in categories and "policy_file_state" in existing:
                rows.extend(_file_rows(conn, session_id))
            if "file_stream" in categories and "policy_tool_events" in existing:
                rows.extend(
                    _file_stream_rows(
                        conn,
                        session_id,
                        include_bash=False,
                        category="file_stream",
                    )
                )
            if "file_tool_stream" in categories and "policy_tool_events" in existing:
                rows.extend(
                    _file_stream_rows(
                        conn,
                        session_id,
                        include_bash=False,
                        category="file_tool_stream",
                        tool_names=_STRUCTURED_FILE_TOOL_NAMES,
                    )
                )
            if "bash_file_stream" in categories and "policy_tool_events" in existing:
                rows.extend(
                    _file_stream_rows(
                        conn,
                        session_id,
                        include_bash=True,
                        category="bash_file_stream",
                    )
                )
            if "bash_commands" in categories and "policy_tool_events" in existing:
                rows.extend(_bash_command_rows(conn, session_id))
            if "ifg_actions" in categories and "ifg_actions" in existing:
                rows.extend(_ifg_action_rows(conn, session_id, ifg_extractor_version))
            if "ifg_source_units" in categories and "ifg_source_units" in existing:
                rows.extend(
                    _ifg_source_unit_rows(conn, session_id, ifg_extractor_version)
                )
            if "ifg_files" in categories and "ifg_files" in existing:
                rows.extend(_ifg_file_rows(conn, session_id, ifg_extractor_version))
            if "ifg_symbols" in categories and "ifg_symbols" in existing:
                rows.extend(_ifg_symbol_rows(conn, session_id, ifg_extractor_version))
            if "entities" in categories and "policy_entity_state" in existing:
                rows.extend(_entity_rows(conn, session_id))
            if "errors" in categories and "policy_eval_error" in existing:
                rows.extend(_eval_error_rows(conn, session_id))
            if not rows:
                rows.append(
                    _empty_category_row(session_id, db_path, counts, categories)
                )
            return tuple(rows)
    finally:
        engine.dispose()


def _rows(
    conn: Connection,
    sql: str,
    params: Sequence[object] = (),
) -> list[RowMapping]:
    return list(conn.exec_driver_sql(sql, tuple(params)).mappings().all())


def _row(
    conn: Connection,
    sql: str,
    params: Sequence[object] = (),
) -> RowMapping | None:
    return conn.exec_driver_sql(sql, tuple(params)).mappings().fetchone()


def _summary_rows(
    conn: Connection,
    session_id: str,
    db_path: Path,
    existing: set[str],
    counts: Mapping[str, int],
    *,
    ifg_extractor_version: str,
) -> list[TraceRow]:
    rows = [
        TraceRow(
            key=f"policy:summary:counts:{session_id}",
            kind="policy",
            title="Projection Counts",
            preview=_counts_preview(counts),
            content=_json_content(
                {
                    "session_id": session_id,
                    "db_path": str(db_path),
                    "ifg_extractor_version": ifg_extractor_version,
                    "counts": dict(counts),
                }
            ),
            metadata={
                "category": "summary",
                "db_path": str(db_path),
                "ifg_extractor_version": ifg_extractor_version,
            },
        )
    ]

    if "policy_session_summary" in existing:
        row = _row(
            conn,
            """
            SELECT updated_at, summary_json
            FROM policy_session_summary
            WHERE session_id = ?
            """,
            (session_id,),
        )
        if row is not None:
            summary = _loads(row["summary_json"])
            rows.extend(
                load_policy_indicator_rows(
                    conn,
                    session_id,
                    existing,
                    summary,
                )
            )
            rows.append(
                TraceRow(
                    key=f"policy:summary:session:{session_id}",
                    kind="policy",
                    title="Session Summary",
                    preview=_json_preview(summary),
                    content=_json_content(
                        {
                            "session_id": session_id,
                            "updated_at": row["updated_at"],
                            "summary": summary,
                        }
                    ),
                    metadata={"category": "summary"},
                )
            )

    if "ifg_session_summary" in existing:
        row = _row(
            conn,
            """
            SELECT updated_at, summary_json
            FROM ifg_session_summary
            WHERE session_id = ? AND extractor_version = ?
            """,
            (session_id, ifg_extractor_version),
        )
        if row is not None:
            summary = _loads(row["summary_json"])
            rows.append(
                TraceRow(
                    key=f"policy:summary:ifg:{session_id}:{ifg_extractor_version}",
                    kind="policy",
                    title="IFG Model Summary",
                    preview=_json_preview(summary),
                    content=_json_content(
                        {
                            "session_id": session_id,
                            "extractor_version": ifg_extractor_version,
                            "updated_at": row["updated_at"],
                            "summary": summary,
                        }
                    ),
                    metadata={
                        "category": "summary",
                        "extractor_version": ifg_extractor_version,
                    },
                )
            )

    if "event_log" in existing and counts.get("event_log", 0):
        rows.append(_aggregate_row(conn, session_id, "effects"))
    if "policy_tool_events" in existing and counts.get("policy_tool_events", 0):
        rows.append(_aggregate_row(conn, session_id, "tools"))
    if "policy_entity_state" in existing and counts.get("policy_entity_state", 0):
        rows.append(_aggregate_row(conn, session_id, "entities"))
    if "policy_file_state" in existing and counts.get("policy_file_state", 0):
        rows.append(_aggregate_row(conn, session_id, "file_hotspots"))

    if "policy_turn_summary" in existing:
        for row in _rows(
            conn,
            """
            SELECT turn_index, updated_at, summary_json
            FROM policy_turn_summary
            WHERE session_id = ?
            ORDER BY turn_index ASC
            """,
            (session_id,),
        ):
            summary = _loads(row["summary_json"])
            rows.append(
                TraceRow(
                    key=f"policy:summary:turn:{session_id}:{row['turn_index']}",
                    kind="policy",
                    title=f"Turn Summary T{row['turn_index']}",
                    preview=_turn_summary_preview(summary),
                    content=_json_content(
                        {
                            "session_id": session_id,
                            "turn_index": row["turn_index"],
                            "updated_at": row["updated_at"],
                            "summary": summary,
                        }
                    ),
                    turn_index=row["turn_index"],
                    metadata={"category": "summary"},
                )
            )

    return rows


def _aggregate_row(
    conn: Connection,
    session_id: str,
    category: str,
) -> TraceRow:
    if category == "effects":
        data = [
            dict(row)
            for row in _rows(
                conn,
                """
                SELECT rule_id, mode, effect, COUNT(*) AS count
                FROM event_log
                WHERE session_id = ?
                GROUP BY rule_id, mode, effect
                ORDER BY count DESC, rule_id ASC
                """,
                (session_id,),
            )
        ]
        return TraceRow(
            key=f"policy:aggregate:effects:{session_id}",
            kind="policy",
            title="Effects by Rule",
            preview=_json_preview(data),
            content=_json_content(data),
            metadata={"category": "summary", "aggregate": "effects"},
        )
    if category == "tools":
        data = [
            dict(row)
            for row in _rows(
                conn,
                """
                SELECT tool_name, phase, COUNT(*) AS count
                FROM policy_tool_events
                WHERE session_id = ?
                GROUP BY tool_name, phase
                ORDER BY count DESC, tool_name ASC, phase ASC
                """,
                (session_id,),
            )
        ]
        return TraceRow(
            key=f"policy:aggregate:tools:{session_id}",
            kind="policy",
            title="Tool Events by Tool",
            preview=_json_preview(data),
            content=_json_content(data),
            metadata={"category": "summary", "aggregate": "tools"},
        )
    if category == "entities":
        data = [
            dict(row)
            for row in _rows(
                conn,
                """
                SELECT entity_type, COUNT(*) AS count,
                       SUM(occurrence_count) AS occurrences
                FROM policy_entity_state
                WHERE session_id = ?
                GROUP BY entity_type
                ORDER BY count DESC, entity_type ASC
                """,
                (session_id,),
            )
        ]
        return TraceRow(
            key=f"policy:aggregate:entities:{session_id}",
            kind="policy",
            title="Entities by Type",
            preview=_json_preview(data),
            content=_json_content(data),
            metadata={"category": "summary", "aggregate": "entities"},
        )

    data = [
        dict(row)
        for row in _rows(
            conn,
            """
            SELECT path, read_count, write_count, first_read_turn,
                   last_read_turn, last_write_turn, content_hash
            FROM policy_file_state
            WHERE session_id = ?
            ORDER BY (read_count + write_count) DESC, path ASC
            LIMIT 50
            """,
            (session_id,),
        )
    ]
    return TraceRow(
        key=f"policy:aggregate:file_hotspots:{session_id}",
        kind="policy",
        title="File Hotspots",
        preview=_json_preview(data),
        content=_json_content(data),
        metadata={"category": "summary", "aggregate": "file_hotspots"},
    )


def _effect_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM event_log
        WHERE session_id = ?
        ORDER BY ts ASC, id ASC
        """,
        (session_id,),
    ):
        context = _loads(row["context_json"])
        state = f"{row['mode'] or '-'}:{row['effect'] or '-'}"
        yield TraceRow(
            key=f"policy:effect:{row['id']}",
            kind="policy",
            title=row["rule_id"] or "Policy Effect",
            preview=row["reason"] or _json_preview(context),
            content=_json_content(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "session_id": row["session_id"],
                    "rule_id": row["rule_id"],
                    "mode": row["mode"],
                    "effect": row["effect"],
                    "reason": row["reason"],
                    "turn": row["turn"],
                    "context": context,
                }
            ),
            turn_index=row["turn"],
            cause=row["effect"],
            is_error=row["mode"] == "enforce" or row["effect"] in {"block", "deny"},
            metadata={"category": "effect", "rule_id": row["rule_id"], "state": state},
        )


def _tool_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_tool_events
        WHERE session_id = ?
        ORDER BY turn ASC, id ASC
        """,
        (session_id,),
    ):
        args = _loads(row["args_json"])
        result = _loads(row["result_json"])
        state = _loads(row["state_json"])
        processed = _loads(row["processed_json"])
        error = _tool_is_error(row, result, state, processed)
        display_name = f"{row['phase']} {row['tool_name']}"
        state_label = _tool_policy_state_label(row, state, processed, error)
        preview = _tool_policy_preview(row, args, result, state, processed)
        yield TraceRow(
            key=f"policy:tool:{row['id']}",
            kind="policy",
            title=display_name,
            preview=preview,
            content=_json_content(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "session_id": row["session_id"],
                    "turn": row["turn"],
                    "phase": row["phase"],
                    "tool_call_id": row["tool_call_id"],
                    "tool_name": row["tool_name"],
                    "args_hash": row["args_hash"],
                    "args": args,
                    "result": result,
                    "state": state,
                    "processed": processed,
                    "exit_code": row["exit_code"],
                    "duration_ms": row["duration_ms"],
                    "result_content_hash": row["result_content_hash"],
                    "policy_display": {
                        "name": display_name,
                        "state": state_label,
                        "preview": preview,
                    },
                    "rule_author_event": _tool_rule_author_event(
                        row,
                        args,
                        result,
                        state,
                        processed,
                    ),
                }
            ),
            turn_index=row["turn"],
            tool_name=row["tool_name"],
            display_name=display_name,
            is_error=error,
            cause=state_label,
            metadata={
                "category": "tool",
                "phase": row["phase"],
                "tool_call_id": row["tool_call_id"],
                "args_hash": row["args_hash"],
                "exit_code": row["exit_code"],
                "duration_ms": row["duration_ms"],
                "error_category": _tool_error_category(state, processed),
                "error_fingerprint": _tool_error_fingerprint(state, processed),
            },
        )


def _file_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    events_by_path = _file_events_by_path(
        _merge_file_events(_file_events(conn, session_id))
    )
    state_paths: set[str] = set()
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_file_state
        WHERE session_id = ?
        ORDER BY (read_count + write_count) DESC, path ASC
        """,
        (session_id,),
    ):
        state_paths.add(row["path"])
        history = events_by_path.get(row["path"], [])
        last_turn = _last_turn(row["last_write_turn"], row["last_read_turn"])
        refs = sum(1 for event in history if event.event.operation == "reference")
        latest = _latest_file_event(history)
        content_signals = _file_content_signals(history)
        state = (
            f"r:{row['read_count']} w:{row['write_count']} ref:{refs} ev:{len(history)}"
        )
        yield TraceRow(
            key=f"policy:file:{row['path']}",
            kind="policy",
            title=row["path"],
            preview=(
                f"first:{_dash(row['first_read_turn'])} "
                f"last:{_dash(last_turn)} latest:{latest} "
                f"hash:{row['content_hash'] or '-'}"
            ),
            content=_json_content(
                {
                    "session_id": row["session_id"],
                    "path": row["path"],
                    "updated_at": row["updated_at"],
                    "first_read_turn": row["first_read_turn"],
                    "last_read_turn": row["last_read_turn"],
                    "last_write_turn": row["last_write_turn"],
                    "read_count": row["read_count"],
                    "write_count": row["write_count"],
                    "content_hash": row["content_hash"],
                    "reverts_to_prior_hash": bool(row["reverts_to_prior_hash"]),
                    "state": _loads(row["state_json"]),
                    "history": [_merged_file_event_json(event) for event in history],
                    "content_signals": content_signals,
                }
            ),
            turn_index=last_turn,
            cause=state,
            metadata={
                "category": "file",
                "path": row["path"],
                "state": state,
                "reads": row["read_count"],
                "writes": row["write_count"],
                "refs": refs,
                "events": len(history),
                "latest": latest,
            },
        )

    for path, history in sorted(events_by_path.items()):
        if path in state_paths:
            continue
        if not history:
            continue
        reads = sum(1 for event in history if event.event.operation == "read")
        writes = sum(1 for event in history if event.event.operation == "write")
        refs = sum(1 for event in history if event.event.operation == "reference")
        latest_event = history[-1]
        state = f"r:{reads} w:{writes} ref:{refs} ev:{len(history)}"
        latest_label = _latest_file_event(history)
        content_signals = _file_content_signals(history)
        yield TraceRow(
            key=f"policy:file:derived:{path}",
            kind="policy",
            title=path,
            preview=f"latest:{latest_label}",
            content=_json_content(
                {
                    "session_id": session_id,
                    "path": path,
                    "state": "derived from policy_tool_events",
                    "history": [_merged_file_event_json(event) for event in history],
                    "content_signals": content_signals,
                }
            ),
            turn_index=latest_event.event.turn,
            cause=state,
            metadata={
                "category": "file",
                "path": path,
                "state": state,
                "reads": reads,
                "writes": writes,
                "refs": refs,
                "events": len(history),
                "latest": latest_label,
            },
        )


def _file_stream_rows(
    conn: Connection,
    session_id: str,
    *,
    include_bash: bool,
    category: str,
    tool_names: frozenset[str] | None = None,
) -> Iterable[TraceRow]:
    for merged in _merge_file_events(_file_events(conn, session_id)):
        event = merged.event
        is_bash = event.tool_name == "bash"
        if include_bash != is_bash:
            continue
        if tool_names is not None and event.tool_name not in tool_names:
            continue
        is_error = event.tool_status.startswith(("error", "exit:"))
        tool_call_id = event.policy_tool_event.get("tool_call_id")
        phase = merged.phase_label
        source = merged.source_label
        command = _single_line(
            _command_text(event.policy_tool_event.get("args")),
            limit=240,
        )
        evidence = _file_evidence_label(event, source)
        yield TraceRow(
            key=(
                f"policy:{category}:{event.turn}:{tool_call_id}:"
                f"{event.operation}:{event.path}:{phase}"
            ),
            kind="policy",
            title=event.path,
            preview=(
                f"hash:{_dash(_short_text(event.args_hash, 8))} "
                f"call:{_dash(tool_call_id)} raw:{len(merged.events)}"
            ),
            content=_json_content(
                {
                    "session_id": session_id,
                    "file_event": _merged_file_event_json(merged),
                    "raw_file_events": [
                        _file_event_json(raw_event) for raw_event in merged.events
                    ],
                    "policy_tool_event": event.policy_tool_event,
                }
            ),
            turn_index=event.turn,
            tool_name=event.tool_name,
            display_name=event.path,
            is_error=is_error,
            cause=event.tool_status,
            metadata={
                "category": category,
                "path": event.path,
                "operation": event.operation,
                "phase": phase,
                "result": event.tool_status,
                "file_result": event.file_status,
                "tool_result": event.tool_status,
                "evidence": evidence,
                "source": source,
                "tool": event.tool_name,
                "tool_call_id": tool_call_id,
                "tool_exit_code": event.tool_exit_code,
                "duration_ms": event.duration_ms,
                "command": command,
                "args_hash": event.args_hash,
                "raw_event_count": len(merged.events),
                "error_category": event.error_category,
                "error_fingerprint": event.error_fingerprint,
            },
        )


def _bash_command_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    clusters: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in _preferred_bash_tool_rows(conn, session_id):
        for segment in _bash_command_segments(row):
            clusters[str(segment["template"])].append(segment)

    for template, segments in sorted(
        clusters.items(),
        key=lambda item: (-len(item[1]), str(item[0])),
    ):
        first = min(segments, key=lambda item: (item["turn"], item["event_id"]))
        last = max(segments, key=lambda item: (item["turn"], item["event_id"]))
        errors = sum(1 for item in segments if item["is_error"])
        families = Counter(str(item["family"]) for item in segments)
        commands = Counter(str(item["command"]) for item in segments)
        family = families.most_common(1)[0][0]
        command = commands.most_common(1)[0][0]
        examples = _unique_command_examples(segments)
        yield TraceRow(
            key=f"policy:bash_command:{_stable_id(template)}",
            kind="policy",
            title=template,
            preview=(
                f"count:{len(segments)} errors:{errors} "
                f"family:{family} example:{_single_line(examples[0], limit=160)}"
            ),
            content=_json_content(
                {
                    "session_id": session_id,
                    "template": template,
                    "template_tokens": first["template_tokens"],
                    "count": len(segments),
                    "errors": errors,
                    "family_counts": dict(families),
                    "command_counts": dict(commands),
                    "first_turn": first["turn"],
                    "last_turn": last["turn"],
                    "examples": examples[:8],
                    "segments": segments[:200],
                }
            ),
            turn_index=last["turn"] if isinstance(last["turn"], int) else None,
            tool_name="bash",
            display_name=template,
            is_error=errors > 0,
            cause=family,
            metadata={
                "category": "bash_commands",
                "count": len(segments),
                "family": family,
                "command": command,
                "template": template,
                "errors": errors,
                "first_turn": first["turn"],
                "last_turn": last["turn"],
                "example": _single_line(examples[0], limit=240),
            },
        )


def _ifg_action_rows(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM ifg_actions
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY turn ASC, event_id ASC, segment_index ASC, action_id ASC
        """,
        (session_id, extractor_version),
    ):
        raw_evidence = _loads(row["raw_evidence_json"])
        template = row["template"] or row["command"] or row["action_kind"]
        yield TraceRow(
            key=f"ifg:action:{row['action_id']}",
            kind="policy",
            title=str(template),
            preview=(
                f"{row['action_kind']} {row['confidence']} {row['source']} "
                f"tool:{row['tool_name']}"
            ),
            content=_json_content(
                {
                    "session_id": row["session_id"],
                    "action_id": row["action_id"],
                    "turn": row["turn"],
                    "event_id": row["event_id"],
                    "tool_call_id": row["tool_call_id"],
                    "tool_name": row["tool_name"],
                    "segment_index": row["segment_index"],
                    "command": row["command"],
                    "action_kind": row["action_kind"],
                    "family": row["family"],
                    "template": row["template"],
                    "source": row["source"],
                    "confidence": row["confidence"],
                    "extractor_version": row["extractor_version"],
                    "raw_evidence": raw_evidence,
                }
            ),
            turn_index=row["turn"],
            tool_name=row["tool_name"],
            display_name=str(template),
            cause=row["action_kind"],
            metadata={
                "category": "ifg_action",
                "action_kind": row["action_kind"],
                "family": row["family"],
                "command": row["command"] or "",
                "template": row["template"] or "",
                "confidence": row["confidence"],
                "source": row["source"],
                "tool": row["tool_name"],
                "extractor_version": row["extractor_version"],
            },
        )


def _ifg_source_unit_rows(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM ifg_source_units
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY turn ASC, event_id ASC, kind ASC, source_unit_id ASC
        """,
        (session_id, extractor_version),
    ):
        metadata = _loads(row["metadata_json"])
        raw_evidence = _loads(row["raw_evidence_json"])
        line_range = _loads(row["line_range_json"])
        span = _loads(row["span_json"])
        text = row["content_text"] or ""
        path = row["path"] or ""
        target = path or row["origin"]
        preview = (
            f"{row['kind']} {row['relation']} {target} {_single_line(text, limit=140)}"
        ).strip()
        confidence = (
            metadata.get("confidence", "") if isinstance(metadata, Mapping) else ""
        )
        yield TraceRow(
            key=f"ifg:source-unit:{row['source_unit_id']}",
            kind="policy",
            title=f"{row['kind']} {target}",
            preview=preview,
            content=_json_content(
                {
                    "session_id": row["session_id"],
                    "source_unit_id": row["source_unit_id"],
                    "action_id": row["action_id"],
                    "turn": row["turn"],
                    "event_id": row["event_id"],
                    "tool_name": row["tool_name"],
                    "kind": row["kind"],
                    "origin": row["origin"],
                    "path": row["path"],
                    "relation": row["relation"],
                    "language": row["language"],
                    "content_hash": row["content_hash"],
                    "previous_content_hash": row["previous_content_hash"],
                    "result_content_hash": row["result_content_hash"],
                    "unit_hash": row["unit_hash"],
                    "content_state": row["content_state"],
                    "line_range": line_range,
                    "span": span,
                    "content_text": row["content_text"],
                    "extractor_version": row["extractor_version"],
                    "metadata": metadata,
                    "raw_evidence": raw_evidence,
                }
            ),
            turn_index=row["turn"],
            tool_name=row["tool_name"],
            display_name=str(target),
            cause=row["relation"],
            metadata={
                "category": "ifg_source_unit",
                "kind": row["kind"],
                "source_kind": row["kind"],
                "origin": row["origin"],
                "relation": row["relation"],
                "path": path,
                "state": row["content_state"],
                "content_state": row["content_state"],
                "confidence": str(confidence),
                "preview": _single_line(text, limit=240),
                "extractor_version": row["extractor_version"],
            },
        )


def _ifg_file_rows(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM ifg_files
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY last_seen_turn DESC, observation_count DESC, path ASC
        """,
        (session_id, extractor_version),
    ):
        metadata = _loads(row["metadata_json"])
        raw_evidence = _loads(row["raw_evidence_json"])
        relation_counts = (
            metadata.get("relation_counts", {}) if isinstance(metadata, Mapping) else {}
        )
        if not isinstance(relation_counts, Mapping):
            relation_counts = {}
        reads = _count_relations(relation_counts, "read")
        writes = sum(
            _count_relations(relation_counts, relation)
            for relation in ("write", "edit", "delete")
        )
        refs = _count_relations(relation_counts, "reference")
        anchor_count = _coerce_int(metadata.get("anchor_count")) or 0
        resolved_count = _coerce_int(metadata.get("resolved_candidate_count")) or 0
        evidence = f"{anchor_count}A+{resolved_count}R"
        existence_counts = metadata.get("existence_counts")
        existence = _ifg_existence_label(existence_counts)
        display_path = _path_tail(row["path"], parts=3)
        history = _ifg_file_history(
            conn, session_id, row["path"], row["extractor_version"]
        )
        preview = (
            f"r:{reads} w:{writes} ref:{refs} "
            f"obs:{row['observation_count']} last:T{row['last_seen_turn']}"
        )
        yield TraceRow(
            key=f"ifg:file:{row['extractor_version']}:{row['path']}",
            kind="policy",
            title=row["path"],
            preview=preview,
            content=_json_content(
                {
                    "session_id": row["session_id"],
                    "path": row["path"],
                    "first_seen_turn": row["first_seen_turn"],
                    "last_seen_turn": row["last_seen_turn"],
                    "observation_count": row["observation_count"],
                    "source": row["source"],
                    "confidence": row["confidence"],
                    "extractor_version": row["extractor_version"],
                    "metadata": metadata,
                    "raw_evidence": raw_evidence,
                    "history": history,
                }
            ),
            turn_index=row["last_seen_turn"],
            display_name=display_path,
            cause=preview,
            metadata={
                "category": "ifg_file",
                "name": display_path,
                "path": row["path"],
                "reads": reads,
                "writes": writes,
                "refs": refs,
                "evidence": evidence,
                "existence": existence,
                "confidence": row["confidence"],
                "preview": preview,
                "extractor_version": row["extractor_version"],
            },
        )


def _ifg_symbol_rows(
    conn: Connection,
    session_id: str,
    extractor_version: str,
) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM ifg_symbols
        WHERE session_id = ? AND extractor_version = ?
        ORDER BY last_seen_turn DESC, kind ASC, qualified_name ASC, symbol_id ASC
        """,
        (session_id, extractor_version),
    ):
        metadata = _loads(row["metadata_json"])
        raw_evidence = _loads(row["raw_evidence_json"])
        path = row["path"] or ""
        validation = metadata.get("validation", "unavailable")
        preview = (
            f"{row['kind']} {row['qualified_name']} {path} "
            f"obs:{row['observation_count']}"
        ).strip()
        yield TraceRow(
            key=f"ifg:symbol:{row['symbol_id']}",
            kind="policy",
            title=row["qualified_name"],
            preview=preview,
            content=_json_content(
                {
                    "session_id": row["session_id"],
                    "symbol_id": row["symbol_id"],
                    "kind": row["kind"],
                    "qualified_name": row["qualified_name"],
                    "path": row["path"],
                    "stable_key": row["stable_key"],
                    "first_seen_turn": row["first_seen_turn"],
                    "last_seen_turn": row["last_seen_turn"],
                    "observation_count": row["observation_count"],
                    "source": row["source"],
                    "confidence": row["confidence"],
                    "extractor_version": row["extractor_version"],
                    "metadata": metadata,
                    "raw_evidence": raw_evidence,
                }
            ),
            turn_index=row["last_seen_turn"],
            display_name=row["qualified_name"],
            cause=row["kind"],
            metadata={
                "category": "ifg_symbol",
                "kind": row["kind"],
                "symbol_kind": row["kind"],
                "name": row["qualified_name"],
                "path": path,
                "file": Path(path).name if path else "-",
                "source": row["source"],
                "validation": validation,
                "validation_state": (
                    "repo"
                    if validation == "repository_present"
                    else "observed"
                    if validation == "trajectory_observed"
                    else "unavailable"
                ),
                "confidence": row["confidence"],
                "observations": row["observation_count"],
                "preview": preview,
                "extractor_version": row["extractor_version"],
            },
        )


def _ifg_file_history(
    conn: Connection,
    session_id: str,
    path: str,
    extractor_version: str,
) -> list[dict[str, object]]:
    return [
        {
            "turn": row["turn"],
            "event_id": row["event_id"],
            "relation": row["relation"],
            "source": row["source"],
            "confidence": row["confidence"],
            "action_id": row["action_id"],
            "action_kind": row["action_kind"],
            "command": row["command"],
            "template": row["template"],
            "tool_name": row["tool_name"],
        }
        for row in _rows(
            conn,
            """
            SELECT e.*, a.action_kind, a.command, a.template, a.tool_name
            FROM ifg_action_file_edges e
            LEFT JOIN ifg_actions a ON a.action_id = e.action_id
            WHERE e.session_id = ?
              AND e.path = ?
              AND e.extractor_version = ?
            ORDER BY e.turn ASC, e.event_id ASC, e.edge_id ASC
            LIMIT 200
            """,
            (session_id, path, extractor_version),
        )
    ]


def _count_relations(relation_counts: Mapping[object, object], relation: str) -> int:
    value = relation_counts.get(relation)
    return _coerce_int(value) or 0


def _ifg_existence_label(value: object) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    states = {str(key) for key in value}
    if "present_now" in states:
        return "repo"
    observed = bool(states & {"observed_at_event", "present_after_event"})
    if observed and "unknown" in states:
        return "observed+?"
    if observed:
        return "observed"
    return "unknown"


def _path_tail(path: str, *, parts: int) -> str:
    path_parts = Path(path).parts
    if len(path_parts) <= parts:
        return path
    return "/".join(path_parts[-parts:])


def _preferred_bash_tool_rows(
    conn: Connection,
    session_id: str,
) -> list[RowMapping]:
    rows = _rows(
        conn,
        """
        SELECT *
        FROM policy_tool_events
        WHERE session_id = ? AND tool_name = 'bash'
        ORDER BY turn ASC, id ASC
        """,
        (session_id,),
    )
    by_call: dict[str, RowMapping] = {}
    order: list[str] = []
    for row in rows:
        key = str(row["tool_call_id"] or f"event:{row['id']}")
        if key not in by_call:
            order.append(key)
            by_call[key] = row
            continue
        if by_call[key]["phase"] != "post" and row["phase"] == "post":
            by_call[key] = row
    return [by_call[key] for key in order]


def _bash_command_segments(row: RowMapping) -> Iterable[dict[str, object]]:
    args = _loads(row["args_json"])
    cmd = _command_text(args)
    if not cmd:
        return ()
    result = _loads(row["result_json"])
    state = _loads(row["state_json"])
    processed = _loads(row["processed_json"])
    error = _tool_is_error(row, result, state, processed)
    tool_status = _file_tool_status(row, state, processed, error)
    segments: list[dict[str, object]] = []
    for index, segment in enumerate(parse_bash_segments(cmd)):
        command = list(segment.argv)
        if not command:
            continue
        analysis = analyze_bash_segment(segment)
        template_tokens = analysis.template_tokens
        command_text = _bash_segment_text(segment)
        segments.append(
            {
                "event_id": row["id"],
                "turn": row["turn"],
                "phase": row["phase"],
                "tool_call_id": row["tool_call_id"],
                "command_index": index,
                "command": analysis.command,
                "family": analysis.family,
                "action_kind": analysis.action_kind,
                "confidence": analysis.confidence,
                "tokens": list(command),
                "template_tokens": list(template_tokens),
                "template": analysis.template,
                "command_text": command_text,
                "full_command": _single_line(cmd, limit=600),
                "parser": segment.parser,
                "pipeline_index": segment.pipeline_index,
                "depth": segment.depth,
                "redirects": [
                    _redirect_json(redirect) for redirect in segment.redirects
                ],
                "tool_result": tool_status,
                "exit_code": _tool_exit_code(row, processed),
                "duration_ms": _coerce_int(row["duration_ms"]),
                "is_error": error,
            }
        )
    return segments


def _unique_command_examples(segments: Sequence[Mapping[str, object]]) -> list[str]:
    examples: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        text = _single_line(segment.get("command_text"), limit=600)
        if not text or text in seen:
            continue
        seen.add(text)
        examples.append(text)
    return examples or ["-"]


def _redirect_json(redirect: BashRedirect) -> dict[str, object]:
    return {
        "kind": redirect.kind,
        "operator": redirect.operator,
        "destination": redirect.destination,
        "descriptor": redirect.descriptor,
        "text": redirect.text,
    }


def _file_events(conn: Connection, session_id: str) -> list[_FileEvent]:
    events: list[_FileEvent] = []
    known_paths = _known_file_paths(conn, session_id)
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_tool_events
        WHERE session_id = ?
        ORDER BY turn ASC, id ASC
        """,
        (session_id,),
    ):
        args = _loads(row["args_json"])
        result = _loads(row["result_json"])
        state = _loads(row["state_json"])
        processed = _loads(row["processed_json"])
        error = _tool_is_error(row, result, state, processed)
        paths = _file_event_paths(
            row["tool_name"],
            args,
            processed,
            state,
            known_paths=known_paths,
        )
        if not paths:
            continue
        tool_status = _file_tool_status(row, state, processed, error)
        tool_exit_code = _tool_exit_code(row, processed)
        policy_tool_event = {
            "id": row["id"],
            "ts": row["ts"],
            "turn": row["turn"],
            "phase": row["phase"],
            "tool_call_id": row["tool_call_id"],
            "tool_name": row["tool_name"],
            "args_hash": row["args_hash"],
            "args": args,
            "result": result,
            "processed": processed,
            "state": state,
            "exit_code": row["exit_code"],
            "duration_ms": row["duration_ms"],
            "result_content_hash": row["result_content_hash"],
        }
        for path, source in paths:
            operation = _file_operation(row["tool_name"], args, processed, source)
            file_status = _file_operation_status(
                row["tool_name"],
                row["phase"],
                tool_status,
            )
            known_paths.add(path)
            events.append(
                _FileEvent(
                    event_id=row["id"],
                    ts=row["ts"],
                    turn=row["turn"],
                    phase=row["phase"],
                    tool_name=row["tool_name"],
                    operation=operation,
                    path=path,
                    source=source,
                    args_hash=row["args_hash"],
                    file_status=file_status,
                    tool_status=tool_status,
                    tool_exit_code=tool_exit_code,
                    duration_ms=_coerce_int(
                        row["duration_ms"]
                        if row["duration_ms"] is not None
                        else (
                            processed.get("duration_ms")
                            if isinstance(processed, Mapping)
                            else None
                        )
                    ),
                    error_category=_tool_error_category(state, processed),
                    error_fingerprint=_tool_error_fingerprint(state, processed),
                    result_content_hash=row["result_content_hash"],
                    content_hash=_mapping_str(processed, "content_hash"),
                    previous_content_hash=_mapping_str(
                        processed, "previous_content_hash"
                    ),
                    policy_tool_event=policy_tool_event,
                )
            )
    return events


def _merge_file_events(events: Sequence[_FileEvent]) -> list[_MergedFileEvent]:
    grouped: dict[tuple[object, ...], list[_FileEvent]] = {}
    order: list[tuple[object, ...]] = []
    for event in events:
        key = _file_event_merge_key(event)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(event)

    merged: list[_MergedFileEvent] = []
    for key in order:
        raw_events = grouped[key]
        display_event = next(
            (event for event in reversed(raw_events) if event.phase == "post"),
            raw_events[-1],
        )
        merged.append(
            _MergedFileEvent(
                event=display_event,
                phases=_unique_labels(event.phase for event in raw_events),
                sources=_unique_labels(event.source for event in raw_events),
                events=tuple(raw_events),
            )
        )
    return merged


def _file_event_merge_key(event: _FileEvent) -> tuple[object, ...]:
    tool_call_id = event.policy_tool_event.get("tool_call_id")
    if tool_call_id is None:
        return ("event", event.event_id)
    return (
        "tool_call_file",
        event.turn,
        tool_call_id,
        event.tool_name,
        event.operation,
        event.path,
    )


def _unique_labels(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    labels: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        labels.append(value)
    return tuple(labels)


def _file_events_by_path(
    events: Sequence[_MergedFileEvent],
) -> dict[str, list[_MergedFileEvent]]:
    grouped: defaultdict[str, list[_MergedFileEvent]] = defaultdict(list)
    for event in events:
        grouped[event.event.path].append(event)
    return dict(grouped)


def _known_file_paths(conn: Connection, session_id: str) -> set[str]:
    paths: set[str] = set()
    try:
        rows = _rows(
            conn,
            """
            SELECT path
            FROM policy_file_state
            WHERE session_id = ?
            """,
            (session_id,),
        )
    except Exception as exc:
        logger.debug("policy known file lookup failed: {}", exc)
        return paths
    for row in rows:
        path = _clean_path(row["path"])
        if path:
            paths.add(path)
    return paths


def _file_event_paths(
    tool_name: str,
    args: object,
    processed: object,
    state: object,
    *,
    known_paths: set[str],
) -> list[tuple[str, str]]:
    candidates: list[tuple[object, str]] = []
    if isinstance(args, Mapping):
        candidates.extend(
            [
                (args.get("path"), "args.path"),
                (args.get("file_path"), "args.file_path"),
            ]
        )
    if isinstance(processed, Mapping):
        candidates.append((processed.get("path"), "processed.path"))
    entry = _tool_log_entry(state)
    candidates.append((entry.get("path"), "tool_log.path"))

    cmd = _command_text(args)
    if cmd and tool_name == "bash":
        candidates.extend(_paths_from_command(cmd, known_paths=known_paths))

    seen: set[str] = set()
    paths: list[tuple[str, str]] = []
    for raw, source in candidates:
        path = _clean_path(raw)
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append((path, source))
    return paths


def _file_operation(
    tool_name: str,
    args: object,
    processed: object,
    source: str,
) -> str:
    raw_op = _mapping_str(processed, "file_op")
    if raw_op:
        lowered = raw_op.lower()
        if lowered in {"read", "open", "glob", "list"}:
            return "read"
        if lowered in {"edit", "write", "patch", "delete", "remove"}:
            return "write"
    if tool_name in {"read", "glob"}:
        return "read"
    if tool_name in {"edit", "write"}:
        return "write"
    if source in {"cmd.write", "cmd.redirect.write"}:
        return "write"
    if source in {"cmd.read", "cmd.reader", "cmd.redirect.read"}:
        return "read"
    if source == "cmd.query":
        return "query"
    if source in {"cmd.edit", "cmd.delete"}:
        return "write"
    if source == "cmd.known":
        return _bash_file_operation(_command_text(args))
    return "reference"


def _bash_file_operation(cmd: str) -> str:
    categories = [
        analyze_bash_segment(segment).action_kind
        for segment in parse_bash_segments(cmd)
    ]
    if "write" in categories:
        return "write"
    if "query" in categories:
        return "query"
    if "read" in categories:
        return "read"
    return "reference"


def _file_tool_status(
    row: RowMapping,
    state: object,
    processed: object,
    error: bool,
) -> str:
    if row["phase"] == "pre":
        return "intent"
    if error:
        category = _tool_error_category(state, processed) or "unknown"
        exit_code = _tool_exit_code(row, processed)
        if exit_code is not None:
            return f"exit:{exit_code} {category}"
        return f"error:{category}"
    return f"ok:{_dash(_tool_exit_code(row, processed))}"


def _file_operation_status(tool_name: str, phase: str, tool_status: str) -> str:
    if phase == "pre":
        return "intent"
    if tool_name == "bash":
        return "observed"
    return tool_status


def _file_evidence_label(event: _FileEvent, source: str) -> str:
    if event.tool_name != "bash":
        return "structured"
    if event.phase == "pre":
        return "intent"
    source_parts = set(source.split("+"))
    if "cmd.query" in source_parts:
        return "query"
    if source_parts.intersection(
        {
            "cmd.read",
            "cmd.reader",
            "cmd.write",
            "cmd.edit",
            "cmd.delete",
            "cmd.redirect.read",
            "cmd.redirect.write",
        }
    ):
        return "shell-op"
    return "reference"


def _tool_exit_code(row: RowMapping, processed: object) -> int | None:
    exit_code = _coerce_int(row["exit_code"])
    if exit_code is not None:
        return exit_code
    if isinstance(processed, Mapping):
        return _coerce_int(processed.get("exit_code"))
    return None


def _file_event_json(event: _FileEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "ts": event.ts,
        "turn": event.turn,
        "phase": event.phase,
        "tool_name": event.tool_name,
        "operation": event.operation,
        "path": event.path,
        "source": event.source,
        "args_hash": event.args_hash,
        "status": event.tool_status,
        "file_status": event.file_status,
        "tool_status": event.tool_status,
        "tool_exit_code": event.tool_exit_code,
        "duration_ms": event.duration_ms,
        "error_category": event.error_category,
        "error_fingerprint": event.error_fingerprint,
        "result_content_hash": event.result_content_hash,
        "content_hash": event.content_hash,
        "previous_content_hash": event.previous_content_hash,
    }


def _merged_file_event_json(event: _MergedFileEvent) -> dict[str, object]:
    data = _file_event_json(event.event)
    data["phase"] = event.phase_label
    data["source"] = event.source_label
    data["raw_event_count"] = len(event.events)
    return data


def _latest_file_event(history: Sequence[_MergedFileEvent]) -> str:
    if not history:
        return "-"
    event = history[-1]
    return (
        f"T{event.event.turn} {event.event.operation}/"
        f"{event.phase_label} {event.event.tool_name}"
    )


def _file_content_signals(
    history: Sequence[_MergedFileEvent],
) -> list[dict[str, object]]:
    signals: list[dict[str, object]] = []
    for merged in history:
        event = merged.event
        text = _file_event_content_text(event)
        if not any(
            (
                event.content_hash,
                event.previous_content_hash,
                event.result_content_hash,
                text,
            )
        ):
            continue
        tokens = _content_identifier_tokens(text)
        signals.append(
            {
                "turn": event.turn,
                "operation": event.operation,
                "phase": merged.phase_label,
                "content_hash": event.content_hash,
                "previous_content_hash": event.previous_content_hash,
                "result_content_hash": event.result_content_hash,
                "content_preview": _single_line(text, limit=240) if text else "",
                "identifier_tokens": [
                    {"token": token, "count": count}
                    for token, count in Counter(tokens).most_common(30)
                ],
            }
        )
    return signals


def _file_event_content_text(event: _FileEvent) -> str:
    if event.tool_name != "read" or event.operation != "read":
        return ""
    raw = _tool_result_text_for_rule(event.policy_tool_event.get("result"))
    return raw if isinstance(raw, str) else ""


def _content_identifier_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [match.group(0) for match in _IDENTIFIER_RE.finditer(text[:200_000])]


def _command_text(args: object) -> str:
    if not isinstance(args, Mapping):
        return ""
    raw = args.get("cmd") or args.get("command")
    return raw if isinstance(raw, str) else ""


def _paths_from_command(
    cmd: str,
    *,
    known_paths: set[str],
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    for segment in parse_bash_segments(cmd):
        analysis = analyze_bash_segment(segment)
        command = segment.argv
        candidates.extend(
            _known_paths_from_tokens(
                command,
                known_paths,
                source=_known_path_source_for_command(analysis.action_kind),
            )
        )
        candidates.extend(
            (ref.path, ref.source)
            for ref in analysis.path_refs
            if ref.path_kind == "file"
        )

    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for path, source in candidates:
        clean = _clean_path(path)
        if not clean or _is_virtual_filesystem_path(clean):
            continue
        item = (clean, source)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _bash_segment_text(segment: BashSegment) -> str:
    if not segment.redirects or segment.parser == "shlex-fallback":
        return segment.text
    return " ".join([segment.text, *(redirect.text for redirect in segment.redirects)])


def _known_paths_from_tokens(
    tokens: Sequence[str],
    known_paths: set[str],
    *,
    source: str,
) -> Iterable[tuple[str, str]]:
    if not known_paths:
        return ()
    known_by_variant: dict[str, str] = {}
    for path in known_paths:
        for variant in _known_path_variants(path):
            known_by_variant.setdefault(variant, path)

    matches: list[tuple[str, str]] = []
    for token in tokens:
        clean = _clean_path(token)
        if not clean:
            continue
        known = known_by_variant.get(clean)
        if known is not None:
            matches.append((known, source))
    return matches


def _known_path_source_for_command(category: str) -> str:
    if category == "query":
        return "cmd.query"
    if category == "read":
        return "cmd.reader"
    return "cmd.known"


def _known_path_variants(path: str) -> Iterable[str]:
    clean = _clean_path(path)
    if not clean:
        return ()
    variants = [clean]
    if clean.startswith("/"):
        parts = [part for part in clean.split("/") if part]
        variants.extend("/".join(parts[index:]) for index in range(1, len(parts) - 1))
    return variants


def _is_probable_shell_path(value: object) -> bool:
    path = _clean_path(value)
    if not path or _is_virtual_filesystem_path(path):
        return False
    return path.startswith(("/", "./", "../")) or "/" in path or "." in path


def _is_virtual_filesystem_path(path: str) -> bool:
    parts = [part for part in path.split("/") if part]
    return bool(path.startswith("/") and parts and parts[0] in {"dev", "proc", "sys"})


def _clean_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.strip().strip("\"'`,;:()[]{}")
    if not path or len(path) < 3:
        return None
    if path.startswith("-") or "*" in path or "://" in path:
        return None
    return path


def _entity_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_entity_state
        WHERE session_id = ?
        ORDER BY occurrence_count DESC, updated_at DESC, entity ASC
        """,
        (session_id,),
    ):
        evidence = _loads(row["evidence_json"])
        yield TraceRow(
            key=f"policy:entity:{row['entity']}",
            kind="policy",
            title=row["entity"],
            preview=(
                f"type:{row['entity_type']} count:{row['occurrence_count']} "
                f"evidence:{_evidence_preview(evidence)}"
            ),
            content=_json_content(
                {
                    "session_id": row["session_id"],
                    "entity": row["entity"],
                    "entity_type": row["entity_type"],
                    "updated_at": row["updated_at"],
                    "first_seen_turn": row["first_seen_turn"],
                    "last_seen_turn": row["last_seen_turn"],
                    "occurrence_count": row["occurrence_count"],
                    "evidence": evidence,
                }
            ),
            turn_index=row["last_seen_turn"],
            metadata={
                "category": "entity",
                "entity_type": row["entity_type"],
                "occurrence_count": row["occurrence_count"],
            },
        )


def _eval_error_rows(conn: Connection, session_id: str) -> Iterable[TraceRow]:
    for row in _rows(
        conn,
        """
        SELECT *
        FROM policy_eval_error
        WHERE session_id = ?
        ORDER BY ts ASC, id ASC
        """,
        (session_id,),
    ):
        yield TraceRow(
            key=f"policy:eval_error:{row['id']}",
            kind="policy",
            title=row["rule_id"],
            preview=row["error"],
            content=_json_content(dict(row)),
            turn_index=row["turn"],
            tool_name=row["tool_name"],
            is_error=True,
            cause="eval_error",
            metadata={"category": "eval_error", "channel": row["channel"]},
        )


def _diagnostic_row(
    session_id: str,
    *,
    title: str,
    preview: str,
    content: str,
) -> TraceRow:
    return TraceRow(
        key=f"policy:diagnostic:{session_id}:{title}",
        kind="policy",
        title=title,
        preview=preview,
        content=content,
        is_error=True,
        cause="missing",
        metadata={"category": "diagnostic"},
    )


def _empty_category_row(
    session_id: str,
    db_path: Path,
    counts: Mapping[str, int],
    categories: frozenset[str],
) -> TraceRow:
    title = "No Policy Projection"
    preview = "No persisted policy rows for this session."
    hint = (
        "Run policy_engine backfill --session "
        f"{session_id} to compute policy projection rows."
    )
    if categories == _ERROR_CATEGORIES:
        title = "No Policy Eval Errors"
        preview = "No policy evaluation errors for this session."
        hint = "This session has no persisted policy evaluation errors."
    elif categories == _EFFECT_CATEGORIES:
        title = "No Policy Effects"
        preview = "No policy rule firings for this session."
        hint = "This session has no persisted policy rule firings."
    elif categories == _TOOL_CATEGORIES:
        title = "No Policy Tool Events"
        preview = "No policy-observed tool events for this session."
    elif categories == _FILE_CATEGORIES:
        title = "No Policy File State"
        preview = "No persisted policy file state for this session."
    elif categories == _FILE_STREAM_CATEGORIES:
        title = "No Policy File Stream"
        preview = "No structured non-bash file events could be derived."
    elif categories == _FILE_TOOL_STREAM_CATEGORIES:
        title = "No Policy File Tool Stream"
        preview = "No direct read/write/edit file-tool events could be derived."
    elif categories == _BASH_FILE_STREAM_CATEGORIES:
        title = "No Policy Bash File Stream"
        preview = "No bash-derived file events could be derived for this session."
    elif categories == _BASH_COMMAND_CATEGORIES:
        title = "No Policy Bash Commands"
        preview = "No bash command templates could be derived for this session."
    elif categories == _IFG_ACTION_CATEGORIES:
        title = "No IFG Actions"
        preview = "No persisted IFG actions for this session."
        hint = f"Run policy_engine ifg backfill --session {session_id}."
    elif categories == _IFG_SOURCE_UNIT_CATEGORIES:
        title = "No IFG Source Units"
        preview = "No persisted IFG source units for this session."
        hint = f"Run policy_engine ifg backfill --session {session_id}."
    elif categories == _IFG_FILE_CATEGORIES:
        title = "No IFG Files"
        preview = "No persisted IFG file nodes for this session."
        hint = f"Run policy_engine ifg backfill --session {session_id}."
    elif categories == _IFG_SYMBOL_CATEGORIES:
        title = "No IFG Symbols"
        preview = "No persisted IFG symbols for this session."
        hint = f"Run policy_engine ifg backfill --session {session_id}."
    elif categories == _ENTITY_CATEGORIES:
        title = "No Policy Entity State"
        preview = "No persisted policy entity state for this session."

    return TraceRow(
        key=f"policy:empty:{session_id}:{','.join(sorted(categories))}",
        kind="policy",
        title=title,
        preview=preview,
        content=_json_content(
            {
                "session_id": session_id,
                "db_path": str(db_path),
                "counts": dict(counts),
                "categories": sorted(categories),
                "hint": hint,
            }
        ),
        metadata={"category": "diagnostic"},
    )


def _existing_tables(conn: Connection) -> set[str]:
    return {
        str(row["name"])
        for row in _rows(conn, "SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _session_counts(
    conn: Connection,
    session_id: str,
    existing: set[str],
    *,
    ifg_extractor_version: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in (*_POLICY_TABLES, *_IFG_TABLES):
        if table not in existing:
            counts[table] = 0
            continue
        if table in _IFG_TABLES:
            row = _row(
                conn,
                f"SELECT COUNT(*) AS count FROM {table} "  # noqa: S608
                "WHERE session_id = ? AND extractor_version = ?",
                (session_id, ifg_extractor_version),
            )
        else:
            row = _row(
                conn,
                f"SELECT COUNT(*) AS count FROM {table} WHERE session_id = ?",  # noqa: S608
                (session_id,),
            )
        counts[table] = int(row["count"]) if row is not None else 0
    return counts


def _loads(raw: object) -> object:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _json_content(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _json_preview(value: object, *, limit: int = 150) -> str:
    if value is None:
        return ""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _counts_preview(counts: Mapping[str, int]) -> str:
    parts = [
        f"effects:{counts.get('event_log', 0)}",
        f"tools:{counts.get('policy_tool_events', 0)}",
        f"files:{counts.get('policy_file_state', 0)}",
        f"entities:{counts.get('policy_entity_state', 0)}",
        f"errors:{counts.get('policy_eval_error', 0)}",
        f"ifg-actions:{counts.get('ifg_actions', 0)}",
        f"source-units:{counts.get('ifg_source_units', 0)}",
        f"path-candidates:{counts.get('ifg_path_candidates', 0)}",
        f"ifg-files:{counts.get('ifg_files', 0)}",
        f"symbols:{counts.get('ifg_symbols', 0)}",
    ]
    return " ".join(parts)


def _turn_summary_preview(summary: object) -> str:
    if not isinstance(summary, Mapping):
        return _json_preview(summary)
    parts = []
    for key in (
        "tool_calls_count",
        "tool_log_entries",
        "effect_log_entries",
        "error_count",
        "context_tokens",
        "total_context_tokens",
    ):
        if key in summary:
            parts.append(f"{key}:{summary[key]}")
    return " ".join(parts) or _json_preview(summary)


def _tool_policy_state_label(
    row: RowMapping,
    state: object,
    processed: object,
    is_error: bool,
) -> str:
    phase = str(row["phase"])
    if phase == "pre":
        labels = _taint_labels(state)
        joined = ",".join(labels) if labels else "-"
        return f"pre taint:{joined}"
    if is_error:
        category = _tool_error_category(state, processed) or "error"
        return f"post err:{category}"
    exit_code = row["exit_code"]
    if exit_code is None and isinstance(processed, Mapping):
        exit_code = processed.get("exit_code")
    return f"post exit:{_dash(exit_code)}"


def _tool_policy_preview(
    row: RowMapping,
    args: object,
    result: object,
    state: object,
    processed: object,
) -> str:
    phase = str(row["phase"])
    parts = [f"hash:{_short_text(row['args_hash'], 8)}"]
    if phase == "pre":
        labels = _taint_labels(state)
        parts.append(f"taint:{','.join(labels) if labels else '-'}")
    else:
        parts.extend(
            [
                f"exit:{_dash(row['exit_code'])}",
                f"err:{_dash(_tool_error_category(state, processed))}",
                f"fp:{_dash(_tool_error_fingerprint(state, processed))}",
                f"result:{_dash(row['result_content_hash'])}",
                f"len:{_dash(_tool_result_length(result, processed, state))}",
            ]
        )
    source = _tool_source_preview(args, result, state, processed)
    if source:
        parts.append(source)
    return " ".join(parts)


def _tool_rule_author_event(
    row: RowMapping,
    args: object,
    result: object,
    state: object,
    processed: object,
) -> dict[str, object]:
    phase = str(row["phase"])
    if phase == "pre":
        return {
            "channel": "tool_call_pre",
            "event.tool_name": row["tool_name"],
            "event.args": args,
            "event.result": {},
            "event.taint": _taint_labels(state),
            "query_notes": [
                "This is the current event inspected by rules on tool_call_pre.",
                "Rules can also query prior calls via tool_log/file_state/entity_evidence.",
            ],
        }

    return {
        "channel": "tool_call_post",
        "event.tool_name": row["tool_name"],
        "event.args": {},
        "event.result": {
            "text": _tool_result_text_for_rule(result),
            "error": _tool_error_text(result, processed, state),
        },
        "event.taint": [],
        "current_tool_args": args,
        "query_notes": [
            "Current tool args are persisted for debugging here, but are not passed "
            "directly as event.args to tool_call_post predicates today.",
            "Post-result rules usually query the recorded call through tool_log.last(...).",
        ],
    }


def _tool_preview(
    args: object,
    result: object,
    state: object,
    processed: object,
) -> str:
    for value in _candidate_values(args, ("cmd", "command", "path", "file_path")):
        return str(value)
    for value in _candidate_values(processed, ("error", "error_category", "summary")):
        return str(value)
    for value in _candidate_values(result, ("error", "message", "output", "content")):
        return str(value)
    for value in _candidate_values(state, ("processed", "tool_log_entry")):
        return _json_preview(value)
    return _json_preview({"args": args, "result": result, "processed": processed})


def _tool_source_preview(
    args: object,
    result: object,
    state: object,
    processed: object,
) -> str:
    for value in _candidate_values(args, ("cmd", "command", "path", "file_path")):
        return _single_line(value)
    for value in _candidate_values(processed, ("path", "file_op", "summary")):
        return _single_line(value)
    for value in _candidate_values(result, ("error", "message", "output", "content")):
        return _single_line(value)
    for value in _candidate_values(state, ("tool_log_entry", "processed")):
        return _json_preview(value, limit=80)
    return ""


def _candidate_values(value: object, keys: Sequence[str]) -> Iterable[object]:
    if not isinstance(value, Mapping):
        return ()
    return (value[key] for key in keys if key in value and value[key] not in (None, ""))


def _tool_is_error(
    row: RowMapping,
    result: object,
    state: object,
    processed: object,
) -> bool:
    if row["exit_code"] not in (None, 0):
        return True
    for value in (result, processed):
        if isinstance(value, Mapping) and value.get("is_error") is True:
            return True
    if isinstance(state, Mapping):
        entry = state.get("tool_log_entry")
        if isinstance(entry, Mapping):
            return bool(entry.get("error") or entry.get("error_category"))
    return False


def _taint_labels(state: object) -> list[str]:
    if not isinstance(state, Mapping):
        return []
    raw = state.get("taint_labels")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _tool_error_category(state: object, processed: object) -> str | None:
    value = _mapping_value(processed, "error_category")
    if value:
        return str(value)
    entry = _tool_log_entry(state)
    value = _mapping_value(entry, "error_category")
    return str(value) if value else None


def _tool_error_fingerprint(state: object, processed: object) -> str | None:
    value = _mapping_value(processed, "error_fingerprint")
    if value:
        return str(value)
    entry = _tool_log_entry(state)
    value = _mapping_value(entry, "error_fingerprint")
    return str(value) if value else None


def _tool_error_text(result: object, processed: object, state: object) -> object:
    for value in _candidate_values(processed, ("error",)):
        return value
    for value in _candidate_values(result, ("error", "message")):
        return value
    entry = _tool_log_entry(state)
    value = _mapping_value(entry, "error")
    return value


def _tool_result_length(result: object, processed: object, state: object) -> object:
    for value in _candidate_values(processed, ("result_length", "text_length")):
        return value
    entry = _tool_log_entry(state)
    value = _mapping_value(entry, "result_length")
    if value is not None:
        return value
    text = _tool_result_text_for_rule(result)
    return len(text) if isinstance(text, str) else None


def _tool_result_text_for_rule(result: object) -> object:
    if isinstance(result, Mapping):
        text = result.get("text")
        if isinstance(text, str):
            return text
        content = result.get("content")
        if isinstance(content, list):
            parts = [
                text
                for item in content
                if isinstance(item, Mapping)
                and isinstance((text := item.get("text")), str)
            ]
            return "".join(parts) if parts else None
    return None


def _tool_log_entry(state: object) -> Mapping[str, object]:
    if not isinstance(state, Mapping):
        return {}
    raw = state.get("tool_log_entry")
    return raw if isinstance(raw, Mapping) else {}


def _mapping_value(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _mapping_str(value: object, key: str) -> str | None:
    raw = _mapping_value(value, key)
    return str(raw) if raw not in (None, "") else None


def _single_line(value: object, *, limit: int = 100) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _short_text(value: object, keep: int) -> str:
    text = "" if value is None else str(value)
    return text[:keep]


def _stable_id(value: object, *, keep: int = 16) -> str:
    raw = "" if value is None else str(value)
    return hashlib.sha256(raw.encode()).hexdigest()[:keep]


def _evidence_preview(evidence: object) -> str:
    if not isinstance(evidence, list):
        return _json_preview(evidence)
    counts: dict[str, int] = {}
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _last_turn(*values: object) -> int | None:
    ints = [coerced for value in values if (coerced := _coerce_int(value)) is not None]
    return max(ints) if ints else None


def _dash(value: object) -> str:
    return "-" if value is None else str(value)


def _summary_text(
    db_path: Path,
    rows: Sequence[TraceRow],
    filtered: Sequence[TraceRow],
) -> str:
    return (
        f"{len(filtered)} row(s) | total:{len(rows)} | policy_db:{_short_path(db_path)}"
    )


def _short_path(path: Path) -> str:
    try:
        return str(path.expanduser()).replace(str(Path.home()), "~", 1)
    except RuntimeError:
        return str(path)


__all__ = [
    "PolicyTraceViewProvider",
    "build_policy_trace_view_registry",
    "default_policy_db_path",
    "load_policy_trace_rows",
    "run_policy_trace_viewer",
]
