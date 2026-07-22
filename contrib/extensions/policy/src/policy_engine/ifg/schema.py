"""SQLite schema management for policy IFG tables."""

from __future__ import annotations

from sqlalchemy.engine import Connection

from policy_engine.source_parser import SYMBOL_EXTRACTOR_VERSION
from policy_engine.source_semantics import BASH_SEMANTICS_EXTRACTOR_VERSION

IFG_EXTRACTOR_VERSION = (
    f"ifg-v5+{BASH_SEMANTICS_EXTRACTOR_VERSION}+{SYMBOL_EXTRACTOR_VERSION}"
)

_IFG_REQUIRED_COLUMNS = {
    "ifg_action_file_edges": {
        "is_anchor": "INTEGER NOT NULL DEFAULT 1",
        "content_hash": "TEXT",
        "before_hash": "TEXT",
        "after_hash": "TEXT",
        "content_state": "TEXT",
        "line_range_json": "TEXT NOT NULL DEFAULT '{}'",
        "span_json": "TEXT NOT NULL DEFAULT '{}'",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    },
}

_IFG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ifg_nodes (
    node_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    node_type TEXT NOT NULL,
    stable_key TEXT NOT NULL,
    display_name TEXT NOT NULL,
    first_seen_turn INTEGER NOT NULL,
    last_seen_turn INTEGER NOT NULL,
    observation_count INTEGER NOT NULL,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ifg_nodes_identity
    ON ifg_nodes(session_id, extractor_version, node_type, stable_key);
CREATE INDEX IF NOT EXISTS idx_ifg_nodes_session_type
    ON ifg_nodes(session_id, node_type, last_seen_turn);

CREATE TABLE IF NOT EXISTS ifg_edges (
    edge_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    from_node_id TEXT NOT NULL,
    to_node_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_edges_session_relation
    ON ifg_edges(session_id, relation, turn);
CREATE INDEX IF NOT EXISTS idx_ifg_edges_from_node
    ON ifg_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_ifg_edges_to_node
    ON ifg_edges(to_node_id);

CREATE TABLE IF NOT EXISTS ifg_actions (
    action_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    tool_call_id TEXT,
    tool_name TEXT NOT NULL,
    segment_index INTEGER,
    command TEXT,
    action_kind TEXT NOT NULL,
    family TEXT NOT NULL,
    template TEXT,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_actions_session_turn
    ON ifg_actions(session_id, turn, event_id);
CREATE INDEX IF NOT EXISTS idx_ifg_actions_kind
    ON ifg_actions(session_id, action_kind);

CREATE TABLE IF NOT EXISTS ifg_normalized_tool_events (
    normalized_event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_id INTEGER,
    tool_call_id TEXT,
    phase TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    processed_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    cwd TEXT,
    turn INTEGER NOT NULL,
    ts REAL NOT NULL,
    is_error INTEGER NOT NULL,
    source TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_normalized_tool_events_session_turn
    ON ifg_normalized_tool_events(session_id, turn, event_id);

CREATE TABLE IF NOT EXISTS ifg_files (
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    path TEXT NOT NULL,
    first_seen_turn INTEGER NOT NULL,
    last_seen_turn INTEGER NOT NULL,
    observation_count INTEGER NOT NULL,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (session_id, extractor_version, path)
);

CREATE INDEX IF NOT EXISTS idx_ifg_files_session_last_seen
    ON ifg_files(session_id, last_seen_turn);

CREATE TABLE IF NOT EXISTS ifg_action_file_edges (
    edge_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    path TEXT NOT NULL,
    relation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    is_anchor INTEGER NOT NULL,
    extractor_version TEXT NOT NULL,
    content_hash TEXT,
    before_hash TEXT,
    after_hash TEXT,
    content_state TEXT,
    line_range_json TEXT NOT NULL DEFAULT '{}',
    span_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_action_file_edges_session_path
    ON ifg_action_file_edges(session_id, path, turn);
CREATE INDEX IF NOT EXISTS idx_ifg_action_file_edges_action
    ON ifg_action_file_edges(action_id);
CREATE TABLE IF NOT EXISTS ifg_source_units (
    source_unit_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    action_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    origin TEXT NOT NULL,
    path TEXT,
    relation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    tool_name TEXT NOT NULL,
    language TEXT,
    content_hash TEXT,
    previous_content_hash TEXT,
    result_content_hash TEXT,
    unit_hash TEXT,
    content_state TEXT NOT NULL,
    line_range_json TEXT NOT NULL,
    span_json TEXT NOT NULL,
    content_text TEXT,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_source_units_session_path
    ON ifg_source_units(session_id, path, turn);
CREATE INDEX IF NOT EXISTS idx_ifg_source_units_action
    ON ifg_source_units(action_id);
CREATE INDEX IF NOT EXISTS idx_ifg_source_units_kind
    ON ifg_source_units(session_id, kind, turn);

CREATE TABLE IF NOT EXISTS ifg_path_candidates (
    candidate_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    action_id TEXT NOT NULL,
    source_unit_id TEXT,
    path_text TEXT NOT NULL,
    normalized_path TEXT NOT NULL,
    path_kind TEXT NOT NULL,
    relation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_path_candidates_session_path
    ON ifg_path_candidates(session_id, extractor_version, normalized_path, turn);
CREATE INDEX IF NOT EXISTS idx_ifg_path_candidates_action
    ON ifg_path_candidates(action_id);

CREATE TABLE IF NOT EXISTS ifg_symbols (
    symbol_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    kind TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    path TEXT,
    stable_key TEXT NOT NULL,
    first_seen_turn INTEGER NOT NULL,
    last_seen_turn INTEGER NOT NULL,
    observation_count INTEGER NOT NULL,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ifg_symbols_identity
    ON ifg_symbols(session_id, extractor_version, stable_key);
CREATE INDEX IF NOT EXISTS idx_ifg_symbols_session_name
    ON ifg_symbols(session_id, qualified_name, last_seen_turn);

CREATE TABLE IF NOT EXISTS ifg_action_symbol_edges (
    edge_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    action_id TEXT NOT NULL,
    symbol_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_action_symbol_edges_action
    ON ifg_action_symbol_edges(action_id);
CREATE INDEX IF NOT EXISTS idx_ifg_action_symbol_edges_symbol
    ON ifg_action_symbol_edges(symbol_id);

CREATE TABLE IF NOT EXISTS ifg_file_symbol_edges (
    edge_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    path TEXT NOT NULL,
    symbol_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_file_symbol_edges_path
    ON ifg_file_symbol_edges(session_id, path, turn);
CREATE INDEX IF NOT EXISTS idx_ifg_file_symbol_edges_symbol
    ON ifg_file_symbol_edges(symbol_id);

CREATE TABLE IF NOT EXISTS ifg_symbol_symbol_edges (
    edge_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    from_symbol_id TEXT NOT NULL,
    to_symbol_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_symbol_symbol_edges_from
    ON ifg_symbol_symbol_edges(from_symbol_id);
CREATE INDEX IF NOT EXISTS idx_ifg_symbol_symbol_edges_to
    ON ifg_symbol_symbol_edges(to_symbol_id);
CREATE INDEX IF NOT EXISTS idx_ifg_symbol_symbol_edges_session_relation
    ON ifg_symbol_symbol_edges(session_id, relation, turn);

CREATE TABLE IF NOT EXISTS ifg_symbol_mentions (
    mention_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    action_id TEXT NOT NULL,
    source_unit_id TEXT,
    symbol_text TEXT NOT NULL,
    turn INTEGER NOT NULL,
    event_id INTEGER,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    path TEXT,
    metadata_json TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_symbol_mentions_session_symbol
    ON ifg_symbol_mentions(session_id, symbol_text, turn);
CREATE INDEX IF NOT EXISTS idx_ifg_symbol_mentions_source_unit
    ON ifg_symbol_mentions(source_unit_id);

CREATE TABLE IF NOT EXISTS ifg_extraction_error (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn INTEGER,
    event_id INTEGER,
    tool_call_id TEXT,
    tool_name TEXT,
    extractor_version TEXT NOT NULL,
    error TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ifg_extraction_error_session
    ON ifg_extraction_error(session_id, ts);

CREATE TABLE IF NOT EXISTS ifg_session_summary (
    session_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    updated_at REAL NOT NULL,
    summary_json TEXT NOT NULL,
    PRIMARY KEY (session_id, extractor_version)
);
"""

IFG_TABLES = (
    "ifg_edges",
    "ifg_nodes",
    "ifg_normalized_tool_events",
    "ifg_actions",
    "ifg_files",
    "ifg_action_file_edges",
    "ifg_source_units",
    "ifg_path_candidates",
    "ifg_symbols",
    "ifg_action_symbol_edges",
    "ifg_file_symbol_edges",
    "ifg_symbol_symbol_edges",
    "ifg_symbol_mentions",
    "ifg_extraction_error",
    "ifg_session_summary",
)

IFG_DERIVED_TABLES = (
    "ifg_edges",
    "ifg_nodes",
    "ifg_files",
    "ifg_symbols",
    "ifg_action_symbol_edges",
    "ifg_file_symbol_edges",
    "ifg_symbol_symbol_edges",
)


def ensure_ifg_schema(conn: Connection) -> None:
    for statement in _IFG_SCHEMA_SQL.split(";"):
        sql = statement.strip()
        if sql:
            conn.exec_driver_sql(sql)
    _ensure_ifg_columns(conn)
    conn.exec_driver_sql(
        """
        CREATE INDEX IF NOT EXISTS idx_ifg_action_file_edges_anchor
        ON ifg_action_file_edges(session_id, extractor_version, is_anchor)
        """
    )


def _ensure_ifg_columns(conn: Connection) -> None:
    for table, columns in _IFG_REQUIRED_COLUMNS.items():
        existing = {
            str(row[1])
            for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
        }
        for name, ddl in columns.items():
            if name in existing:
                continue
            conn.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"  # noqa: S608
            )


def delete_ifg_session(
    conn: Connection,
    session_id: str,
    *,
    extractor_version: str | None = None,
) -> int:
    ensure_ifg_schema(conn)
    tables = (
        "ifg_edges",
        "ifg_nodes",
        "ifg_normalized_tool_events",
        "ifg_actions",
        "ifg_files",
        "ifg_action_file_edges",
        "ifg_source_units",
        "ifg_path_candidates",
        "ifg_symbols",
        "ifg_action_symbol_edges",
        "ifg_file_symbol_edges",
        "ifg_symbol_symbol_edges",
        "ifg_symbol_mentions",
        "ifg_extraction_error",
        "ifg_session_summary",
    )
    deleted = 0
    for table in tables:
        if extractor_version is None:
            cursor = conn.exec_driver_sql(
                f"DELETE FROM {table} WHERE session_id = ?",  # noqa: S608
                (session_id,),
            )
        else:
            cursor = conn.exec_driver_sql(
                f"DELETE FROM {table} WHERE session_id = ? AND extractor_version = ?",  # noqa: S608
                (session_id, extractor_version),
            )
        deleted += cursor.rowcount
    return deleted
