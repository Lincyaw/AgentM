"""Vault SQLite schema — DDL creation, vec support probe, table clearing."""

from __future__ import annotations

import sqlite3

_TABLES_SQL = """\
CREATE TABLE IF NOT EXISTS notes (
    path TEXT PRIMARY KEY,
    type TEXT,
    title TEXT,
    confidence TEXT,
    status TEXT DEFAULT 'active',
    created_at TEXT,
    updated_at TEXT,
    frontmatter TEXT,
    body_hash TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    path, title, body, tags,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS links (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    PRIMARY KEY (source, target)
);
CREATE INDEX IF NOT EXISTS idx_links_target ON links(target);

CREATE TABLE IF NOT EXISTS tags (
    path TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (path, tag)
);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
"""

_VEC_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(
    path TEXT PRIMARY KEY,
    embedding FLOAT[512]
);
"""

_DATA_TABLES = ("notes", "notes_fts", "links", "tags")


def has_vec_support(conn: sqlite3.Connection) -> bool:
    """Probe whether sqlite-vec extension is available. Never raises."""
    try:
        import sqlite_vec  # noqa: F401

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except Exception:  # noqa: BLE001
        return False


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all vault tables. Idempotent (IF NOT EXISTS)."""
    conn.executescript(_TABLES_SQL)
    if has_vec_support(conn):
        conn.executescript(_VEC_SQL)


def clear_all(conn: sqlite3.Connection) -> None:
    """DELETE FROM all data tables. Schema is preserved."""
    for table in _DATA_TABLES:
        conn.execute(f"DELETE FROM {table}")  # noqa: S608
    # Clear vec table if it exists
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='notes_vec'"
    ).fetchone()
    if row:
        conn.execute("DELETE FROM notes_vec")
    conn.commit()
