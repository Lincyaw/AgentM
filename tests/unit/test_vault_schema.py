"""Focused regression tests for vault schema lifecycle helpers."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

from agentm.tools.vault.schema import clear_all, create_schema, has_vec_support


def test_create_schema_is_idempotent_and_creates_core_tables_and_indexes() -> None:
    conn = sqlite3.connect(":memory:")

    create_schema(conn)
    create_schema(conn)

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    assert {"notes", "notes_fts", "links", "tags"}.issubset(tables)

    notes_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()
    }
    assert {
        "path",
        "type",
        "title",
        "confidence",
        "status",
        "created_at",
        "updated_at",
        "frontmatter",
        "body_hash",
    }.issubset(notes_columns)

    indexes = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    }
    assert "idx_links_target" in indexes
    assert "idx_tags_tag" in indexes


def test_has_vec_support_returns_false_when_extension_import_fails() -> None:
    conn = sqlite3.connect(":memory:")

    with patch("builtins.__import__", side_effect=ImportError("sqlite_vec unavailable")):
        assert has_vec_support(conn) is False


def test_clear_all_clears_data_but_keeps_schema_usable() -> None:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)

    conn.execute(
        "INSERT INTO notes (path, type, title, confidence) VALUES ('a.md', 'skill', 'A', 'fact')"
    )
    conn.execute(
        "INSERT INTO notes_fts (path, title, body, tags) VALUES ('a.md', 'A', 'body', 'x')"
    )
    conn.execute("INSERT INTO links (source, target) VALUES ('a.md', 'b.md')")
    conn.execute("INSERT INTO tags (path, tag) VALUES ('a.md', 'x')")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS note_embeddings (path TEXT PRIMARY KEY, embedding BLOB)"
    )
    conn.execute(
        "INSERT INTO note_embeddings (path, embedding) VALUES ('a.md', ?)",
        (b"\\x00" * 32,),
    )
    conn.commit()

    clear_all(conn)

    assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM notes_fts").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM tags").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM note_embeddings").fetchone()[0] == 0

    conn.execute(
        "INSERT INTO notes (path, type, title, confidence) VALUES ('x.md', 'concept', 'X', 'pattern')"
    )
    assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 1
