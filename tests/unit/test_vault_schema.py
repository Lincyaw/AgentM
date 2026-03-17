"""Tests for vault schema module — DDL creation, vec support probe, clear_all."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from agentm.tools.vault.schema import clear_all, create_schema, has_vec_support

_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _make_import_blocker(blocked_name: str):
    """Return an __import__ replacement that blocks one module."""
    def _blocker(name, *args, **kwargs):
        if name == blocked_name:
            raise ImportError(f"mocked: {name} not available")
        return _real_import(name, *args, **kwargs)
    return _blocker


@pytest.fixture
def conn():
    """In-memory SQLite connection for testing."""
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    """Return set of user table/virtual table names."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow') "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    # For virtual tables, also check directly
    vt_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    all_names = {r[0] for r in rows} | {r[0] for r in vt_rows}
    return all_names


def _index_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# create_schema
# ---------------------------------------------------------------------------


class TestCreateSchema:
    def test_should_create_notes_table(self, conn: sqlite3.Connection):
        create_schema(conn)
        # Verify notes table exists with expected columns
        info = conn.execute("PRAGMA table_info(notes)").fetchall()
        col_names = {row[1] for row in info}
        assert col_names == {
            "path",
            "type",
            "title",
            "confidence",
            "status",
            "created_at",
            "updated_at",
            "frontmatter",
            "body_hash",
        }

    def test_should_create_notes_table_with_correct_defaults(
        self, conn: sqlite3.Connection
    ):
        create_schema(conn)
        info = conn.execute("PRAGMA table_info(notes)").fetchall()
        status_col = [row for row in info if row[1] == "status"][0]
        # dflt_value is at index 4
        assert status_col[4] == "'active'"

    def test_should_create_fts5_table(self, conn: sqlite3.Connection):
        create_schema(conn)
        # FTS5 virtual tables appear in sqlite_master
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE name='notes_fts'"
        ).fetchall()
        assert len(rows) == 1

    def test_should_create_links_table_with_composite_pk(
        self, conn: sqlite3.Connection
    ):
        create_schema(conn)
        info = conn.execute("PRAGMA table_info(links)").fetchall()
        col_names = {row[1] for row in info}
        assert col_names == {"source", "target"}
        # pk columns indicated by non-zero pk field (index 5)
        pk_cols = {row[1] for row in info if row[5] > 0}
        assert pk_cols == {"source", "target"}

    def test_should_create_tags_table_with_composite_pk(
        self, conn: sqlite3.Connection
    ):
        create_schema(conn)
        info = conn.execute("PRAGMA table_info(tags)").fetchall()
        col_names = {row[1] for row in info}
        assert col_names == {"path", "tag"}
        pk_cols = {row[1] for row in info if row[5] > 0}
        assert pk_cols == {"path", "tag"}

    def test_should_create_indexes(self, conn: sqlite3.Connection):
        create_schema(conn)
        idx = _index_names(conn)
        assert "idx_links_target" in idx
        assert "idx_tags_tag" in idx

    def test_should_be_idempotent(self, conn: sqlite3.Connection):
        """Calling create_schema twice must not raise."""
        create_schema(conn)
        create_schema(conn)
        # Verify tables still intact
        info = conn.execute("PRAGMA table_info(notes)").fetchall()
        assert len(info) == 9  # 9 columns in notes


# ---------------------------------------------------------------------------
# has_vec_support
# ---------------------------------------------------------------------------


class TestHasVecSupport:
    def test_should_return_bool(self, conn: sqlite3.Connection):
        result = has_vec_support(conn)
        assert isinstance(result, bool)

    def test_should_never_raise(self, conn: sqlite3.Connection):
        """Even with a broken connection scenario, should not propagate."""
        result = has_vec_support(conn)
        assert result is False or result is True  # no exception

    def test_should_return_false_when_import_fails(self, conn: sqlite3.Connection):
        """Simulated import failure must yield False, not an exception."""
        with patch(
            "builtins.__import__",
            side_effect=_make_import_blocker("sqlite_vec"),
        ):
            result = has_vec_support(conn)
            assert result is False


# ---------------------------------------------------------------------------
# create_schema + vec interaction
# ---------------------------------------------------------------------------


class TestCreateSchemaVec:
    def test_should_skip_vec_table_when_unavailable(
        self, conn: sqlite3.Connection
    ):
        """When sqlite-vec import fails, notes_vec must not be created."""
        with patch(
            "agentm.tools.vault.schema.has_vec_support", return_value=False
        ):
            create_schema(conn)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE name='notes_vec'"
        ).fetchall()
        assert len(rows) == 0

    def test_should_create_vec_table_lazily_when_available(
        self, conn: sqlite3.Connection
    ):
        """notes_vec is NOT created by create_schema — it is created lazily
        on first embedding write. Verify the lazy path works."""
        if not has_vec_support(conn):
            pytest.skip("sqlite-vec not installed")
        create_schema(conn)
        # create_schema does NOT create notes_vec
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE name='notes_vec'"
        ).fetchall()
        assert len(rows) == 0
        # Lazy creation (as production code does)
        conn.executescript(
            "CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0("
            "path TEXT PRIMARY KEY, embedding FLOAT[512]);"
        )
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE name='notes_vec'"
        ).fetchall()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# clear_all
# ---------------------------------------------------------------------------


class TestClearAll:
    def test_should_empty_all_data_tables(self, conn: sqlite3.Connection):
        create_schema(conn)
        # Insert test data
        conn.execute(
            "INSERT INTO notes (path, type, title, confidence) "
            "VALUES ('a.md', 'skill', 'Test', 'fact')"
        )
        conn.execute(
            "INSERT INTO notes_fts (path, title, body, tags) "
            "VALUES ('a.md', 'Test', 'body text', 'tag1')"
        )
        conn.execute("INSERT INTO links (source, target) VALUES ('a.md', 'b.md')")
        conn.execute("INSERT INTO tags (path, tag) VALUES ('a.md', 'db')")
        conn.commit()

        clear_all(conn)

        assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM notes_fts").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM tags").fetchone()[0] == 0

    def test_should_preserve_schema_after_clear(self, conn: sqlite3.Connection):
        create_schema(conn)
        clear_all(conn)
        # Tables still exist — can insert again
        conn.execute(
            "INSERT INTO notes (path, type, title, confidence) "
            "VALUES ('x.md', 'concept', 'X', 'pattern')"
        )
        count = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
        assert count == 1

    def test_should_not_raise_on_empty_tables(self, conn: sqlite3.Connection):
        create_schema(conn)
        # Tables are already empty; clear should not raise
        clear_all(conn)

    def test_should_clear_vec_table_when_present(self, conn: sqlite3.Connection):
        """clear_all must also empty notes_vec if the table exists."""
        if not has_vec_support(conn):
            pytest.skip("sqlite-vec not installed")
        create_schema(conn)
        # Lazily create notes_vec (as production code does in store.py)
        conn.executescript(
            "CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0("
            "path TEXT PRIMARY KEY, embedding FLOAT[512]);"
        )
        conn.execute(
            "INSERT INTO notes_vec (path, embedding) VALUES (?, ?)",
            ("a.md", b"\x00" * 2048),  # 512 floats x 4 bytes
        )
        conn.commit()
        clear_all(conn)
        count = conn.execute("SELECT count(*) FROM notes_vec").fetchone()[0]
        assert count == 0
