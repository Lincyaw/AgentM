"""Tests for vault store module -- MarkdownVault CRUD, edit, rename, batch, index."""

from __future__ import annotations

import threading

import pytest

from agentm.tools.vault.store import MarkdownVault


@pytest.fixture
def vault(tmp_path):
    """Create a MarkdownVault in a temporary directory."""
    return MarkdownVault(tmp_path)


def _read_raw(vault: MarkdownVault, path: str) -> str:
    """Read the raw .md file content from disk."""
    fp = vault._vault_dir / (path + ".md")
    return fp.read_text(encoding="utf-8")


def _db_row(vault: MarkdownVault, table: str, path: str) -> dict | None:
    """Fetch one row from a table by path column."""
    conn = vault._get_conn()
    row = conn.execute(f"SELECT * FROM {table} WHERE path = ?", (path,)).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_should_create_vault_dir_if_missing(self, tmp_path):
        sub = tmp_path / "deep" / "vault"
        v = MarkdownVault(sub)
        assert sub.is_dir()
        assert (sub / ".vault.db").exists() or v._get_conn() is not None

    def test_should_create_db_with_schema(self, vault: MarkdownVault):
        conn = vault._get_conn()
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "notes" in tables
        assert "links" in tables
        assert "tags" in tables

    def test_should_use_wal_mode(self, vault: MarkdownVault):
        conn = vault._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class TestWrite:
    def test_should_create_md_file(self, vault: MarkdownVault):
        vault.write("skill/timeout", {"type": "skill", "tags": ["db"]}, "# Timeout\n\nBody.")
        content = _read_raw(vault, "skill/timeout")
        assert "# Timeout" in content
        assert "type: skill" in content

    def test_should_create_subdirectories(self, vault: MarkdownVault):
        vault.write("deep/nested/note", {}, "hello")
        assert (vault._vault_dir / "deep" / "nested" / "note.md").exists()

    def test_should_index_notes_table(self, vault: MarkdownVault):
        vault.write(
            "skill/x",
            {"type": "skill", "confidence": "fact", "tags": ["a"]},
            "# X Title\n\nBody text.",
        )
        row = _db_row(vault, "notes", "skill/x")
        assert row is not None
        assert row["type"] == "skill"
        assert row["title"] == "X Title"
        assert row["confidence"] == "fact"

    def test_should_index_links(self, vault: MarkdownVault):
        vault.write("a", {}, "See [[b]] and [[c/d]].")
        conn = vault._get_conn()
        links = conn.execute(
            "SELECT target FROM links WHERE source = ? ORDER BY target", ("a",)
        ).fetchall()
        assert [r[0] for r in links] == ["b", "c/d"]

    def test_should_index_tags(self, vault: MarkdownVault):
        vault.write("t", {"tags": ["alpha", "beta"]}, "body")
        conn = vault._get_conn()
        tags = conn.execute(
            "SELECT tag FROM tags WHERE path = ? ORDER BY tag", ("t",)
        ).fetchall()
        assert [r[0] for r in tags] == ["alpha", "beta"]

    def test_should_index_fts(self, vault: MarkdownVault):
        vault.write("f", {"tags": ["search"]}, "# Title\n\nSearchable body text.")
        conn = vault._get_conn()
        rows = conn.execute(
            "SELECT path FROM notes_fts WHERE notes_fts MATCH 'searchable'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "f"

    def test_should_be_atomic_via_tmp_file(self, vault: MarkdownVault, tmp_path):
        """After write, no .tmp files should remain."""
        vault.write("atomic", {}, "content")
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == []

    def test_should_overwrite_existing_note(self, vault: MarkdownVault):
        vault.write("x", {"type": "skill"}, "v1")
        vault.write("x", {"type": "concept"}, "v2")
        row = _db_row(vault, "notes", "x")
        assert row["type"] == "concept"
        content = _read_raw(vault, "x")
        assert "v2" in content

    def test_should_update_links_on_overwrite(self, vault: MarkdownVault):
        vault.write("x", {}, "[[a]]")
        vault.write("x", {}, "[[b]]")
        conn = vault._get_conn()
        targets = conn.execute(
            "SELECT target FROM links WHERE source = ?", ("x",)
        ).fetchall()
        assert [r[0] for r in targets] == ["b"]

    def test_should_store_body_hash(self, vault: MarkdownVault):
        vault.write("h", {}, "some body")
        row = _db_row(vault, "notes", "h")
        assert row["body_hash"] is not None
        assert len(row["body_hash"]) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


class TestRead:
    def test_should_return_parsed_note(self, vault: MarkdownVault):
        vault.write("r", {"type": "skill", "tags": ["a"]}, "# R\n\nBody.")
        result = vault.read("r")
        assert result is not None
        assert result["frontmatter"]["type"] == "skill"
        assert "# R" in result["body"]

    def test_should_return_none_for_missing(self, vault: MarkdownVault):
        assert vault.read("nonexistent") is None

    def test_should_include_path(self, vault: MarkdownVault):
        vault.write("p", {}, "body")
        result = vault.read("p")
        assert result["path"] == "p"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TestEdit:
    def test_should_replace_string(self, vault: MarkdownVault):
        vault.write("e", {}, "The value is 100ms.")
        vault.edit("e", "replace_string", {"old": "100ms", "new": "200ms"})
        content = _read_raw(vault, "e")
        assert "200ms" in content
        assert "100ms" not in content

    def test_should_set_frontmatter(self, vault: MarkdownVault):
        vault.write("e", {"type": "skill", "confidence": "heuristic"}, "body")
        vault.edit("e", "set_frontmatter", {"confidence": "fact", "status": "archived"})
        result = vault.read("e")
        assert result["frontmatter"]["confidence"] == "fact"
        assert result["frontmatter"]["status"] == "archived"
        # Unspecified fields preserved
        assert result["frontmatter"]["type"] == "skill"

    def test_should_replace_section(self, vault: MarkdownVault):
        body = "# Title\n\n## Evidence\n- old evidence\n\n## Related\n- links\n"
        vault.write("e", {}, body)
        vault.edit("e", "replace_section", {
            "heading": "## Evidence",
            "body": "- new evidence\n\n",
        })
        content = _read_raw(vault, "e")
        assert "new evidence" in content
        assert "old evidence" not in content
        assert "## Related" in content

    def test_should_append_section(self, vault: MarkdownVault):
        body = "# Title\n\n## Evidence\n- item1\n\n## Related\n- links\n"
        vault.write("e", {}, body)
        vault.edit("e", "append_section", {
            "heading": "## Evidence",
            "content": "- item2\n",
        })
        content = _read_raw(vault, "e")
        assert "item1" in content
        assert "item2" in content

    def test_should_reindex_after_edit(self, vault: MarkdownVault):
        vault.write("e", {}, "See [[old_target]].")
        vault.edit("e", "replace_string", {"old": "[[old_target]]", "new": "[[new_target]]"})
        conn = vault._get_conn()
        targets = conn.execute(
            "SELECT target FROM links WHERE source = ?", ("e",)
        ).fetchall()
        assert [r[0] for r in targets] == ["new_target"]

    def test_should_raise_on_unknown_operation(self, vault: MarkdownVault):
        vault.write("e", {}, "body")
        with pytest.raises(ValueError, match="Unknown edit operation"):
            vault.edit("e", "invalid_op", {})

    def test_should_raise_on_missing_note(self, vault: MarkdownVault):
        with pytest.raises(FileNotFoundError):
            vault.edit("missing", "replace_string", {"old": "a", "new": "b"})


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_should_remove_file(self, vault: MarkdownVault):
        vault.write("d", {}, "body")
        vault.delete("d")
        assert not (vault._vault_dir / "d.md").exists()

    def test_should_remove_index_entries(self, vault: MarkdownVault):
        vault.write("d", {"tags": ["x"]}, "[[link]]")
        vault.delete("d")
        conn = vault._get_conn()
        assert _db_row(vault, "notes", "d") is None
        assert conn.execute("SELECT count(*) FROM links WHERE source = ?", ("d",)).fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM tags WHERE path = ?", ("d",)).fetchone()[0] == 0

    def test_should_raise_on_missing(self, vault: MarkdownVault):
        with pytest.raises(FileNotFoundError):
            vault.delete("nonexistent")


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------


class TestRename:
    def test_should_move_file(self, vault: MarkdownVault):
        vault.write("old", {}, "body")
        vault.rename("old", "new")
        assert not (vault._vault_dir / "old.md").exists()
        assert (vault._vault_dir / "new.md").exists()

    def test_should_update_index(self, vault: MarkdownVault):
        vault.write("old", {"type": "skill"}, "body")
        vault.rename("old", "new")
        assert _db_row(vault, "notes", "old") is None
        row = _db_row(vault, "notes", "new")
        assert row is not None
        assert row["type"] == "skill"

    def test_should_rewrite_backlinks_in_referencing_files(self, vault: MarkdownVault):
        vault.write("target", {}, "I am target.")
        vault.write("ref1", {}, "See [[target]] here.")
        vault.write("ref2", {}, "Also [[target]].")
        vault.rename("target", "new_target")
        assert "[[new_target]]" in _read_raw(vault, "ref1")
        assert "[[new_target]]" in _read_raw(vault, "ref2")
        assert "[[target]]" not in _read_raw(vault, "ref1")

    def test_should_update_links_table_for_referencing_files(self, vault: MarkdownVault):
        vault.write("target", {}, "body")
        vault.write("ref", {}, "[[target]]")
        vault.rename("target", "new_target")
        conn = vault._get_conn()
        targets = conn.execute(
            "SELECT target FROM links WHERE source = ?", ("ref",)
        ).fetchall()
        assert [r[0] for r in targets] == ["new_target"]

    def test_should_raise_on_missing_source(self, vault: MarkdownVault):
        with pytest.raises(FileNotFoundError):
            vault.rename("nonexistent", "new")

    def test_should_create_target_subdirectories(self, vault: MarkdownVault):
        vault.write("flat", {}, "body")
        vault.rename("flat", "deep/nested/note")
        assert (vault._vault_dir / "deep" / "nested" / "note.md").exists()


# ---------------------------------------------------------------------------
# write_batch
# ---------------------------------------------------------------------------


class TestWriteBatch:
    def test_should_write_multiple_notes(self, vault: MarkdownVault):
        entries = [
            {"path": "a", "frontmatter": {"type": "skill"}, "body": "A body"},
            {"path": "b", "frontmatter": {"type": "concept"}, "body": "B body"},
        ]
        vault.write_batch(entries)
        assert vault.read("a") is not None
        assert vault.read("b") is not None

    def test_should_index_all_entries(self, vault: MarkdownVault):
        entries = [
            {"path": "x", "frontmatter": {"tags": ["t1"]}, "body": "[[y]]"},
            {"path": "y", "frontmatter": {"tags": ["t2"]}, "body": "[[x]]"},
        ]
        vault.write_batch(entries)
        conn = vault._get_conn()
        assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM tags").fetchone()[0] == 2

    def test_should_handle_empty_batch(self, vault: MarkdownVault):
        vault.write_batch([])  # no error


# ---------------------------------------------------------------------------
# list_notes
# ---------------------------------------------------------------------------


class TestListNotes:
    def test_should_list_top_level(self, vault: MarkdownVault):
        vault.write("a", {"type": "skill"}, "body")
        vault.write("b", {"type": "concept"}, "body")
        vault.write("sub/c", {"type": "skill"}, "body")
        result = vault.list_notes("", depth=1)
        paths = {r["path"] for r in result}
        assert "a" in paths
        assert "b" in paths
        # sub/c is depth 2 from root, should not appear at depth=1
        # unless path prefix matches — depends on implementation
        # At depth=1 from "", sub/c has 2 parts so excluded

    def test_should_list_subdirectory(self, vault: MarkdownVault):
        vault.write("skill/x", {}, "body")
        vault.write("skill/y", {}, "body")
        vault.write("concept/z", {}, "body")
        result = vault.list_notes("skill", depth=1)
        paths = {r["path"] for r in result}
        assert paths == {"skill/x", "skill/y"}

    def test_should_filter_by_type(self, vault: MarkdownVault):
        vault.write("a", {"type": "skill"}, "body")
        vault.write("b", {"type": "concept"}, "body")
        result = vault.list_notes("", depth=10, type_filter="skill")
        paths = {r["path"] for r in result}
        assert "a" in paths
        assert "b" not in paths

    def test_should_return_empty_for_no_matches(self, vault: MarkdownVault):
        result = vault.list_notes("nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# rebuild_index
# ---------------------------------------------------------------------------


class TestRebuildIndex:
    def test_should_reindex_all_files(self, vault: MarkdownVault):
        vault.write("a", {"type": "skill", "tags": ["t"]}, "[[b]]")
        vault.write("b", {"type": "concept"}, "body")
        # Corrupt index by clearing
        conn = vault._get_conn()
        conn.execute("DELETE FROM notes")
        conn.execute("DELETE FROM links")
        conn.execute("DELETE FROM tags")
        conn.execute("DELETE FROM notes_fts")
        conn.commit()
        # Rebuild
        count = vault.rebuild_index()
        assert count == 2
        assert _db_row(vault, "notes", "a") is not None
        assert _db_row(vault, "notes", "b") is not None
        links = conn.execute("SELECT target FROM links WHERE source = 'a'").fetchall()
        assert len(links) == 1

    def test_should_return_zero_for_empty_vault(self, vault: MarkdownVault):
        assert vault.rebuild_index() == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_should_handle_concurrent_writes(self, vault: MarkdownVault):
        """Multiple threads writing should not corrupt the vault."""
        errors = []

        def writer(i: int):
            try:
                vault.write(f"note-{i}", {"type": "skill"}, f"Body {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        conn = vault._get_conn()
        count = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
        assert count == 10


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_should_warn_on_missing_required_fields(self, vault: MarkdownVault):
        warnings = vault.write("test/note", {}, "# Title\n\nBody")
        assert any("type" in w for w in warnings)
        assert any("confidence" in w for w in warnings)

    def test_should_warn_on_unknown_type(self, vault: MarkdownVault):
        fm = {"type": "invalid_type", "confidence": "fact"}
        warnings = vault.write("test/note", fm, "# Title\n\nBody")
        assert any("Unknown type" in w for w in warnings)

    def test_should_warn_on_unknown_confidence(self, vault: MarkdownVault):
        fm = {"type": "skill", "confidence": "maybe"}
        warnings = vault.write("test/note", fm, "# Title\n\nBody")
        assert any("Unknown confidence" in w for w in warnings)

    def test_should_warn_on_missing_h1(self, vault: MarkdownVault):
        fm = {"type": "skill", "confidence": "fact"}
        warnings = vault.write("test/note", fm, "No heading here")
        assert any("H1" in w for w in warnings)

    def test_should_warn_on_dead_wikilink(self, vault: MarkdownVault):
        fm = {"type": "skill", "confidence": "fact"}
        warnings = vault.write("test/note", fm, "# Title\n\n[[nonexistent/target]]")
        assert any("Dead link" in w for w in warnings)

    def test_should_not_warn_on_valid_note(self, vault: MarkdownVault):
        vault.write("concept/other", {"type": "concept", "confidence": "fact"}, "# Other\n\n")
        fm = {"type": "skill", "confidence": "pattern"}
        warnings = vault.write("test/note", fm, "# Title\n\n[[concept/other]]")
        assert warnings == []

    def test_should_return_warnings_from_edit(self, vault: MarkdownVault):
        fm = {"type": "skill", "confidence": "fact"}
        vault.write("test/note", fm, "# Title\n\nBody text")
        warnings = vault.edit("test/note", "set_frontmatter", {"confidence": "bogus"})
        assert any("Unknown confidence" in w for w in warnings)

    def test_batch_should_not_warn_on_cross_references(self, vault: MarkdownVault):
        """Batch write: notes referencing each other within the batch should not warn."""
        entries = [
            {"path": "a", "frontmatter": {"type": "skill", "confidence": "fact"}, "body": "# A\n\n[[b]]"},
            {"path": "b", "frontmatter": {"type": "concept", "confidence": "pattern"}, "body": "# B\n\n[[a]]"},
        ]
        warnings = vault.write_batch(entries)
        assert not any("Dead link" in w for w in warnings)
