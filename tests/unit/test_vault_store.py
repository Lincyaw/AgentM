"""Focused regression tests for MarkdownVault core behaviors."""

from __future__ import annotations

import threading

import pytest

from agentm.tools.vault.store import MarkdownVault


@pytest.fixture
def vault(tmp_path):
    return MarkdownVault(tmp_path)


def _read_raw(vault: MarkdownVault, path: str) -> str:
    return (vault._vault_dir / f"{path}.md").read_text(encoding="utf-8")


def _db_row(vault: MarkdownVault, table: str, path: str) -> dict | None:
    conn = vault._get_conn()
    row = conn.execute(f"SELECT * FROM {table} WHERE path = ?", (path,)).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
    return dict(zip(cols, row))


def test_init_creates_schema_and_uses_wal_mode(vault: MarkdownVault) -> None:
    conn = vault._get_conn()
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    assert {"notes", "links", "tags"}.issubset(tables)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_write_indexes_note_links_tags_and_fts(vault: MarkdownVault) -> None:
    warnings = vault.write(
        "skill/timeout",
        {"type": "skill", "confidence": "fact", "tags": ["db", "timeout"]},
        "# Timeout\n\nSee [[concept/pool]].",
    )

    assert any("Dead link" in w for w in warnings)
    row = _db_row(vault, "notes", "skill/timeout")
    assert row is not None
    assert row["title"] == "Timeout"
    assert row["body_hash"] and len(row["body_hash"]) == 64

    conn = vault._get_conn()
    assert conn.execute("SELECT target FROM links WHERE source = ?", ("skill/timeout",)).fetchone()[0] == "concept/pool"
    tags = conn.execute("SELECT tag FROM tags WHERE path = ? ORDER BY tag", ("skill/timeout",)).fetchall()
    assert [t[0] for t in tags] == ["db", "timeout"]
    assert conn.execute("SELECT count(*) FROM notes_fts WHERE notes_fts MATCH 'timeout'").fetchone()[0] == 1


def test_overwrite_replaces_links(vault: MarkdownVault) -> None:
    vault.write("x", {"type": "skill", "confidence": "fact"}, "# X\n[[a]]")
    vault.write("x", {"type": "skill", "confidence": "fact"}, "# X\n[[b]]")

    targets = vault._get_conn().execute(
        "SELECT target FROM links WHERE source = ?", ("x",)
    ).fetchall()
    assert [r[0] for r in targets] == ["b"]


def test_read_returns_none_for_missing(vault: MarkdownVault) -> None:
    assert vault.read("missing") is None


def test_edit_reindexes_links_and_rejects_unknown_operation(vault: MarkdownVault) -> None:
    vault.write("e", {"type": "skill", "confidence": "fact"}, "# E\n[[old_target]]")

    vault.edit("e", "replace_string", {"old": "[[old_target]]", "new": "[[new_target]]"})
    targets = vault._get_conn().execute(
        "SELECT target FROM links WHERE source = ?", ("e",)
    ).fetchall()
    assert [r[0] for r in targets] == ["new_target"]

    with pytest.raises(ValueError, match="Unknown edit operation"):
        vault.edit("e", "bad_op", {})


def test_delete_removes_file_and_index_entries(vault: MarkdownVault) -> None:
    vault.write("d", {"type": "skill", "confidence": "fact", "tags": ["x"]}, "# D\n[[link]]")
    vault.delete("d")

    assert not (vault._vault_dir / "d.md").exists()
    assert _db_row(vault, "notes", "d") is None
    conn = vault._get_conn()
    assert conn.execute("SELECT count(*) FROM links WHERE source = ?", ("d",)).fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM tags WHERE path = ?", ("d",)).fetchone()[0] == 0


def test_rename_rewrites_backlinks_and_updates_index(vault: MarkdownVault) -> None:
    vault.write("target", {"type": "skill", "confidence": "fact"}, "# Target")
    vault.write("ref", {"type": "concept", "confidence": "fact"}, "# Ref\n[[target]]")

    vault.rename("target", "new_target")

    assert vault.read("target") is None
    assert vault.read("new_target") is not None
    assert "[[new_target]]" in _read_raw(vault, "ref")

    targets = vault._get_conn().execute(
        "SELECT target FROM links WHERE source = ?", ("ref",)
    ).fetchall()
    assert [r[0] for r in targets] == ["new_target"]


def test_write_batch_handles_cross_references_without_dead_link_warnings(vault: MarkdownVault) -> None:
    warnings = vault.write_batch(
        [
            {
                "path": "a",
                "frontmatter": {"type": "skill", "confidence": "fact"},
                "body": "# A\n[[b]]",
            },
            {
                "path": "b",
                "frontmatter": {"type": "concept", "confidence": "pattern"},
                "body": "# B\n[[a]]",
            },
        ]
    )

    assert not any("Dead link" in w for w in warnings)
    assert vault._get_conn().execute("SELECT count(*) FROM notes").fetchone()[0] == 2


def test_list_notes_respects_depth_and_type_filter(vault: MarkdownVault) -> None:
    vault.write("a", {"type": "skill", "confidence": "fact"}, "# A")
    vault.write("sub/b", {"type": "skill", "confidence": "fact"}, "# B")
    vault.write("sub/c", {"type": "concept", "confidence": "fact"}, "# C")

    root_depth_1 = {n["path"] for n in vault.list_notes("", depth=1)}
    assert "a" in root_depth_1
    assert "sub/b" not in root_depth_1

    skills = {n["path"] for n in vault.list_notes("sub", depth=2, type_filter="skill")}
    assert skills == {"sub/b"}


def test_rebuild_index_restores_rows_after_manual_index_corruption(vault: MarkdownVault) -> None:
    vault.write("a", {"type": "skill", "confidence": "fact", "tags": ["t"]}, "# A\n[[b]]")
    vault.write("b", {"type": "concept", "confidence": "fact"}, "# B")

    conn = vault._get_conn()
    for table in ("notes", "notes_fts", "links", "tags"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()

    count = vault.rebuild_index()
    assert count == 2
    assert _db_row(vault, "notes", "a") is not None
    assert _db_row(vault, "notes", "b") is not None


def test_validation_reports_structural_issues_and_allows_valid_note(vault: MarkdownVault) -> None:
    warnings = vault.write("bad", {}, "No heading")
    assert any("Missing required frontmatter field: 'type'" in w for w in warnings)
    assert any("Missing required frontmatter field: 'confidence'" in w for w in warnings)
    assert any("H1" in w for w in warnings)

    vault.write("concept/other", {"type": "concept", "confidence": "fact"}, "# Other")
    clean = vault.write("good", {"type": "skill", "confidence": "pattern"}, "# Good\n[[concept/other]]")
    assert clean == []


def test_concurrent_writes_do_not_corrupt_index(vault: MarkdownVault) -> None:
    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            vault.write(
                f"note-{i}",
                {"type": "skill", "confidence": "fact"},
                f"# Note {i}",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert vault._get_conn().execute("SELECT count(*) FROM notes").fetchone()[0] == 8
