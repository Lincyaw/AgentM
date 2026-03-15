"""Integration tests for Knowledge Store — file-system backend.

Bug prevented: Knowledge tools write/read from disk but may silently lose data
(bad path mapping, stale cache, broken index). These tests verify end-to-end
correctness against a real temp directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agentm.tools import knowledge as knowledge_module
from agentm.tools.knowledge import (
    knowledge_delete,
    knowledge_list,
    knowledge_read,
    knowledge_search,
    knowledge_write,
)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path: Path):
    """Initialise a fresh file-system store in a temp directory."""
    knowledge_module.init(base_dir=str(tmp_path / "knowledge"))
    yield tmp_path / "knowledge"
    # Reset module state
    knowledge_module._base_dir = None
    knowledge_module._entries = {}
    knowledge_module._inv_index = {}
    knowledge_module._embeddings = {}


def _sample_entry(**overrides: object) -> dict[str, object]:
    """Create a sample knowledge entry dict."""
    entry: dict[str, object] = {
        "title": "Connection pool exhaustion",
        "description": "Database connection pool reaches max connections under high load",
        "category": "failure-patterns",
        "tags": ["database", "connection-pool", "timeout"],
        "confidence": 0.9,
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# Write & Read
# ---------------------------------------------------------------------------


class TestWriteAndRead:
    """Write -> Read round-trip verification."""

    def test_write_then_read_returns_entry(self) -> None:
        """Written entry can be read back by exact path."""
        entry = _sample_entry()
        result = knowledge_write("/failure-patterns/pool-exhaustion", entry)
        assert "Written" in result

        read_result = json.loads(knowledge_read("/failure-patterns/pool-exhaustion"))
        assert read_result["title"] == "Connection pool exhaustion"
        assert read_result["confidence"] == 0.9

    def test_read_nonexistent_returns_error(self) -> None:
        """Reading a non-existent path returns error dict."""
        result = json.loads(knowledge_read("/failure-patterns/nonexistent"))
        assert "error" in result
        assert "Not found" in result["error"]

    def test_write_overwrites_existing(self) -> None:
        """Second write to same path overwrites the entry."""
        knowledge_write("/failure-patterns/test", _sample_entry(title="v1"))
        knowledge_write("/failure-patterns/test", _sample_entry(title="v2"))

        result = json.loads(knowledge_read("/failure-patterns/test"))
        assert result["title"] == "v2"

    def test_write_creates_json_file(self, fresh_store: Path) -> None:
        """Write creates the expected .json file on disk."""
        knowledge_write("/failure-patterns/disk-check", _sample_entry())
        expected = fresh_store / "failure-patterns" / "disk-check.json"
        assert expected.exists()
        data = json.loads(expected.read_text())
        assert data["title"] == "Connection pool exhaustion"


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    """Verify knowledge_delete."""

    def test_delete_removes_entry(self) -> None:
        """Deleted entry is no longer readable."""
        knowledge_write("/failure-patterns/to-delete", _sample_entry())

        delete_result = knowledge_delete("/failure-patterns/to-delete")
        assert "Deleted" in delete_result

        read_result = json.loads(knowledge_read("/failure-patterns/to-delete"))
        assert "error" in read_result
        assert "Not found" in read_result["error"]

    def test_delete_nonexistent_does_not_raise(self) -> None:
        """Deleting a non-existent entry does not raise an error."""
        result = knowledge_delete("/failure-patterns/never-existed")
        assert "Deleted" in result

    def test_delete_removes_file(self, fresh_store: Path) -> None:
        """Delete removes the .json file from disk."""
        knowledge_write("/failure-patterns/fs-delete", _sample_entry())
        fs_path = fresh_store / "failure-patterns" / "fs-delete.json"
        assert fs_path.exists()

        knowledge_delete("/failure-patterns/fs-delete")
        assert not fs_path.exists()


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestListNamespace:
    """Verify knowledge_list with depth control."""

    def test_list_shows_written_entries(self) -> None:
        """Written entries appear in list results."""
        knowledge_write("/failure-patterns/pool", _sample_entry(title="Pool"))
        knowledge_write("/failure-patterns/locks", _sample_entry(title="Locks"))

        result = json.loads(knowledge_list("list", path="/failure-patterns", depth=1))
        assert result["path"] == "/failure-patterns"
        titles = [e["title"] for e in result["entries"]]
        assert "Pool" in titles
        assert "Locks" in titles

    def test_list_root_shows_categories(self) -> None:
        """Listing root shows top-level categories as sub_paths."""
        knowledge_write("/failure-patterns/test", _sample_entry())
        knowledge_write("/diagnostic-workflows/test", _sample_entry(title="Skill"))

        result = json.loads(knowledge_list("list", path="/"))
        sub_paths = result["sub_paths"]
        assert any("failure-patterns" in p for p in sub_paths)
        assert any("diagnostic-workflows" in p for p in sub_paths)

    def test_list_empty_path(self) -> None:
        """Listing a path with no entries returns empty."""
        result = json.loads(knowledge_list("list", path="/nonexistent"))
        assert result["entries"] == []

    def test_list_depth_zero_unlimited(self) -> None:
        """depth=0 returns all entries recursively."""
        knowledge_write("/a/b", _sample_entry(title="B"))
        knowledge_write("/a/c", _sample_entry(title="C"))

        result = json.loads(knowledge_list("list", path="/", depth=0))
        titles = [e["title"] for e in result["entries"]]
        assert "B" in titles
        assert "C" in titles

    def test_list_depth_limits_results(self) -> None:
        """depth=1 at root only shows immediate children (sub_paths)."""
        knowledge_write("/cat/sub/deep", _sample_entry(title="Deep"))

        result = json.loads(knowledge_list("list", path="/", depth=1))
        # At depth 1 from root, /cat/sub/deep is too deep — shows as sub_path
        assert any("cat" in p for p in result["sub_paths"])


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    """Verify keyword-based search."""

    def test_search_finds_by_title_keyword(self) -> None:
        knowledge_write(
            "/failure-patterns/pool", _sample_entry(title="Pool exhaustion")
        )
        knowledge_write(
            "/failure-patterns/locks", _sample_entry(title="Lock contention")
        )

        results = json.loads(knowledge_search("pool", path="/", mode="keyword"))
        paths = [r["path"] for r in results]
        assert "/failure-patterns/pool" in paths

    def test_search_and_semantics(self) -> None:
        """All query tokens must match (AND semantics)."""
        knowledge_write(
            "/failure-patterns/pool",
            _sample_entry(title="Connection pool exhaustion"),
        )
        knowledge_write(
            "/failure-patterns/other",
            {
                "title": "Something else entirely",
                "description": "unrelated topic about networking",
                "category": "failure-patterns",
                "tags": ["network"],
                "confidence": 0.5,
            },
        )

        results = json.loads(
            knowledge_search("connection pool", path="/", mode="keyword")
        )
        paths = [r["path"] for r in results]
        assert "/failure-patterns/pool" in paths
        assert "/failure-patterns/other" not in paths

    def test_search_with_path_prefix(self) -> None:
        """Search scoped to a path prefix filters results."""
        knowledge_write("/failure-patterns/pool", _sample_entry(title="Pool"))
        knowledge_write("/diagnostics/pool", _sample_entry(title="Pool diag"))

        results = json.loads(
            knowledge_search("pool", path="/failure-patterns", mode="keyword")
        )
        paths = [r["path"] for r in results]
        assert "/failure-patterns/pool" in paths
        assert "/diagnostics/pool" not in paths

    def test_search_no_results(self) -> None:
        """Search with no matches returns 'No results found.'."""
        result = knowledge_search("nonexistent-query-xyz", mode="keyword")
        assert result == "No results found."


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    """Verify semantic search (with mocked model)."""

    def test_semantic_fallback_to_keyword_when_model_unavailable(self) -> None:
        """When embedding model fails, semantic search degrades to keyword."""
        knowledge_write(
            "/failure-patterns/pool",
            _sample_entry(title="Connection pool exhaustion"),
        )

        with patch.object(knowledge_module, "_get_model", return_value=None):
            results = json.loads(
                knowledge_search("connection pool", path="/", mode="semantic")
            )
            paths = [r["path"] for r in results]
            assert "/failure-patterns/pool" in paths


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """Verify hybrid search (RRF fusion)."""

    def test_hybrid_returns_results(self) -> None:
        """Hybrid search returns results combining keyword and semantic."""
        knowledge_write(
            "/failure-patterns/pool",
            _sample_entry(title="Connection pool exhaustion"),
        )

        # With model unavailable, hybrid = keyword + keyword fallback
        with patch.object(knowledge_module, "_get_model", return_value=None):
            results = json.loads(
                knowledge_search("connection pool", path="/", mode="hybrid")
            )
            assert len(results) > 0
            paths = [r["path"] for r in results]
            assert "/failure-patterns/pool" in paths

    def test_hybrid_degrades_gracefully(self) -> None:
        """Hybrid search works even when semantic component fails."""
        knowledge_write(
            "/failure-patterns/test",
            _sample_entry(title="Test entry keyword match"),
        )

        with patch.object(knowledge_module, "_get_model", return_value=None):
            results = json.loads(
                knowledge_search("test keyword", path="/", mode="hybrid")
            )
            assert len(results) > 0


# ---------------------------------------------------------------------------
# Not initialized
# ---------------------------------------------------------------------------


class TestNotInitialized:
    """Verify proper error when init() has not been called."""

    def test_read_without_init_raises(self) -> None:
        knowledge_module._base_dir = None
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_read("/any/path")

    def test_write_without_init_raises(self) -> None:
        knowledge_module._base_dir = None
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_write("/any/path", {"title": "test"})

    def test_delete_without_init_raises(self) -> None:
        knowledge_module._base_dir = None
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_delete("/any/path")

    def test_search_without_init_raises(self) -> None:
        knowledge_module._base_dir = None
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_search("query")

    def test_list_without_init_raises(self) -> None:
        knowledge_module._base_dir = None
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_list("list", path="/")


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


class TestFilePersistence:
    """Write -> re-init -> read verifies on-disk persistence."""

    def test_write_survives_reinit(self, tmp_path: Path) -> None:
        """Data written survives a full re-init from disk."""
        base = str(tmp_path / "persist-test")
        knowledge_module.init(base_dir=base)
        knowledge_write("/failure-patterns/persist", _sample_entry(title="Persistent"))

        # Re-init from scratch
        knowledge_module.init(base_dir=base)
        result = json.loads(knowledge_read("/failure-patterns/persist"))
        assert result["title"] == "Persistent"

    def test_delete_survives_reinit(self, tmp_path: Path) -> None:
        """Deleted entry stays deleted after re-init."""
        base = str(tmp_path / "delete-persist")
        knowledge_module.init(base_dir=base)
        knowledge_write("/failure-patterns/gone", _sample_entry())
        knowledge_delete("/failure-patterns/gone")

        knowledge_module.init(base_dir=base)
        result = json.loads(knowledge_read("/failure-patterns/gone"))
        assert "error" in result


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------


class TestLegacyMigration:
    """Verify migration from flat knowledge.json."""

    def test_migrates_legacy_json(self, tmp_path: Path) -> None:
        """Legacy knowledge.json is migrated to per-entry files."""
        base = tmp_path / "legacy-test"
        base.mkdir()
        legacy = base / "knowledge.json"
        legacy.write_text(
            json.dumps(
                [
                    {
                        "path": "/failure-patterns/pool",
                        "title": "Pool issue",
                        "description": "Connection pool problems",
                    }
                ]
            )
        )

        knowledge_module.init(base_dir=str(base))

        # Entry should be readable
        result = json.loads(knowledge_read("/failure-patterns/pool"))
        assert result["title"] == "Pool issue"

        # Legacy file should be renamed
        assert not legacy.exists()
        assert (base / "knowledge.json.migrated").exists()
