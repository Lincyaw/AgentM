"""Integration tests for Knowledge Store tools with InMemoryStore backend.

Bug prevented: Knowledge tools accept a Store but may fail on real Store operations
(namespace handling, search API, key resolution). These tests verify end-to-end
correctness against a real InMemoryStore instance.
"""
from __future__ import annotations

import pytest
from langgraph.store.memory import InMemoryStore

from agentm.tools.knowledge import (
    knowledge_delete,
    knowledge_list,
    knowledge_read,
    knowledge_search,
    knowledge_write,
    set_store,
)


@pytest.fixture(autouse=True)
def fresh_store():
    """Inject a fresh InMemoryStore before each test, clean up after."""
    store = InMemoryStore()
    set_store(store)
    yield store
    set_store(None)


def _sample_entry(**overrides: object) -> dict[str, object]:
    """Create a sample knowledge entry dict."""
    entry: dict[str, object] = {
        "title": "Connection pool exhaustion",
        "description": "Database connection pool reaches max connections under high load",
        "tags": ["database", "connection_pool", "timeout"],
        "confidence": "fact",
        "frequency": 3,
        "domain": "database",
    }
    entry.update(overrides)
    return entry


class TestWriteAndRead:
    """Write -> Read round-trip verification."""

    def test_write_then_read_returns_entry(self) -> None:
        """Written entry can be read back by exact path."""
        entry = _sample_entry()
        result = knowledge_write("/failure_pattern/database/pool_exhaustion", entry)
        assert "Written" in result

        read_result = knowledge_read("/failure_pattern/database/pool_exhaustion")
        assert read_result["title"] == "Connection pool exhaustion"
        assert read_result["confidence"] == "fact"

    def test_read_nonexistent_returns_error(self) -> None:
        """Reading a non-existent path returns error dict."""
        result = knowledge_read("/failure_pattern/nonexistent/entry")
        assert "error" in result
        assert "Not found" in result["error"]

    def test_write_overwrites_existing(self) -> None:
        """Second write to same path overwrites the entry."""
        knowledge_write("/failure_pattern/db/test", _sample_entry(title="v1"))
        knowledge_write("/failure_pattern/db/test", _sample_entry(title="v2"))

        result = knowledge_read("/failure_pattern/db/test")
        assert result["title"] == "v2"

    def test_write_with_merge(self) -> None:
        """Write with merge=True merges into existing entry."""
        knowledge_write("/failure_pattern/db/merge_test", _sample_entry(frequency=1))
        knowledge_write(
            "/failure_pattern/db/merge_test",
            {"frequency": 5, "new_field": "added"},
            merge=True,
        )

        result = knowledge_read("/failure_pattern/db/merge_test")
        assert result["frequency"] == 5
        assert result["new_field"] == "added"
        # Original fields preserved after merge
        assert result["title"] == "Connection pool exhaustion"

    def test_write_with_merge_on_nonexistent_creates_entry(self) -> None:
        """Merge on a path with no existing entry creates the entry as-is."""
        knowledge_write(
            "/failure_pattern/db/new_merge",
            {"title": "Brand new", "description": "Fresh entry"},
            merge=True,
        )

        result = knowledge_read("/failure_pattern/db/new_merge")
        assert result["title"] == "Brand new"


class TestDelete:
    """Verify knowledge_delete."""

    def test_delete_removes_entry(self) -> None:
        """Deleted entry is no longer readable."""
        knowledge_write("/failure_pattern/db/to_delete", _sample_entry())

        delete_result = knowledge_delete("/failure_pattern/db/to_delete")
        assert "Deleted" in delete_result

        read_result = knowledge_read("/failure_pattern/db/to_delete")
        assert "error" in read_result
        assert "Not found" in read_result["error"]

    def test_delete_nonexistent_does_not_raise(self) -> None:
        """Deleting a non-existent entry does not raise an error."""
        # InMemoryStore.delete silently ignores missing keys
        result = knowledge_delete("/failure_pattern/db/never_existed")
        assert "Deleted" in result


class TestListNamespace:
    """Verify knowledge_list browsing."""

    def test_list_shows_written_entries(self) -> None:
        """Written entries appear in list results."""
        knowledge_write("/failure_pattern/database/pool", _sample_entry(title="Pool"))
        knowledge_write("/failure_pattern/database/locks", _sample_entry(title="Locks"))

        result = knowledge_list("/failure_pattern/database/")
        assert result["path"] == "/failure_pattern/database/"

        entry_keys = [e["key"] for e in result["entries"]]
        assert "pool" in entry_keys
        assert "locks" in entry_keys

    def test_list_root_shows_categories(self) -> None:
        """Listing root shows top-level category namespaces."""
        knowledge_write("/failure_pattern/db/test", _sample_entry())
        knowledge_write("/diagnostic_skill/general/test", _sample_entry(title="Skill"))

        result = knowledge_list("/")
        sub_paths = result["sub_paths"]
        assert any("failure_pattern" in p for p in sub_paths)
        assert any("diagnostic_skill" in p for p in sub_paths)

    def test_list_empty_path(self) -> None:
        """Listing a path with no entries returns empty."""
        result = knowledge_list("/nonexistent/")
        assert result["entries"] == []


class TestSearchNamespace:
    """Verify knowledge_search with InMemoryStore.

    InMemoryStore does not have a real embedding model, so semantic relevance
    scoring is not tested. We verify structural search (entries returned from
    correct namespace) and that the result format is correct.
    """

    def test_search_returns_written_entries(self) -> None:
        """Search finds entries written under the queried namespace."""
        knowledge_write("/failure_pattern/database/pool", _sample_entry(title="Pool"))
        knowledge_write("/failure_pattern/database/locks", _sample_entry(title="Locks"))

        results = knowledge_search("pool", path="/failure_pattern/database/")
        keys = [r["key"] for r in results]
        assert "pool" in keys

    def test_search_result_structure(self) -> None:
        """Each search result has key, namespace, value, and score fields."""
        knowledge_write("/failure_pattern/db/item", _sample_entry())

        results = knowledge_search("connection", path="/failure_pattern/db/")
        assert len(results) == 1
        result = results[0]
        assert "key" in result
        assert "namespace" in result
        assert "value" in result
        assert "score" in result
        assert result["key"] == "item"
        assert result["value"]["title"] == "Connection pool exhaustion"

    def test_search_with_path_prefix_filters(self) -> None:
        """Search scoped to a path prefix does not return entries from other namespaces."""
        knowledge_write("/failure_pattern/db/pool", _sample_entry(title="DB Pool"))
        knowledge_write("/failure_pattern/network/timeout", _sample_entry(title="Network Timeout"))

        results = knowledge_search("", path="/failure_pattern/db/")
        keys = [r["key"] for r in results]
        assert "pool" in keys
        assert "timeout" not in keys


class TestStoreNotInitialized:
    """Verify proper error when store is not set."""

    def test_read_without_store_raises(self) -> None:
        """Operations without store raise RuntimeError."""
        set_store(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_read("/any/path")

    def test_write_without_store_raises(self) -> None:
        """Write without store raises RuntimeError."""
        set_store(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_write("/any/path", {"title": "test"})

    def test_delete_without_store_raises(self) -> None:
        """Delete without store raises RuntimeError."""
        set_store(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_delete("/any/path")

    def test_search_without_store_raises(self) -> None:
        """Search without store raises RuntimeError."""
        set_store(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_search("query")

    def test_list_without_store_raises(self) -> None:
        """List without store raises RuntimeError."""
        set_store(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            knowledge_list("/")
