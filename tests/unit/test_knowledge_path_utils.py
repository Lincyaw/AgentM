"""Tests for Knowledge Store path utilities.

Ref: designs/system-design-overview.md § Knowledge Store — Namespace hierarchy
Ref: designs/orchestrator.md § Orchestrator Tools — knowledge_search, knowledge_read

Path utilities convert between filesystem-like paths (used in config/tools)
and LangGraph Store namespace tuples (used at runtime).
"""

from __future__ import annotations

from agentm.tools.knowledge import (
    namespace_to_path,
    path_to_namespace,
    path_to_namespace_and_key,
)


class TestPathToNamespace:
    """Ref: designs/system-design-overview.md § Knowledge Store — namespace hierarchy

    Bug: wrong namespace tuple → Store lookups silently miss, returning empty results.
    """

    def test_root_path(self):
        assert path_to_namespace("/") == ("knowledge",)

    def test_deep_path(self):
        assert path_to_namespace("/failure_pattern/database") == (
            "knowledge", "failure_pattern", "database",
        )

    def test_trailing_slash_ignored(self):
        """Bug: with/without trailing slash produce different namespaces → inconsistent lookups."""
        assert path_to_namespace("/a/b/") == path_to_namespace("/a/b")

    def test_empty_string_treated_as_root(self):
        """Bug: empty path causes crash instead of defaulting to root namespace."""
        assert path_to_namespace("") == ("knowledge",)


class TestPathToNamespaceAndKey:
    """Ref: designs/system-design-overview.md § Knowledge Store — knowledge_read(path)

    Bug: key split wrong → Store.get() receives wrong key, returns None.
    """

    def test_splits_last_segment_as_key(self):
        ns, key = path_to_namespace_and_key("/failure_pattern/database/connection_pool_exhaustion")
        assert ns == ("knowledge", "failure_pattern", "database")
        assert key == "connection_pool_exhaustion"

    def test_single_segment_is_key_under_root(self):
        """Bug: single-segment path has no parent dir → key extraction fails."""
        ns, key = path_to_namespace_and_key("/my_entry")
        assert ns == ("knowledge",)
        assert key == "my_entry"


class TestRoundTrip:
    """path → namespace → path must be idempotent (after normalization).

    Bug: lossy round-trip → paths stored in config don't match runtime lookups.
    """

    def test_root_round_trip(self):
        assert namespace_to_path(path_to_namespace("/")) == "/"

    def test_deep_path_round_trip(self):
        assert namespace_to_path(path_to_namespace("/a/b/c/")) == "/a/b/c/"

    def test_no_trailing_slash_normalizes_to_trailing_slash(self):
        """Bug: inconsistent trailing slash → same path stored two different ways."""
        result = namespace_to_path(path_to_namespace("/failure_pattern/database"))
        assert result == "/failure_pattern/database/"
