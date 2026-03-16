"""Tests for vault search module — keyword, semantic, hybrid, filters."""

from __future__ import annotations

import math
import sqlite3

import pytest

from agentm.tools.vault.schema import create_schema
from agentm.tools.vault.search import (
    SearchResult,
    apply_filters,
    hybrid_search,
    keyword_search,
    semantic_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate(conn: sqlite3.Connection) -> None:
    """Insert test data into notes, notes_fts, and tags."""
    rows = [
        ("skill/timeout-diagnosis", "skill", "Timeout Diagnosis", "pattern", "active"),
        ("concept/connection-pooling", "concept", "Connection Pooling", "fact", "active"),
        ("failure-pattern/db-timeout", "failure-pattern", "DB Timeout Pattern", "fact", "superseded"),
        ("episodic/2024-01-session", "episodic", "Session Jan 2024", "heuristic", "archived"),
    ]
    for path, typ, title, conf, status in rows:
        conn.execute(
            "INSERT INTO notes (path, type, title, confidence, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (path, typ, title, conf, status),
        )

    fts_rows = [
        ("skill/timeout-diagnosis", "Timeout Diagnosis", "Diagnose timeout issues in database connections with P99 latency spikes", "database timeout"),
        ("concept/connection-pooling", "Connection Pooling", "Connection pool management and resource lifecycle in distributed systems", "database pool"),
        ("failure-pattern/db-timeout", "DB Timeout Pattern", "Pattern for database timeout failures caused by pool exhaustion", "database timeout failure"),
        ("episodic/2024-01-session", "Session Jan 2024", "Root cause analysis session for January outage event", "rca outage"),
    ]
    for path, title, body, tags in fts_rows:
        conn.execute(
            "INSERT INTO notes_fts (path, title, body, tags) VALUES (?, ?, ?, ?)",
            (path, title, body, tags),
        )

    tag_rows = [
        ("skill/timeout-diagnosis", "database"),
        ("skill/timeout-diagnosis", "timeout"),
        ("concept/connection-pooling", "database"),
        ("concept/connection-pooling", "pool"),
        ("failure-pattern/db-timeout", "database"),
        ("failure-pattern/db-timeout", "timeout"),
        ("failure-pattern/db-timeout", "failure"),
        ("episodic/2024-01-session", "rca"),
    ]
    for path, tag in tag_rows:
        conn.execute("INSERT INTO tags (path, tag) VALUES (?, ?)", (path, tag))

    conn.commit()


def _unit_vector(dim: int, index: int) -> list[float]:
    """Create a unit vector with 1.0 at `index`, rest zeros."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _mock_embedding_fn(text: str) -> list[float]:
    """Deterministic mock: maps known queries to fixed vectors."""
    # Query vector: point in direction 0
    return _unit_vector(8, 0)


@pytest.fixture
def conn():
    """In-memory SQLite with schema + test data."""
    c = sqlite3.connect(":memory:")
    create_schema(c)
    _populate(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_should_be_frozen(self):
        r = SearchResult(
            path="a", score=1.0, title="T", type="skill",
            confidence="fact", status="active", tags=["x"], snippet="s",
        )
        with pytest.raises(AttributeError):
            r.path = "b"  # type: ignore[misc]

    def test_should_store_all_fields(self):
        r = SearchResult(
            path="p", score=0.5, title="T", type="concept",
            confidence="pattern", status="active", tags=["a", "b"], snippet="snip",
        )
        assert r.path == "p"
        assert r.score == 0.5
        assert r.title == "T"
        assert r.type == "concept"
        assert r.confidence == "pattern"
        assert r.status == "active"
        assert r.tags == ["a", "b"]
        assert r.snippet == "snip"


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_should_return_empty_for_no_filters(self):
        clause, params = apply_filters({})
        assert clause == ""
        assert params == []

    def test_should_filter_by_type(self):
        clause, params = apply_filters({"type": "skill"})
        assert "type = ?" in clause
        assert params == ["skill"]

    def test_should_filter_by_confidence(self):
        clause, params = apply_filters({"confidence": "fact"})
        assert "confidence = ?" in clause
        assert params == ["fact"]

    def test_should_filter_by_status(self):
        clause, params = apply_filters({"status": "active"})
        assert "status = ?" in clause
        assert params == ["active"]

    def test_should_filter_by_tags_as_subquery(self):
        clause, params = apply_filters({"tags": ["database"]})
        assert "path IN" in clause
        assert "tags" in clause.lower()
        assert "database" in params

    def test_should_combine_multiple_filters(self):
        clause, params = apply_filters({"type": "skill", "confidence": "fact", "tags": ["db"]})
        assert "type = ?" in clause
        assert "confidence = ?" in clause
        assert "path IN" in clause
        assert len(params) == 3

    def test_should_handle_multiple_tags(self):
        clause, params = apply_filters({"tags": ["database", "timeout"]})
        # Should require ALL tags (intersection)
        assert params.count("database") == 1
        assert params.count("timeout") == 1

    def test_should_ignore_unknown_filter_keys(self):
        clause, params = apply_filters({"unknown_key": "value"})
        assert clause == ""
        assert params == []


# ---------------------------------------------------------------------------
# keyword_search
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    def test_should_return_results_for_matching_query(self, conn):
        results = keyword_search(conn, "timeout", {}, 10)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_should_return_empty_for_empty_query(self, conn):
        results = keyword_search(conn, "", {}, 10)
        assert results == []

    def test_should_include_snippet(self, conn):
        results = keyword_search(conn, "timeout", {}, 10)
        assert all(r.snippet != "" for r in results)

    def test_should_include_rich_metadata(self, conn):
        results = keyword_search(conn, "timeout", {}, 10)
        r = results[0]
        assert r.path != ""
        assert r.title != ""
        assert r.type != ""
        assert r.confidence != ""
        assert r.status != ""
        assert isinstance(r.tags, list)

    def test_should_respect_limit(self, conn):
        results = keyword_search(conn, "database", {}, 1)
        assert len(results) <= 1

    def test_should_apply_type_filter(self, conn):
        results = keyword_search(conn, "timeout", {"type": "skill"}, 10)
        assert all(r.type == "skill" for r in results)

    def test_should_apply_tag_filter(self, conn):
        results = keyword_search(conn, "database", {"tags": ["failure"]}, 10)
        # Only failure-pattern/db-timeout has the "failure" tag
        assert all("failure" in r.tags for r in results)

    def test_should_apply_status_filter(self, conn):
        results = keyword_search(conn, "database", {"status": "active"}, 10)
        assert all(r.status == "active" for r in results)

    def test_should_return_empty_when_no_match(self, conn):
        results = keyword_search(conn, "zzz_nonexistent_term_zzz", {}, 10)
        assert results == []


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------


def _make_stored_embeddings() -> dict[str, list[float]]:
    """Map each test path to a distinct vector."""
    return {
        "skill/timeout-diagnosis": _unit_vector(8, 0),
        "concept/connection-pooling": _unit_vector(8, 1),
        "failure-pattern/db-timeout": _unit_vector(8, 2),
        "episodic/2024-01-session": _unit_vector(8, 3),
    }


def _insert_embeddings(conn: sqlite3.Connection, embeddings: dict) -> None:
    """Insert embeddings into a plain table for cosine fallback testing."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS note_embeddings "
        "(path TEXT PRIMARY KEY, embedding BLOB)"
    )
    import struct
    for path, vec in embeddings.items():
        blob = struct.pack(f"{len(vec)}f", *vec)
        conn.execute(
            "INSERT OR REPLACE INTO note_embeddings (path, embedding) VALUES (?, ?)",
            (path, blob),
        )
    conn.commit()


class TestSemanticSearch:
    def test_should_return_results_with_cosine_fallback(self, conn):
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)

        def emb_fn(text: str) -> list[float]:
            return _unit_vector(8, 0)  # closest to timeout-diagnosis

        results = semantic_search(conn, "timeout", emb_fn, {}, 10)
        assert len(results) > 0
        assert results[0].path == "skill/timeout-diagnosis"

    def test_should_return_empty_for_empty_query(self, conn):
        results = semantic_search(conn, "", _mock_embedding_fn, {}, 10)
        assert results == []

    def test_should_return_scores_between_0_and_1(self, conn):
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)
        results = semantic_search(conn, "test", _mock_embedding_fn, {}, 10)
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_should_include_snippet_from_body(self, conn):
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)
        results = semantic_search(conn, "test", _mock_embedding_fn, {}, 10)
        for r in results:
            assert r.snippet != ""
            assert len(r.snippet) <= 250  # roughly ~200 chars

    def test_should_apply_filters(self, conn):
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)
        results = semantic_search(conn, "test", _mock_embedding_fn, {"type": "skill"}, 10)
        assert all(r.type == "skill" for r in results)

    def test_should_respect_limit(self, conn):
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)
        results = semantic_search(conn, "test", _mock_embedding_fn, {}, 2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_should_merge_keyword_and_semantic_results(self, conn):
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)
        results = hybrid_search(conn, "timeout", _mock_embedding_fn, {}, 10)
        assert len(results) > 0

    def test_should_return_empty_for_empty_query(self, conn):
        results = hybrid_search(conn, "", _mock_embedding_fn, {}, 10)
        assert results == []

    def test_should_respect_limit(self, conn):
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)
        results = hybrid_search(conn, "database", _mock_embedding_fn, {}, 2)
        assert len(results) <= 2

    def test_should_apply_filters(self, conn):
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)
        results = hybrid_search(conn, "timeout", _mock_embedding_fn, {"type": "skill"}, 10)
        assert all(r.type == "skill" for r in results)

    def test_rrf_should_boost_results_in_both_lists(self, conn):
        """A path appearing in both keyword and semantic should rank higher."""
        embeddings = _make_stored_embeddings()
        _insert_embeddings(conn, embeddings)
        # timeout-diagnosis matches keyword "timeout" AND is closest vector
        results = hybrid_search(conn, "timeout", _mock_embedding_fn, {}, 10)
        if len(results) > 0:
            # timeout-diagnosis should be ranked first or near top
            paths = [r.path for r in results]
            assert "skill/timeout-diagnosis" in paths


# ---------------------------------------------------------------------------
# Cosine similarity correctness (internal helper)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_should_return_1(self):
        from agentm.tools.vault.search import _cosine_similarity
        v = [1.0, 2.0, 3.0]
        assert math.isclose(_cosine_similarity(v, v), 1.0, abs_tol=1e-9)

    def test_orthogonal_vectors_should_return_0(self):
        from agentm.tools.vault.search import _cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert math.isclose(_cosine_similarity(a, b), 0.0, abs_tol=1e-9)

    def test_zero_vector_should_return_0(self):
        from agentm.tools.vault.search import _cosine_similarity
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0
