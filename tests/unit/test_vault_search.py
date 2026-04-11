"""Focused regression tests for vault search behaviors."""

from __future__ import annotations

import math
import sqlite3
import struct

from agentm.tools.vault.schema import create_schema
from agentm.tools.vault.search import (
    SearchResult,
    _cosine_similarity,
    apply_filters,
    hybrid_search,
    keyword_search,
    semantic_search,
)


def _unit_vector(dim: int, index: int) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _populate(conn: sqlite3.Connection) -> None:
    rows = [
        ("skill/timeout", "skill", "Timeout", "pattern", "active"),
        ("concept/pool", "concept", "Pooling", "fact", "active"),
        ("failure/db-timeout", "failure-pattern", "DB Timeout", "fact", "superseded"),
    ]
    for path, typ, title, conf, status in rows:
        conn.execute(
            "INSERT INTO notes (path, type, title, confidence, status) VALUES (?, ?, ?, ?, ?)",
            (path, typ, title, conf, status),
        )

    fts_rows = [
        ("skill/timeout", "Timeout", "Diagnose timeout issues", "database timeout"),
        ("concept/pool", "Pooling", "Connection pool lifecycle", "database pool"),
        ("failure/db-timeout", "DB Timeout", "Timeout failures from pool exhaustion", "database timeout failure"),
    ]
    for path, title, body, tags in fts_rows:
        conn.execute(
            "INSERT INTO notes_fts (path, title, body, tags) VALUES (?, ?, ?, ?)",
            (path, title, body, tags),
        )

    for path, tag in [
        ("skill/timeout", "database"),
        ("skill/timeout", "timeout"),
        ("concept/pool", "database"),
        ("concept/pool", "pool"),
        ("failure/db-timeout", "database"),
        ("failure/db-timeout", "failure"),
    ]:
        conn.execute("INSERT INTO tags (path, tag) VALUES (?, ?)", (path, tag))

    conn.commit()


def _insert_embeddings(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS note_embeddings (path TEXT PRIMARY KEY, embedding BLOB)"
    )
    for path, vec in {
        "skill/timeout": _unit_vector(8, 0),
        "concept/pool": _unit_vector(8, 1),
        "failure/db-timeout": _unit_vector(8, 2),
    }.items():
        conn.execute(
            "INSERT OR REPLACE INTO note_embeddings (path, embedding) VALUES (?, ?)",
            (path, struct.pack("8f", *vec)),
        )
    conn.commit()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    _populate(conn)
    return conn


def test_search_result_is_frozen_dataclass() -> None:
    result = SearchResult(
        path="x",
        score=1.0,
        title="T",
        type="skill",
        confidence="fact",
        status="active",
        tags=["a"],
        snippet="snippet",
    )
    try:
        result.path = "y"  # type: ignore[misc]
        raised = False
    except AttributeError:
        raised = True
    assert raised


def test_apply_filters_builds_combined_clause_and_params() -> None:
    clause, params = apply_filters({"type": "skill", "status": "active", "tags": ["database"]})
    assert "n.type = ?" in clause
    assert "n.status = ?" in clause
    assert "SELECT path FROM tags" in clause
    assert params == ["skill", "active", "database"]


def test_keyword_search_handles_empty_query_and_filters() -> None:
    conn = _conn()

    assert keyword_search(conn, "", {}, 10) == []

    results = keyword_search(conn, "timeout", {"type": "skill"}, 10)
    assert len(results) == 1
    assert results[0].path == "skill/timeout"
    assert results[0].snippet


def test_semantic_search_uses_cosine_fallback_and_applies_filters() -> None:
    conn = _conn()
    _insert_embeddings(conn)

    def emb_fn(_: str) -> list[float]:
        return _unit_vector(8, 0)

    results = semantic_search(conn, "timeout", emb_fn, {"type": "skill"}, 10)
    assert [r.path for r in results] == ["skill/timeout"]
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_hybrid_search_merges_results_and_respects_limit() -> None:
    conn = _conn()
    _insert_embeddings(conn)

    results = hybrid_search(conn, "timeout", lambda _: _unit_vector(8, 0), {}, 1)
    assert len(results) == 1
    assert results[0].path in {"skill/timeout", "failure/db-timeout"}


def test_cosine_similarity_guards_zero_vector() -> None:
    assert math.isclose(_cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0, abs_tol=1e-9)
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
