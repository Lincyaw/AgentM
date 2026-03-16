"""Vault search — FTS5 keyword, cosine semantic, hybrid RRF, filter application."""

from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SearchResult:
    """A single search hit with rich metadata."""

    path: str
    score: float
    title: str
    type: str
    confidence: str
    status: str
    tags: list[str]
    snippet: str



def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tags_for_path(conn: sqlite3.Connection, path: str) -> list[str]:
    """Fetch all tags for a given note path."""
    rows = conn.execute("SELECT tag FROM tags WHERE path = ?", (path,)).fetchall()
    return [r[0] for r in rows]


def _enrich_result(
    conn: sqlite3.Connection,
    path: str,
    score: float,
    snippet: str,
) -> SearchResult | None:
    """Look up notes metadata and build a SearchResult, or None if not found."""
    row = conn.execute(
        "SELECT title, type, confidence, status FROM notes WHERE path = ?",
        (path,),
    ).fetchone()
    if row is None:
        return None
    title, typ, confidence, status = row
    tags = _tags_for_path(conn, path)
    return SearchResult(
        path=path,
        score=score,
        title=title or "",
        type=typ or "",
        confidence=confidence or "",
        status=status or "",
        tags=tags,
        snippet=snippet,
    )


def _body_snippet(conn: sqlite3.Connection, path: str, max_len: int = 200) -> str:
    """Extract first ~max_len chars of FTS body as a snippet."""
    row = conn.execute(
        "SELECT body FROM notes_fts WHERE path = ?", (path,)
    ).fetchone()
    if row is None or not row[0]:
        return ""
    body = row[0]
    if len(body) <= max_len:
        return body
    return body[:max_len] + "..."


_EQUALITY_KEYS = ("type", "confidence", "status")


def apply_filters(filters: dict) -> tuple[str, list]:
    """Build WHERE clause fragments from a filter dict.

    Returns (clause_string, params_list). The clause_string is empty when
    no recognised filters are provided; otherwise it starts with " AND ".
    """
    clauses: list[str] = []
    params: list = []

    for key in _EQUALITY_KEYS:
        if key in filters:
            clauses.append(f"n.{key} = ?")
            params.append(filters[key])

    if "tags" in filters:
        tag_list = filters["tags"]
        if isinstance(tag_list, list) and tag_list:
            for tag in tag_list:
                clauses.append(
                    "n.path IN (SELECT path FROM tags WHERE tag = ?)"
                )
                params.append(tag)

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    filters: dict,
    limit: int,
) -> list[SearchResult]:
    """FTS5 full-text search over notes_fts with optional filters."""
    if not query or not query.strip():
        return []

    filter_clause, filter_params = apply_filters(filters)

    sql = (
        "SELECT f.path, "
        "  snippet(notes_fts, 2, '<b>', '</b>', '...', 32) AS snip, "
        "  f.rank "
        "FROM notes_fts f "
        "JOIN notes n ON f.path = n.path "
        f"WHERE notes_fts MATCH ?{filter_clause} "
        "ORDER BY f.rank "
        "LIMIT ?"
    )
    params = [query, *filter_params, limit]

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    results: list[SearchResult] = []
    for path, snippet_text, rank in rows:
        # FTS5 rank is negative (more negative = better), normalise to positive
        score = -rank if rank else 0.0
        r = _enrich_result(conn, path, score, snippet_text or "")
        if r is not None:
            results.append(r)
    return results


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    embedding_fn: Callable[[str], list[float]],
    filters: dict,
    limit: int,
) -> list[SearchResult]:
    """Vector similarity search. Uses sqlite-vec if available, else cosine fallback."""
    if not query or not query.strip():
        return []

    query_vec = embedding_fn(query)

    # Try sqlite-vec first
    results = _vec_search(conn, query_vec, filters, limit)
    if results is not None:
        return results

    # Fallback: in-memory cosine over note_embeddings table
    return _cosine_fallback(conn, query_vec, filters, limit)


def _vec_search(
    conn: sqlite3.Connection,
    query_vec: list[float],
    filters: dict,
    limit: int,
) -> list[SearchResult] | None:
    """Attempt sqlite-vec KNN. Returns None if table doesn't exist."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='notes_vec'"
    ).fetchone()
    if row is None:
        return None

    try:
        blob = struct.pack(f"{len(query_vec)}f", *query_vec)
        rows = conn.execute(
            "SELECT path, distance FROM notes_vec "
            "WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (blob, limit * 3),  # over-fetch for filter
        ).fetchall()
    except sqlite3.OperationalError:
        return None

    filter_clause, filter_params = apply_filters(filters)
    results: list[SearchResult] = []
    for path, distance in rows:
        if filter_clause:
            check = conn.execute(
                f"SELECT 1 FROM notes n WHERE n.path = ?{filter_clause}",
                [path, *filter_params],
            ).fetchone()
            if check is None:
                continue
        score = max(0.0, 1.0 - distance)
        snippet = _body_snippet(conn, path)
        r = _enrich_result(conn, path, score, snippet)
        if r is not None:
            results.append(r)
        if len(results) >= limit:
            break
    return results


def _cosine_fallback(
    conn: sqlite3.Connection,
    query_vec: list[float],
    filters: dict,
    limit: int,
) -> list[SearchResult]:
    """In-memory cosine similarity over note_embeddings table."""
    # Check if note_embeddings table exists
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='note_embeddings'"
    ).fetchone()
    if row is None:
        return []

    dim = len(query_vec)
    filter_clause, filter_params = apply_filters(filters)

    sql = "SELECT e.path, e.embedding FROM note_embeddings e JOIN notes n ON e.path = n.path"
    params: list = []
    if filter_clause:
        sql += f" WHERE 1=1{filter_clause}"
        params = filter_params

    rows = conn.execute(sql, params).fetchall()

    scored: list[tuple[str, float]] = []
    for path, emb_blob in rows:
        stored = list(struct.unpack(f"{dim}f", emb_blob))
        sim = _cosine_similarity(query_vec, stored)
        scored.append((path, sim))

    scored.sort(key=lambda x: x[1], reverse=True)

    results: list[SearchResult] = []
    for path, sim in scored[:limit]:
        snippet = _body_snippet(conn, path)
        r = _enrich_result(conn, path, sim, snippet)
        if r is not None:
            results.append(r)
    return results


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    embedding_fn: Callable[[str], list[float]],
    filters: dict,
    limit: int,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Hybrid search: keyword + semantic merged via Reciprocal Rank Fusion."""
    if not query or not query.strip():
        return []

    kw_results = keyword_search(conn, query, filters, limit * 2)
    sem_results = semantic_search(conn, query, embedding_fn, filters, limit * 2)

    # RRF scoring
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(kw_results):
        rrf_scores[r.path] = rrf_scores.get(r.path, 0.0) + 1.0 / (rrf_k + rank)
        result_map[r.path] = r

    for rank, r in enumerate(sem_results):
        rrf_scores[r.path] = rrf_scores.get(r.path, 0.0) + 1.0 / (rrf_k + rank)
        if r.path not in result_map:
            result_map[r.path] = r

    # Sort by RRF score descending
    sorted_paths = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)

    results: list[SearchResult] = []
    for path in sorted_paths[:limit]:
        original = result_map[path]
        # Replace score with RRF score
        results.append(
            SearchResult(
                path=original.path,
                score=rrf_scores[path],
                title=original.title,
                type=original.type,
                confidence=original.confidence,
                status=original.status,
                tags=original.tags,
                snippet=original.snippet,
            )
        )
    return results
