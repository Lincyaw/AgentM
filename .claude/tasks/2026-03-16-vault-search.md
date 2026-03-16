# Task: Vault Search Module

**Date**: 2026-03-16
**Status**: PENDING
**Plan**: [plan](../plans/2026-03-16-memory-vault.md)
**Design**: [design](../designs/memory-vault.md)
**Assignee**: tdd
**Depends**: [vault-schema](2026-03-16-vault-schema.md)

## Objective

Implement `src/agentm/tools/vault/search.py` — FTS5 keyword search, sqlite-vec semantic search, hybrid RRF, and filter application.

## Inputs

- [Design doc § Search Implementation, § Search Return Format, § Filters](../designs/memory-vault.md)
- `src/agentm/tools/knowledge.py` — existing embedding/cosine patterns for reference

## Outputs

- `src/agentm/tools/vault/search.py` (new)
- `tests/unit/test_vault_search.py` (new)

## Implementation Details

### Functions to implement

1. **`keyword_search(conn, query, filters, limit) -> list[SearchResult]`**
   - FTS5 MATCH query on `notes_fts`
   - `snippet()` function for context extraction
   - Apply filters as SQL WHERE clauses

2. **`semantic_search(conn, query, embedding_fn, filters, limit) -> list[SearchResult]`**
   - sqlite-vec KNN query on `notes_vec`
   - Fallback to in-memory cosine if vec unavailable
   - First ~200 chars of body as snippet

3. **`hybrid_search(conn, query, embedding_fn, filters, limit) -> list[SearchResult]`**
   - Run keyword + semantic
   - Merge via Reciprocal Rank Fusion (k=60)

4. **`apply_filters(base_query, filters) -> tuple[str, list]`**
   - Build WHERE clauses from filter dict
   - `type`, `confidence`, `status` as direct equality
   - `tags` as subquery against tags table

### SearchResult dataclass

```python
@dataclass(frozen=True)
class SearchResult:
    path: str
    score: float
    title: str
    type: str
    confidence: str
    status: str
    tags: list[str]
    snippet: str
```

## Acceptance Conditions

- [ ] FTS5 keyword search returns ranked results with snippets
- [ ] Semantic search works with sqlite-vec when available
- [ ] Semantic search falls back to in-memory cosine when vec unavailable
- [ ] Hybrid RRF merges keyword + semantic correctly
- [ ] Filters apply to all search modes
- [ ] Tag filter uses subquery (not string matching)
- [ ] SearchResult is frozen dataclass
- [ ] Empty query returns empty results (no crash)

## Notes

- `embedding_fn` passed as callable to decouple from model loading
- In-memory cosine fallback reuses existing `_cosine_similarity` pattern
- Keep under 300 lines
