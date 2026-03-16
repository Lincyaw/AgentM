# Plan: Memory Vault — Markdown + SQLite Knowledge Store

**Date**: 2026-03-16
**Status**: DRAFT
**Target design**: [memory-vault](../designs/memory-vault.md)

## Requirements Restatement

Replace the JSON-based `KnowledgeStore` (`tools/knowledge.py`) with a Markdown + YAML frontmatter vault backed by a three-layer SQLite index (structural, link graph, FTS5 + sqlite-vec). The vault provides 10 tools (6 CRUD + 4 search/discovery), supports `[[wikilink]]` bidirectional linking, atomic writes, incremental edits via 4 primitives, and a rebuildable index. It integrates with `scenarios/memory_extraction/` as the tool backend and with `builder.py` for dependency injection.

## Prerequisites

- PyYAML already in dependencies (pyyaml>=6.0)
- Need to add `sqlite-vec` dependency via `uv add sqlite-vec`
- `sentence-transformers` already available for embedding computation
- Existing `KnowledgeStore` in `tools/knowledge.py` remains functional until migration is verified

## Implementation Phases

### Phase 1: Parser Module (no dependencies)
- YAML frontmatter parsing, wikilink extraction, section parsing
- Task: [task](../tasks/2026-03-16-vault-parser.md)
- Size: M

### Phase 2: Schema Module (no dependencies)
- SQLite schema creation (notes, notes_fts, links, tags, notes_vec)
- Task: [task](../tasks/2026-03-16-vault-schema.md)
- Size: S

### Phase 3: Store Module (core CRUD)
- MarkdownVault class: init, write, read, edit, delete, rename, list, rebuild_index
- Depends on: Phase 1 (parser), Phase 2 (schema)
- Task: [task](../tasks/2026-03-16-vault-store.md)
- Size: M

### Phase 4: Search Module
- FTS5 keyword search, sqlite-vec semantic search, hybrid RRF, filters
- Depends on: Phase 2 (schema — query patterns)
- Task: [task](../tasks/2026-03-16-vault-search.md)
- Size: M

### Phase 5: Graph Module
- Backlinks, recursive CTE traversal (forward/backward/both), lint (dead links + orphans)
- Depends on: Phase 2 (schema — links table)
- Task: [task](../tasks/2026-03-16-vault-graph.md)
- Size: S

### Phase 6: Tool Functions
- 10 tool functions as closures over MarkdownVault instance
- Depends on: Phase 3 (store), Phase 4 (search), Phase 5 (graph)
- Task: [task](../tasks/2026-03-16-vault-tools.md)
- Size: M

### Phase 7: Migration Module
- KnowledgeStore JSON -> Markdown conversion utility
- Depends on: Phase 1 (parser), Phase 3 (store)
- Task: [task](../tasks/2026-03-16-vault-migration.md)
- Size: S

### Phase 8: Integration
- Wire into builder.py, update scenarios/memory_extraction/ registration, add sqlite-vec dependency, public API in __init__.py
- Depends on: Phase 6 (tools), Phase 7 (migration)
- Task: [task](../tasks/2026-03-16-vault-integration.md)
- Size: M

## Dependency Graph

```
Phase 1 (parser) ──┐
                    ├──> Phase 3 (store) ──┐
Phase 2 (schema) ──┤                       ├──> Phase 6 (tools) ──┐
                    ├──> Phase 4 (search) ──┘                      ├──> Phase 8 (integration)
                    └──> Phase 5 (graph) ───┘                      │
Phase 1 + Phase 3 ──────────────────── Phase 7 (migration) ────────┘
```

Phases 1 and 2 are independent (can run in parallel).
Phases 4 and 5 are independent of each other (can run in parallel after Phase 2).

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| sqlite-vec import fails at runtime | MEDIUM | Graceful fallback to in-memory cosine similarity. Test both paths |
| FTS5 unicode61 tokenizer not available | LOW | Standard in Python's bundled sqlite3. Verify in CI |
| Atomic file write race condition | MEDIUM | threading.Lock in MarkdownVault, tmp-file + os.replace pattern |
| YAML frontmatter parsing edge cases | MEDIUM | Use PyYAML safe_load, comprehensive test coverage |
| Section parsing ambiguity (heading levels, empty sections) | MEDIUM | Strict heading-level matching, thorough tests |
| Migration data loss | HIGH | Non-destructive (reads JSON, writes .md), old files preserved |
| Builder wiring breaks existing scenario | MEDIUM | Vault replaces KnowledgeStore only for memory_extraction |

## Acceptance Criteria

- [ ] All parser tests pass (frontmatter round-trip, wikilink extraction, section parsing)
- [ ] All schema tests pass (table creation, migration idempotency)
- [ ] All store tests pass (CRUD, atomic write, rebuild_index, edit operations)
- [ ] All search tests pass (FTS5, semantic fallback, hybrid RRF, filters)
- [ ] All graph tests pass (backlinks, traverse, lint)
- [ ] All tool function tests pass (10 tools with correct signatures)
- [ ] Migration converts sample KnowledgeStore data without data loss
- [ ] Integration: builder creates MarkdownVault and injects into tool closures
- [ ] Coverage >= 80% for tools/vault/ package
- [ ] sqlite-vec graceful degradation verified
