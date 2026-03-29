# Design: Memory Vault

**Status**: DRAFT
**Created**: 2026-03-16
**Last Updated**: 2026-03-16

---

## Overview

The Memory Vault is a Markdown + YAML frontmatter based persistent knowledge store with bidirectional wikilinks, replacing the JSON-based `KnowledgeStore` in `tools/knowledge.py`. It serves as a general-purpose toolset for scenarios that need persistent knowledge storage, allowing agents to read, write, search, and navigate interconnected Markdown files indexed by SQLite.

---

## Motivation

The existing `KnowledgeStore` uses flat JSON files with an in-memory inverted index and sidecar `.emb` files. Limitations:

1. **No relationship model** --- `related_entries` is a list of path strings with no backlink resolution or graph traversal.
2. **In-memory index** --- Inverted index and embeddings rebuilt on startup by scanning all files. Does not scale.
3. **No structured metadata** --- JSON blobs with no enforced schema. Fields vary across entries.
4. **No versioning** --- No `status` (active/superseded/archived), no `confidence` as first-class metadata.
5. **Poor discoverability** --- No full-text search (only token-based inverted index), no graph traversal, no dead-link detection.
6. **Full-rewrite only** --- Every write outputs the entire entry. No partial/incremental editing.

---

## Storage Model

### File Layout

```
<vault_dir>/
  .vault.db                              # SQLite index (rebuildable)
  skill/
    timeout-diagnosis.md
    log-pattern-matching.md
  episodic/
    2024-01-15-rca-db-outage.md
  concept/
    connection-pooling.md
  failure-pattern/
    db-connection-exhaustion.md
```

### File Format

```markdown
---
type: skill
tags: [database, timeout]
aliases: [超时诊断]
confidence: pattern
source_threads: [thread-abc]
created: 2024-01-15T14:30:00
updated: 2024-01-20T10:00:00
status: active
---

# Timeout Diagnosis

Description of the diagnostic skill...

## Evidence
- [[episodic/2024-01-15-rca-db-outage]]: First observed during DB outage

## Related
- [[concept/connection-pooling]]
```

### Frontmatter Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `str` | Yes | `skill`, `episodic`, `concept`, `failure-pattern`, `system-knowledge` |
| `tags` | `list[str]` | No | Free-form tags for filtering |
| `aliases` | `list[str]` | No | Alternative names (search matching) |
| `confidence` | `str` | Yes | `fact` / `pattern` / `heuristic` |
| `source_threads` | `list[str]` | No | Thread IDs that contributed |
| `created` | `str` (ISO 8601) | Auto | Creation timestamp |
| `updated` | `str` (ISO 8601) | Auto | Last modification timestamp |
| `status` | `str` | Auto | `active` / `superseded` / `archived` (default: `active`) |

Frontmatter is **extensible** --- unknown fields are preserved on read/write.

### Link Syntax

Wikilinks use `[[path]]` syntax, path relative to vault root without `.md` extension:

- `[[concept/connection-pooling]]` -> `<vault_dir>/concept/connection-pooling.md`
- Parsed from both body and frontmatter string values via regex: `\[\[([^\]]+)\]\]`

---

## Three-Layer Index

All persisted in `<vault_dir>/.vault.db`. The index is a **derived artifact** --- if deleted, fully rebuildable from `.md` files.

### Layer 1: Structural Index

Filesystem folders + filenames. Zero-cost browsing via `pathlib`. The `type` frontmatter field corresponds to the top-level folder by convention.

### Layer 2: Link Graph

Directed edges parsed from `[[wikilink]]` occurrences. Stored in `links` table. Enables backlink queries and graph traversal via recursive CTEs.

### Layer 3: Search Index

- **FTS5** --- Full-text search over title, body, tags. `unicode61` tokenizer for multilingual support.
- **sqlite-vec** --- Vector similarity using `jina-embeddings-v5-text-nano` (512-dim). Optional --- degrades to keyword-only if unavailable.

---

## SQLite Schema

```sql
CREATE TABLE notes (
    path        TEXT PRIMARY KEY,    -- 'skill/timeout-diagnosis'
    type        TEXT,
    title       TEXT,                -- h1 heading
    confidence  TEXT,
    status      TEXT DEFAULT 'active',
    created_at  TEXT,
    updated_at  TEXT,
    frontmatter TEXT,                -- full YAML serialized
    body_hash   TEXT                 -- SHA-256, skip reindex if unchanged
);

CREATE VIRTUAL TABLE notes_fts USING fts5(
    path, title, body, tags,
    tokenize='unicode61'
);

CREATE TABLE links (
    source  TEXT NOT NULL,
    target  TEXT NOT NULL,
    PRIMARY KEY (source, target)
);
CREATE INDEX idx_links_target ON links(target);

CREATE TABLE tags (
    path    TEXT NOT NULL,
    tag     TEXT NOT NULL,
    PRIMARY KEY (path, tag)
);
CREATE INDEX idx_tags_tag ON tags(tag);

CREATE VIRTUAL TABLE notes_vec USING vec0(
    path    TEXT PRIMARY KEY,
    embedding FLOAT[512]
);
```

---

## Toolset

### CRUD & Structural Tools (6)

| Tool | Parameters | Side Effects |
|------|-----------|--------------|
| `vault_write` | `entries: list[WriteEntry]` or single `path`, `frontmatter`, `body` | Create/overwrite `.md` -> update all indexes. Accepts single entry or array for batch writes (single transaction) |
| `vault_read` | `path` | Read only, return parsed frontmatter + body |
| `vault_edit` | `path`, `operation`, `params` (see Edit Operations) | Partial update -> reindex |
| `vault_delete` | `path` | Delete file + clean all indexes |
| `vault_rename` | `old_path`, `new_path` | Move file + rewrite `[[old]]` -> `[[new]]` in all referencing files |
| `vault_list` | `path`, `depth`, `type_filter` | Read only, browse directory |

`vault_write` batch mode:

```python
# Single entry
vault_write(path="skill/timeout-diagnosis", frontmatter={...}, body="...")

# Batch: multiple entries in one atomic transaction
vault_write(entries=[
    {"path": "skill/timeout-diagnosis", "frontmatter": {...}, "body": "..."},
    {"path": "episodic/2024-01-15-session-x", "frontmatter": {...}, "body": "..."},
    {"path": "concept/connection-pooling", "frontmatter": {...}, "body": "..."},
])
# All files written + all indexes updated in a single transaction.
# On failure, no partial state is committed.
```

### Search & Discovery Tools (4)

| Tool | Parameters | Side Effects |
|------|-----------|--------------|
| `vault_search` | `query`, `filters: dict`, `mode` (keyword/semantic/hybrid), `limit` | Read only. Returns rich results (see below) |
| `vault_backlinks` | `path` | Read only, who references me |
| `vault_traverse` | `start`, `depth`, `direction` (forward/backward/both) | Read only, BFS subgraph with direction metadata |
| `vault_lint` | --- | Read only, dead links + orphans |

### Edit Operations (`vault_edit`)

Instead of outputting full file content on every update, `vault_edit` provides four incremental edit primitives. The `operation` parameter selects which one:

#### `replace_string` --- Claude Code Edit-style exact replacement

```python
vault_edit(
    path="skill/timeout-diagnosis",
    operation="replace_string",
    params={
        "old": "P95 latency: 200ms",
        "new": "P95 latency: 450ms (updated 2024-01-20)"
    }
)
```

- Exact string match (whitespace-sensitive, multi-line supported)
- Replaces first occurrence only
- Fails if `old` not found or ambiguous (multiple matches)

#### `set_frontmatter` --- Merge-update YAML fields

```python
vault_edit(
    path="skill/timeout-diagnosis",
    operation="set_frontmatter",
    params={
        "confidence": "fact",
        "tags": ["database", "timeout", "verified"]
    }
)
```

- Merge semantics: specified fields updated, unspecified fields preserved
- Field order maintained where possible

#### `replace_section` --- Replace content under a heading

```python
vault_edit(
    path="skill/timeout-diagnosis",
    operation="replace_section",
    params={
        "heading": "## Evidence",
        "body": "- [[episodic/2024-01-15-rca-db-outage]]: Confirmed root cause\n- [[episodic/2024-01-20-rca-follow-up]]: Validated fix"
    }
)
```

- Heading must match exactly (including `##` level)
- Replaces everything between this heading and the next heading at same/higher level
- Preserves the heading line itself

#### `append_section` --- Append to end of a section

```python
vault_edit(
    path="skill/timeout-diagnosis",
    operation="append_section",
    params={
        "heading": "## Evidence",
        "content": "- [[episodic/2024-01-20-rca-follow-up]]: Validated fix effectiveness"
    }
)
```

- Appends before the next heading at same/higher level
- No duplication check (caller responsibility)

### Why These Four Primitives?

| Primitive | Token Cost | Safety | Best For |
|-----------|-----------|--------|----------|
| `replace_string` | Very low (small diff) | Medium (exact match) | Inline corrections, small edits |
| `set_frontmatter` | Very low (key-value) | Very high (isolated) | Metadata: status, confidence, tags |
| `replace_section` | Medium (section scope) | High (structure preserved) | Section rewrites |
| `append_section` | Low (append-only) | Very high (no deletion) | Growing evidence lists |

All four avoid the agent outputting the full file, which saves tokens and reduces error risk.

---

## Write Flow (Atomic)

Every write/edit/delete/rename operation follows:

```
1. Acquire threading.Lock
2. Parse/validate input
3. Write .md file to disk (atomic: write to .tmp, os.replace)
4. BEGIN TRANSACTION
   a. UPSERT notes (metadata + body_hash)
   b. DELETE + INSERT links (re-parse [[wikilink]])
   c. DELETE + INSERT tags
   d. DELETE + INSERT notes_fts
   e. UPSERT notes_vec (compute embedding if model available)
5. COMMIT
6. Release Lock
```

Markdown files are source of truth. On index corruption, `rebuild_index()` scans all `.md` files and reconstructs `.vault.db`.

---

## Search Implementation

### Keyword (FTS5)

```sql
SELECT path, title,
       snippet(notes_fts, 2, '<b>', '</b>', '...', 32) as snippet,
       rank
FROM notes_fts WHERE notes_fts MATCH ?
ORDER BY rank LIMIT ?;
```

### Semantic (sqlite-vec)

```sql
SELECT path, distance
FROM notes_vec
WHERE embedding MATCH ? AND k = ?
ORDER BY distance;
```

Fallback: If sqlite-vec unavailable, in-memory cosine (same as current `knowledge.py`).

### Hybrid (RRF)

Reciprocal Rank Fusion with `k=60`:

```python
for rank, (path, _) in enumerate(kw_results):
    rrf[path] += 1 / (60 + rank)
for rank, (path, _) in enumerate(sem_results):
    rrf[path] += 1 / (60 + rank)
```

### Filters

Applied as SQL WHERE clauses joined to results:

```python
filters = {"type": "skill", "confidence": "fact", "tags": ["database"]}
# -> WHERE type = ? AND confidence = ?
#    AND path IN (SELECT path FROM tags WHERE tag = ?)
```

### Search Return Format

`vault_search` returns rich results so the agent can make decisions (e.g. dedup, merge) without extra `vault_read` calls:

```python
[
    {
        "path": "failure-pattern/db-timeout",
        "score": 0.87,
        "title": "DB Connection Timeout Pattern",
        "type": "failure-pattern",
        "confidence": "fact",
        "status": "active",
        "tags": ["database", "timeout"],
        "snippet": "Occurs when connection pool exhaustion causes P99 latency spike..."  # ~200 chars from body
    },
    ...
]
```

Fields: `path`, `score`, `title`, `type`, `confidence`, `status`, `tags`, `snippet`. The `snippet` is extracted by FTS5 `snippet()` for keyword mode, or the first ~200 chars of body for semantic mode.

---

## Graph Operations

### Backlinks

```sql
SELECT source FROM links WHERE target = ?;
```

### Traverse (Recursive CTE)

Forward direction:

```sql
WITH RECURSIVE graph(path, depth) AS (
    SELECT ?, 0
    UNION ALL
    SELECT l.target, g.depth + 1
    FROM links l JOIN graph g ON l.source = g.path
    WHERE g.depth < ?
)
SELECT DISTINCT path, depth FROM graph;
```

Backward direction reverses `source`/`target`. Both direction runs two CTEs and merges.

### Traverse Return Format

Results include direction metadata so the agent understands link semantics:

```python
{
    "start": "concept/connection-pooling",
    "nodes": [
        {"path": "concept/connection-pooling", "depth": 0, "title": "Connection Pooling"},
        {"path": "failure-pattern/db-timeout", "depth": 1, "title": "DB Timeout Pattern"},
        {"path": "skill/pool-diagnostics", "depth": 1, "title": "Pool Diagnostics"},
        {"path": "concept/resource-exhaustion", "depth": 2, "title": "Resource Exhaustion"}
    ],
    "edges": [
        {"source": "failure-pattern/db-timeout", "target": "concept/connection-pooling"},
        {"source": "concept/connection-pooling", "target": "skill/pool-diagnostics"},
        {"source": "skill/pool-diagnostics", "target": "concept/resource-exhaustion"}
    ]
}
```

Each edge is a directed `source -> target` pair (matching the `[[wikilink]]` direction in the source file). The agent can distinguish "who references me" from "what I reference".

### Lint

```sql
-- Dead links
SELECT source, target FROM links
WHERE target NOT IN (SELECT path FROM notes);

-- Orphan notes
SELECT path FROM notes
WHERE path NOT IN (SELECT source FROM links)
  AND path NOT IN (SELECT target FROM links);
```

---

## Performance

| Operation | Implementation | Complexity |
|-----------|---------------|------------|
| Keyword search | FTS5 MATCH | O(1) amortized |
| Semantic search | sqlite-vec KNN | O(log n) |
| Hybrid search | FTS5 + vec -> RRF | Two queries |
| Backlinks | Index scan `idx_links_target` | O(1) |
| Forward links | PK scan `links` | O(1) |
| Graph traversal | Recursive CTE | In-database, bounded by depth |
| Property filter | Column index on `notes` | O(1) |
| Dead links | Set difference | Single query |
| Write (single note) | File + txn (5 statements) | O(1) + embedding |
| Full rebuild | Scan `.md` + bulk insert | O(n) |

---

## Session-to-Vault Relationship

```
Session (raw trajectory, not markdown)
    |  trajectory_analysis agent processes
    +-- episodic/2024-01-15-session-x.md   (session summary)
    +-- skill/new-skill-a.md               (extracted skill)
    +-- concept/some-concept.md            (update existing concept, via vault_edit)
```

One session -> multiple markdown outputs, cross-referenced via `[[wikilink]]`.

---

## File Organization

```
src/agentm/tools/vault/
  __init__.py          # Public API: MarkdownVault, tool functions
  store.py             # MarkdownVault class (core implementation)
  parser.py            # YAML frontmatter parsing, wikilink extraction, section parsing
  schema.py            # SQLite schema creation + migration
  search.py            # FTS5, sqlite-vec, hybrid RRF
  graph.py             # Link graph operations, traverse, lint
  tools.py             # 10 tool functions (vault_write, vault_read, etc.)
```

---

## Migration from KnowledgeStore

One-time migration converts JSON entries to Markdown:

| `KnowledgeEntry` field | Vault equivalent |
|------------------------|------------------|
| `id` | Derived from file path |
| `path` | File path relative to vault root |
| `category` | `type` frontmatter + folder |
| `confidence` | `confidence` frontmatter |
| `title` | H1 heading |
| `description` | Markdown body |
| `evidence` | `## Evidence` section with `[[wikilink]]` |
| `tags` | `tags` frontmatter |
| `related_entries` | `[[wikilink]]` references |
| `created_at` / `updated_at` | `created` / `updated` frontmatter |

---

## Integration Points

### scenarios/rca/ and scenarios/trajectory_analysis/

- `register()` registers vault tools via the scenario's `setup()` method
- Vault tools are available to both orchestrator and workers via `ScenarioWiring`
- Vault directory configured via `ScenarioConfig` or system config

### builder.py

- `build_agent_system()` creates `MarkdownVault` instance in phase 1 (platform resources)
- Injected into tool closures (same pattern as `ServiceProfileStore`)

### Configuration

```yaml
vault:
  dir: "./vault"
  embedding_model: "jinaai/jina-embeddings-v5-text-nano"
  sqlite_vec: true
```

---

## Constraints and Decisions

| Decision | Rationale |
|----------|-----------|
| Markdown source of truth, SQLite as index | Human-readable, git-trackable, rebuildable |
| SQLite for all indexes | Single file, zero-config, FTS5 + vec0 |
| `[[wikilink]]` syntax | Established convention, simple regex parsing |
| sqlite-vec with fallback | pip-installable, graceful degradation |
| Atomic file write + transaction | Index never references non-existent files |
| jina-embeddings-v5-text-nano | Already in use, 512-dim, local inference |
| Package layout `tools/vault/` | 7 modules < 400 lines each |
| No MOC notes | Folders + search + graph cover navigation needs |
| `vault_edit` with 4 primitives | Avoid full-file rewrite, save tokens |
| `vault_write` supports batch | Array input, single transaction. Avoids partial state in extract phase |
| No `vault_merge` tool | Merge is a semantic decision the agent must make explicitly via read + write + delete + edit. Auto-merge is unreliable |
| Rich `vault_search` results | Returns title, type, tags, snippet so agent can decide without extra reads |

---

## Open Questions

- [ ] Should `type` be constrained to enum or allow arbitrary strings?
- [ ] Embedding dimension parameterization if model changes?
- [ ] Should `vault_rename` also update `source_threads` in frontmatter?

---

## Related Concepts

- [Scenario Protocol](generic-state-wrapper.md) --- Scenario protocol, `build_agent_system()` vault wiring
- [Sub-Agent](sub-agent.md) --- Worker agents executing vault tool calls
- [System Design Overview](system-design-overview.md) --- Configuration, tool registration
