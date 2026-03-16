# Task: Vault Graph Module

**Date**: 2026-03-16
**Status**: PENDING
**Plan**: [plan](../plans/2026-03-16-memory-vault.md)
**Design**: [design](../designs/memory-vault.md)
**Assignee**: tdd
**Depends**: [vault-schema](2026-03-16-vault-schema.md)

## Objective

Implement `src/agentm/tools/vault/graph.py` — link graph operations using recursive CTEs on the `links` table.

## Inputs

- [Design doc § Graph Operations, § Traverse Return Format](../designs/memory-vault.md)

## Outputs

- `src/agentm/tools/vault/graph.py` (new)
- `tests/unit/test_vault_graph.py` (new)

## Implementation Details

### Functions to implement

1. **`get_backlinks(conn, path) -> list[str]`**
   - `SELECT source FROM links WHERE target = ?`

2. **`traverse(conn, start, depth, direction) -> TraverseResult`**
   - Forward: recursive CTE following source -> target
   - Backward: recursive CTE following target -> source
   - Both: merge two CTEs
   - Return nodes with depth + edges with source/target
   - Join with `notes` table for titles

3. **`lint(conn) -> LintResult`**
   - Dead links: `links.target NOT IN notes.path`
   - Orphan notes: notes with no inbound or outbound links

### Data classes

```python
@dataclass(frozen=True)
class TraverseNode:
    path: str
    depth: int
    title: str

@dataclass(frozen=True)
class TraverseEdge:
    source: str
    target: str

@dataclass(frozen=True)
class TraverseResult:
    start: str
    nodes: list[TraverseNode]
    edges: list[TraverseEdge]

@dataclass(frozen=True)
class LintResult:
    dead_links: list[tuple[str, str]]  # (source, target)
    orphan_notes: list[str]
```

## Acceptance Conditions

- [ ] Backlinks returns all files that reference a given path
- [ ] Forward traversal follows outgoing links to specified depth
- [ ] Backward traversal follows incoming links
- [ ] Both-direction traversal merges forward + backward
- [ ] Traverse returns correct depth for each node
- [ ] Traverse returns edges with direction metadata
- [ ] Lint detects dead links (target file doesn't exist in notes)
- [ ] Lint detects orphan notes (no links in or out)
- [ ] Depth limit prevents infinite recursion on cycles

## Notes

- All functions take a `sqlite3.Connection` — no file I/O
- Keep under 200 lines
- Test with pre-populated in-memory SQLite databases
