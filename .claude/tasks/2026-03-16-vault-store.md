# Task: Vault Store Module (MarkdownVault)

**Date**: 2026-03-16
**Status**: PENDING
**Plan**: [plan](../plans/2026-03-16-memory-vault.md)
**Design**: [design](../designs/memory-vault.md)
**Assignee**: tdd
**Depends**: [vault-parser](2026-03-16-vault-parser.md), [vault-schema](2026-03-16-vault-schema.md)

## Objective

Implement `src/agentm/tools/vault/store.py` — the `MarkdownVault` class providing CRUD operations with atomic file writes and transactional index updates.

## Inputs

- [Design doc § Write Flow, § Storage Model, § Edit Operations](../designs/memory-vault.md)
- `src/agentm/tools/vault/parser.py` — frontmatter/body parsing
- `src/agentm/tools/vault/schema.py` — DB schema creation

## Outputs

- `src/agentm/tools/vault/store.py` (new)
- `tests/unit/test_vault_store.py` (new)

## Implementation Details

### MarkdownVault class

```python
class MarkdownVault:
    def __init__(self, vault_dir: str | Path, embedding_model: str | None = None) -> None: ...
    def write(self, path: str, frontmatter: dict, body: str) -> None: ...
    def write_batch(self, entries: list[dict]) -> None: ...
    def read(self, path: str) -> dict | None: ...
    def edit(self, path: str, operation: str, params: dict) -> None: ...
    def delete(self, path: str) -> None: ...
    def rename(self, old_path: str, new_path: str) -> None: ...
    def list_notes(self, path: str = "", depth: int = 1, type_filter: str | None = None) -> list[dict]: ...
    def rebuild_index(self) -> int: ...
```

### Key behaviors

1. **Atomic write**: Write to `.tmp` file, then `os.replace` to final path
2. **Transactional index**: All index updates in single SQLite transaction
3. **Threading**: `threading.Lock` around all write operations
4. **Edit dispatch**: Route `operation` to parser functions (replace_string, set_frontmatter, replace_section, append_section)
5. **Rename**: Move file + rewrite `[[old_path]]` -> `[[new_path]]` in all referencing files
6. **Write batch**: Multiple entries in single transaction, atomic
7. **Index update helper**: `_index_note(conn, path, frontmatter, body)` — shared by write, edit, rebuild

### Private methods

- `_index_note(conn, path, frontmatter, body)` — UPSERT notes, DELETE+INSERT links/tags/fts
- `_compute_body_hash(body: str) -> str` — SHA-256 for skip-reindex optimization
- `_get_conn() -> sqlite3.Connection` — lazy connection to `.vault.db`

## Acceptance Conditions

- [ ] Write creates .md file and updates all index tables
- [ ] Write is atomic (tmp file + os.replace)
- [ ] Read returns parsed frontmatter + body
- [ ] Read returns None for non-existent path
- [ ] Edit with all 4 operations works correctly
- [ ] Edit re-indexes after modification
- [ ] Delete removes file and all index entries
- [ ] Rename updates file, all backlinks in referencing files, and index
- [ ] Batch write is atomic (all succeed or all fail)
- [ ] List supports depth and type_filter
- [ ] rebuild_index scans all .md files and repopulates index
- [ ] Thread safety: Lock prevents concurrent write corruption

## Notes

- Path convention: `skill/timeout-diagnosis` (no leading `/`, no `.md` extension)
- Keep under 400 lines — delegate parsing to parser.py, schema to schema.py
- Embedding computation delegated to search module (store stores the hash)
