# Task: Vault Schema Module

**Date**: 2026-03-16
**Status**: PENDING
**Plan**: [plan](../plans/2026-03-16-memory-vault.md)
**Design**: [design](../designs/memory-vault.md)
**Assignee**: tdd

## Objective

Implement `src/agentm/tools/vault/schema.py` — SQLite schema creation for the vault index database (`.vault.db`).

## Inputs

- [Design doc § SQLite Schema](../designs/memory-vault.md)

## Outputs

- `src/agentm/tools/vault/schema.py` (new)
- `tests/unit/test_vault_schema.py` (new)

## Implementation Details

### Functions to implement

1. **`create_schema(conn: sqlite3.Connection) -> None`**
   - Create all tables if not exist: `notes`, `notes_fts`, `links`, `tags`
   - Create `notes_vec` only if sqlite-vec is available (graceful skip)
   - Create indexes: `idx_links_target`, `idx_tags_tag`
   - Use `IF NOT EXISTS` for idempotency

2. **`has_vec_support(conn: sqlite3.Connection) -> bool`**
   - Try loading sqlite-vec extension and probe
   - Return True/False, never raise

3. **`clear_all(conn: sqlite3.Connection) -> None`**
   - DELETE FROM all tables (for rebuild_index)
   - Do not DROP — preserve schema

## Acceptance Conditions

- [ ] Schema creation is idempotent (can run twice without error)
- [ ] FTS5 table uses `unicode61` tokenizer
- [ ] `notes_vec` creation gracefully skipped when sqlite-vec unavailable
- [ ] `has_vec_support` returns correct boolean
- [ ] `clear_all` empties all data tables
- [ ] All SQL uses parameterized queries where applicable

## Notes

- sqlite-vec loaded via `import sqlite_vec; conn.enable_load_extension(True); sqlite_vec.load(conn)` pattern
- Keep this module focused — no business logic, only DDL
- Test with in-memory SQLite (`:memory:`)
