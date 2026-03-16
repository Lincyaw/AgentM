# Task: Vault Tool Functions

**Date**: 2026-03-16
**Status**: PENDING
**Plan**: [plan](../plans/2026-03-16-memory-vault.md)
**Design**: [design](../designs/memory-vault.md)
**Assignee**: tdd
**Depends**: [vault-store](2026-03-16-vault-store.md), [vault-search](2026-03-16-vault-search.md), [vault-graph](2026-03-16-vault-graph.md)

## Objective

Implement `src/agentm/tools/vault/tools.py` — 10 tool functions as a factory that creates closures over a `MarkdownVault` instance, following the same pattern as `scenarios/rca/tools.py`.

## Inputs

- [Design doc § Toolset](../designs/memory-vault.md)
- `src/agentm/scenarios/rca/tools.py` — factory closure pattern reference

## Outputs

- `src/agentm/tools/vault/tools.py` (new)
- `tests/unit/test_vault_tools.py` (new)

## Implementation Details

### Factory function

```python
def create_vault_tools(vault: MarkdownVault) -> dict[str, Any]:
    """Create vault tool functions with closure over vault instance."""
```

### 10 tool functions

All return `str` (JSON-serialized results for LLM consumption):

1. **`vault_write`** — Single or batch write. Params: `path`, `frontmatter`, `body` OR `entries: list[dict]`
2. **`vault_read`** — Read note. Params: `path`
3. **`vault_edit`** — Incremental edit. Params: `path`, `operation`, `params`
4. **`vault_delete`** — Delete note. Params: `path`
5. **`vault_rename`** — Rename/move note. Params: `old_path`, `new_path`
6. **`vault_list`** — Browse directory. Params: `path`, `depth`, `type_filter`
7. **`vault_search`** — Search. Params: `query`, `filters`, `mode`, `limit`
8. **`vault_backlinks`** — Who references me. Params: `path`
9. **`vault_traverse`** — BFS subgraph. Params: `start`, `depth`, `direction`
10. **`vault_lint`** — Dead links + orphans. Params: `request` (dummy non-default param for API proxy bug)

### Key constraints

- Every tool function must have at least one non-default parameter (API proxy `required: null` bug)
- All functions return `str` (JSON)
- Error handling: catch exceptions, return JSON error messages (never raise to LLM)
- Docstrings are LLM-facing descriptions

## Acceptance Conditions

- [ ] Factory returns dict with all 10 tool names
- [ ] Each tool function has correct signature with non-default first parameter
- [ ] Each tool returns JSON string
- [ ] Error cases return JSON error (not exceptions)
- [ ] vault_write supports both single and batch mode
- [ ] vault_search passes mode and filters correctly
- [ ] All docstrings are clear LLM-facing descriptions

## Notes

- Follow `create_rca_tools` pattern: factory takes dependencies, returns dict of callables
- Tool functions are sync (not async) — same as existing knowledge tools
- Keep under 300 lines
