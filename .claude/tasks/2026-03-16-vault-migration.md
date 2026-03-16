# Task: Vault Migration Module

**Date**: 2026-03-16
**Status**: PENDING
**Plan**: [plan](../plans/2026-03-16-memory-vault.md)
**Design**: [design](../designs/memory-vault.md)
**Assignee**: tdd
**Depends**: [vault-parser](2026-03-16-vault-parser.md), [vault-store](2026-03-16-vault-store.md)

## Objective

Implement `src/agentm/tools/vault/migration.py` — one-time conversion of existing JSON-based KnowledgeStore entries to Markdown vault format.

## Inputs

- [Design doc § Migration from KnowledgeStore](../designs/memory-vault.md)
- `src/agentm/tools/knowledge.py` — KnowledgeStore JSON format

## Outputs

- `src/agentm/tools/vault/migration.py` (new)
- `tests/unit/test_vault_migration.py` (new)

## Implementation Details

### Function to implement

```python
def migrate_knowledge_store(
    source_dir: str | Path,
    vault: MarkdownVault,
) -> MigrationReport:
    """Migrate KnowledgeStore JSON entries to vault Markdown format.

    Non-destructive: reads JSON files, writes .md files via vault.write_batch().
    Original JSON files are not modified or deleted.
    """
```

### Field mapping

| KnowledgeEntry field | Vault equivalent |
|---------------------|-----------------|
| `path` (`/category/slug`) | path (`category/slug`) — strip leading `/` |
| `category` | `type` frontmatter + folder |
| `confidence` | `confidence` frontmatter |
| `title` | H1 heading in body |
| `description` | Markdown body |
| `evidence` | `## Evidence` section |
| `tags` | `tags` frontmatter list |
| `related_entries` | `[[wikilink]]` in `## Related` section |
| `created_at` / `updated_at` | `created` / `updated` frontmatter |

### MigrationReport dataclass

```python
@dataclass(frozen=True)
class MigrationReport:
    total_entries: int
    migrated: int
    skipped: int  # already exists in vault
    errors: list[str]
```

## Acceptance Conditions

- [ ] Reads JSON files from KnowledgeStore directory structure
- [ ] Converts each entry to Markdown with correct frontmatter
- [ ] `related_entries` paths converted to `[[wikilink]]` syntax
- [ ] Uses vault.write_batch for atomic migration
- [ ] Non-destructive: original JSON files unchanged
- [ ] Handles malformed JSON gracefully (skip + record error)
- [ ] Returns accurate MigrationReport
- [ ] Empty source directory produces zero-migration report (not error)

## Notes

- KnowledgeStore uses `/<category>/<slug>` paths with leading slash; vault uses `category/slug` without
- `.emb` sidecar files can be ignored (embeddings recomputed)
- Handle legacy `knowledge.json` flat file format too (see `_migrate_legacy_json` in knowledge.py)
