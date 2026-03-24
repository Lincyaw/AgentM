# Task 1d: Vault Skills — Content Migration

**Plan**: [2026-03-24-trajectory-analysis](../plans/2026-03-24-trajectory-analysis.md)
**Design**: [trajectory-analysis](../designs/trajectory-analysis.md)
**Status**: TODO

## Scope

Migrate the existing orchestrator_system.j2 and worker extract.j2 content into the Agent Skills directory structure in the vault.

### Source → Target Mapping

| Source | Target |
|--------|--------|
| `orchestrator_system.j2` § purpose, strategy, discipline | `SKILL.md` (core + gotchas) |
| `orchestrator_system.j2` § what_to_extract | `references/what-to-extract.md` |
| `orchestrator_system.j2` § entry_quality + worker extract.j2 § what_makes_a_good_entry | `references/entry-quality.md` |
| `orchestrator_system.j2` § entry_categories | `references/categories.md` |
| `orchestrator_system.j2` § confidence_levels | `references/confidence-levels.md` |
| `orchestrator_system.j2` § entry_format | `references/entry-format.md` |
| `orchestrator_system.j2` § discipline table + worker extract.j2 § discipline | `SKILL.md` gotchas section |

### SKILL.md Requirements

Per Agent Skills spec:
- Frontmatter: `name: memory-extraction`, `description: ...` (max 1024 chars, imperative phrasing)
- Body < 500 lines, < 5000 tokens
- Contains: workflow, decision tree, gotchas
- Links to references/ via relative paths
- No explaining what the agent already knows (what trajectories are, what vault is)

### References Requirements

Each reference file:
- Standalone, focused on one topic
- Loaded on demand via `vault_read`
- SKILL.md tells agent WHEN to load each reference

## Files to Create

All under `knowledge/vault/skill/trajectory-analysis/memory-extraction/`:

- `SKILL.md`
- `references/what-to-extract.md`
- `references/entry-quality.md`
- `references/categories.md`
- `references/confidence-levels.md`
- `references/entry-format.md`

## Source Files to Read

- `config/scenarios/memory_extraction/prompts/orchestrator_system.j2`
- `config/scenarios/memory_extraction/prompts/task_types/extract.j2`

## Dependencies

- Vault directory must exist (`knowledge/vault/`)
