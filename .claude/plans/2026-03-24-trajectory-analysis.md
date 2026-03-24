# Plan: Trajectory Analysis — Replace memory_extraction with skill-driven system

**Date**: 2026-03-24
**Status**: DRAFT

## Requirements Restatement

Replace the hardcoded `memory_extraction` scenario with a general-purpose `trajectory_analysis` scenario. The new system uses Agent Skills (SKILL.md + references/) stored in the vault for progressive disclosure. The orchestrator discovers/activates skills at runtime. Workers are defined via markdown files. All former memory_extraction data types (`KnowledgeEntry`, etc.) are removed — structure is purely prompt-guided.

## Related Designs

- [Trajectory Analysis](../designs/trajectory-analysis.md) — Full design document

## Implementation Phases

### Phase 1: Scenario Code — Replace memory_extraction with trajectory_analysis

Create the new `scenarios/trajectory_analysis/` package with generic state, strategy, answer schema, and output schema. Update registry wiring. Remove old memory_extraction code.

**Can be parallelized**: Tasks 1a–1d are independent of each other.

- Task 1a: [State + Strategy](../tasks/2026-03-24-ta-state-strategy.md) — `TrajectoryAnalysisState`, `TrajectoryAnalysisStrategy`, `SkillCatalogEntry`
- Task 1b: [Schemas + Registry](../tasks/2026-03-24-ta-schemas-registry.md) — `AnalyzeAnswer`, `AnalysisReport`, `register()`, update `discover()`
- Task 1c: [Config files](../tasks/2026-03-24-ta-config.md) — scenario.yaml, orchestrator_system.j2, workers/trajectory-reader.md, output prompt
- Task 1d: [Vault skills](../tasks/2026-03-24-ta-vault-skills.md) — memory-extraction SKILL.md + references/ (content migration from old prompts)

### Phase 2: Integration — Wire strategy into builder, update references

Connect the new strategy to the builder, update CLI references, clean up old imports.

**Depends on**: Phase 1 complete.

- Task 2: [Integration + Cleanup](../tasks/2026-03-24-ta-integration.md) — Builder wiring, CLI updates, remove old `scenarios/memory_extraction/`, update `models/types.py` comment

### Phase 3: Tests — Update all affected tests

Update tests that reference memory_extraction types, state schemas, or strategy.

**Can be parallelized with Phase 2** since tests reference the new types directly.

- Task 3: [Test updates](../tasks/2026-03-24-ta-tests.md) — Update 6 test files, remove obsolete assertions, add trajectory_analysis coverage

## Execution Strategy

Phases 1a–1d are fully independent and should be dispatched to parallel agents. Phase 2 and Phase 3 can also run in parallel once Phase 1 is complete.

```
Phase 1 (parallel):
  1a: State + Strategy  ──┐
  1b: Schemas + Registry ─┤
  1c: Config files ────────┤──→ Phase 2: Integration  ──→ Done
  1d: Vault skills ────────┘    Phase 3: Tests (parallel)
```

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Vault skill discovery depends on vault being initialized | MEDIUM | Strategy gracefully handles empty catalog (returns empty `<available_skills>`) |
| `models/types.py` comment mentions memory-extraction task types | LOW | Update comment, no functional impact |
| 6 test files reference old types | LOW | Straightforward find-and-replace, all tests already identified |
| CLI `run.py` and `main.py` reference memory_extraction | MEDIUM | Check how deep the references go — may be just import or config loading |

## Dependencies

- Vault infrastructure (MarkdownVault) — already implemented and tested
- Existing node-based orchestrator — no changes needed
- Existing builder wiring — minimal changes (strategy registration update)
