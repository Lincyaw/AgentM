# Plan: Stub Implementation — Full AgentM Codebase

**Date**: 2026-03-08
**Status**: DRAFT
**Prereq**: [TDD Skeleton](2026-03-08-tdd-skeleton-and-tests.md) — COMPLETED
**Design refs**: [Orchestrator](../designs/orchestrator.md), [Sub-Agent](../designs/sub-agent.md), [Generic State Wrapper](../designs/generic-state-wrapper.md), [System Overview](../designs/system-design-overview.md)

## Requirements Restatement

All data models, enums, and Pydantic schemas are implemented. All business logic files contain stubs (`raise NotImplementedError`). This plan drives implementation of all stubs in dependency order, using existing skip-marked tests as acceptance criteria and adding Layer 2 snapshot tests.

## Dependency Graph

```
Phase 1: Pure Functions (no LangGraph, no I/O)
  ├─ 1A: notebook.py            (pure dataclass transforms)
  ├─ 1B: config/loader.py       (YAML + env vars)
  └─ 1C: core/prompt.py + tool_registry.py  (Jinja2, simple registry)
           │
Phase 2: Composed Utilities (depends on Phase 1)
  ├─ 2A: config/validator.py + phase_manager.from_config  (depends on 1B, 1C)
  └─ 2B: core/compression.py    (depends on 1A)
           │
Phase 3: Async Core (depends on Phase 2)
  └─ 3: core/task_manager.py    (depends on 2B)
           │
Phase 4: Agent Assembly (depends on Phase 3)
  ├─ 4A: agents/hooks.py + sub_agent.py  (depends on 3, 2B)
  └─ 4B: agents/orchestrator.py (depends on 1A, can parallel with 4A)
           │
Phase 5: Tool Layer (depends on Phase 3 + 4)
  ├─ 5: tools/orchestrator.py   (depends on 3, 1A)
  └─    tools/knowledge.py      (depends on Store)
           │
Phase 6: Builder (depends on Phase 4 + 5)
  └─ 6: builder.py              (depends on everything)
           │
Phase 7: Snapshot Tests (Layer 2 — depends on Phase 5)
  └─ 7: tests/snapshot/         (P1–P8 from testing-strategy.md)
```

## Parallelization Map

```
Wave 1 (3 parallel):   Wave 2 (2 parallel):     Wave 3:        Wave 4 (2 parallel):
  1A (notebook)           2A (validator)           3 (TaskMgr)    4A (hooks+subagent)
  1B (loader)             2B (compression)                        4B (orchestrator)
  1C (prompt+registry)
                                                 Wave 5:        Wave 6:          Wave 7:
                                                   5 (tools)      6 (builder)      7 (tests)
```

## Tasks

| ID | Task | Files | Size | Depends | Task File |
|----|------|-------|------|---------|-----------|
| 1A | Notebook Operations | `core/notebook.py` | M | — | [task](../tasks/2026-03-08-impl-notebook.md) |
| 1B | Config Loader | `config/loader.py` | S | — | [task](../tasks/2026-03-08-impl-config-loader.md) |
| 1C | Prompt + Tool Registry | `core/prompt.py`, `core/tool_registry.py` | S | — | [task](../tasks/2026-03-08-impl-prompt-tool-registry.md) |
| 2A | Config Validator + PhaseManager | `config/validator.py`, `core/phase_manager.py` | S | 1B, 1C | [task](../tasks/2026-03-08-impl-config-validator-phase-manager.md) |
| 2B | Compression | `core/compression.py` | M | 1A | [task](../tasks/2026-03-08-impl-compression.md) |
| 3 | TaskManager | `core/task_manager.py` | L | 2B | [task](../tasks/2026-03-08-impl-task-manager.md) |
| 4A | Hooks + Sub-Agent | `agents/hooks.py`, `agents/sub_agent.py` | M | 3, 2B | [task](../tasks/2026-03-08-impl-hooks-sub-agent.md) |
| 4B | Orchestrator Agent | `agents/orchestrator.py` | S | 1A | [task](../tasks/2026-03-08-impl-orchestrator-agent.md) |
| 5 | Orchestrator Tools + Knowledge | `tools/orchestrator.py`, `tools/knowledge.py` | M | 3, 1A | [task](../tasks/2026-03-08-impl-orchestrator-tools.md) |
| 6 | Builder | `builder.py` | S | 4A, 4B, 5 | [task](../tasks/2026-03-08-impl-builder.md) |
| 7 | Snapshot Tests (Layer 2) | `tests/snapshot/` | M | 5, 6 | [task](../tasks/2026-03-08-snapshot-tests.md) |

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| TaskManager async complexity | HIGH | Implement sync methods first, then async |
| tools/orchestrator.py `Command` objects | HIGH | Follow design doc exactly; verify LangGraph API |
| Compression LLM dependency | MEDIUM | Mock LLM; use tiktoken for counting |
| PhaseManager.from_config YAML format | LOW | Infer from design doc examples |

## Acceptance Criteria

- [ ] All existing tests pass (no regression)
- [ ] `test_notebook_immutability.py` — 5 classes unskipped, 10 tests pass
- [ ] `test_hypothesis_state_machine.py::TestValidateHypothesisTransition` — 15 parametrized tests pass
- [ ] `tests/snapshot/` — 8 new Layer 2 tests pass (P1–P8)
- [ ] Coverage >= 80% on `core/`, `config/`, `agents/`, `tools/`
- [ ] `uv run pytest` passes with no skip markers except Layer 3-4 eval tests
