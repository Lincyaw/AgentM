# Plan: llmharness Two-Phase Cognitive Audit (V1)

**Date**: 2026-05-08
**Status**: DRAFT
**Target design**: [llmharness-two-phase-audit](../designs/llmharness-two-phase-audit.md)
**Predecessor plan**: [V0 plan](2026-05-08-llmharness-cognitive-audit-v0.md)

## Requirements Restatement

V1 splits the V0 single-pass audit into two stateless child sessions with
disjoint inputs and explicit, typed failure semantics:

- **Phase 1 (Extractor)** runs on every `TurnEndEvent`, sees the new-turn
  slice (with thinking blocks) plus a recent graph slice, terminates via
  `submit_events`, and appends `llmharness.audit_event` entries plus an
  `llmharness.extractor_cursor` pointer.
- **Phase 2 (Auditor)** runs every `k` turns (default `k=3`), sees the
  event graph and recent verdicts (NOT the raw trajectory), terminates via
  `submit_verdict`, and appends `llmharness.verdict` entries.

Both phases record typed failure entries (`extractor_no_call`,
`extractor_error`, `extractor_empty`, `audit_no_call`, `audit_error`)
instead of V0's silent `Verdict(drift=False)` fallback.

V1 is a **hard cut**: V0 `audit/{prompt,submit_tool,output,extensions}.py`
are deleted; the V0 smoke tests pinned to V0's schema go away; a new
stub-provider integration test becomes the V1 fail-stop.

Public contract preserved: `schema.py`, `cards.py`, `cards_tools` atom,
`inherit_provider` builtin, and `BeforeAgentStartEvent.system` injection
all unchanged. rca-autorl break-free.

## Prerequisites

- V0 shipped (cognitive_audit plan complete) — confirmed in repo state.
- No new AgentM SDK primitives required; `spawn_child_session`,
  `inherit_provider`, and entry-tree append are V0-stable.

## Implementation Phases (Tasks)

| ID | Task | Size | Depends on |
|----|------|------|-----------|
| 01 | Schema enum parity helper | S | — |
| 02 | Extractor module (`audit/extractor/`) | M | 01 |
| 03 | Auditor module (`audit/auditor/`) | M | 01 |
| 04 | Adapter rewrite (orchestrator) | M | 02, 03 |
| 05 | Failure entry types + `_record_failure` | S | 04 |
| 06 | V0 dismantle (delete V0 files + V0 tests) | S | 05 |
| 07 | V1 stub-provider integration test (fail-stop) | M | 06 |

Tasks 02 and 03 are independent and can be parallelized after 01.

## Dependency Graph

```
01 ──┬── 02 ──┐
     └── 03 ──┴── 04 ── 05 ── 06 ── 07
```

## Files Touched (summary)

Created:
- `scenarios/llmharness/src/llmharness/audit/_enum_schema.py`
- `scenarios/llmharness/src/llmharness/audit/extractor/{__init__,prompt,submit_tool,extensions,output}.py`
- `scenarios/llmharness/src/llmharness/audit/auditor/{__init__,prompt,submit_tool,extensions,output}.py`
- `scenarios/llmharness/tests/test_two_phase_audit_integration.py`

Modified:
- `scenarios/llmharness/src/llmharness/audit/__init__.py` (re-export shape changes)
- `scenarios/llmharness/src/llmharness/adapters/agentm.py` (full rewrite)
- `scenarios/llmharness/project-index.yaml` (deprecate V0-specific REQs, add V1 REQs)

Deleted:
- `scenarios/llmharness/src/llmharness/audit/prompt.py`
- `scenarios/llmharness/src/llmharness/audit/submit_tool.py`
- `scenarios/llmharness/src/llmharness/audit/output.py`
- `scenarios/llmharness/src/llmharness/audit/extensions.py`
- The V0 portions of `scenarios/llmharness/tests/test_smoke.py` (file remains, V0-specific tests removed)

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| `extractor_cursor` index drift across reloads | MEDIUM | Use `last_turn_index` against the live branch each firing; never cache outside entries |
| Thinking-block serialization variance across providers | MEDIUM | Slice from messages already on the entry tree; SDK has normalized them |
| if/then JSON schema rejected by some providers | MEDIUM | Keep adapter-side defensive validation as a second layer; failure → `audit_no_call` |
| Public contract regression (rca-autorl) | HIGH | `schema.py` untouched; CI smoke import test retained |
| 4-turn integration test flaky on stub provider | LOW | Use deterministic stub responses; assert on entry tree, not on free text |

## Acceptance Criteria

- [ ] All tasks 01–07 complete and merged in order.
- [ ] `uv run pytest --tb=short` green inside `scenarios/llmharness/`.
- [ ] `uv run mypy src/llmharness` clean (strict).
- [ ] `uv run ruff check src tests` clean.
- [ ] `python .../validate_index.py project-index.yaml` clean.
- [ ] V1 integration test asserts: Phase 1 fires 4×, Phase 2 fires 1× at turn 3, entry tree has expected mix.
- [ ] `from llmharness import Event, Verdict, Reminder, EventKind, DriftType` still works (rca-autorl contract).
- [ ] No file under `audit/` is named `prompt.py`/`submit_tool.py`/`output.py`/`extensions.py` (V0 layout fully gone).
- [ ] Reviewer signs off against the design doc §1–§11.
