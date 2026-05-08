# Task 06: V0 Dismantle

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-two-phase-audit.md)
**Design**: [design](../designs/llmharness-two-phase-audit.md) §7.1, §11
**Assignee**: implementer
**Size**: S
**Dependencies**: 05

## Objective

Hard-cut the V0 audit layout: delete the V0 flat audit files, remove V0
re-exports, drop V0 smoke tests pinned to the V0 schema, and update
`project-index.yaml` to deprecate V0-specific REQs and add V1 REQs.

## Files

Delete:
- `scenarios/llmharness/src/llmharness/audit/prompt.py`
- `scenarios/llmharness/src/llmharness/audit/submit_tool.py`
- `scenarios/llmharness/src/llmharness/audit/output.py`
- `scenarios/llmharness/src/llmharness/audit/extensions.py`

Modify:
- `scenarios/llmharness/src/llmharness/audit/__init__.py` — drop `AUDIT_SYSTEM_PROMPT`, `RawAuditOutput`, `compose_extensions`, `SUBMIT_AUDIT_TOOL_NAME` re-exports. Add re-exports for the new `extractor` and `auditor` subpackages (or leave the new subpackages to be imported directly; pick one and document).
- `scenarios/llmharness/tests/test_smoke.py` — remove the V0-specific tests:
  - `test_compose_extensions_default_shape`
  - `test_compose_extensions_keeps_submit_tool_when_optional_dropped`
  - any test asserting on `SUBMIT_AUDIT_PARAMETERS`, `SUBMIT_AUDIT_TOOL_NAME`, `AUDIT_SYSTEM_PROMPT`, or `RawAuditOutput`.
  - Keep `test_package_surface` (it exercises the public `schema.py` contract that V1 preserves).
- `scenarios/llmharness/project-index.yaml`:
  - Mark V0 REQs about `submit_audit` / `compose_extensions` / single-pass shape as `superseded` with a pointer to the V1 design.
  - Add V1 REQs covering: extractor module, auditor module, two-phase orchestrator, typed failure entries, schema enum parity, integration test.

## Acceptance

- [ ] No file under `scenarios/llmharness/src/llmharness/audit/` named
      `prompt.py`, `submit_tool.py`, `output.py`, or `extensions.py`.
- [ ] `from llmharness.audit import compose_extensions` raises `ImportError`.
- [ ] `from llmharness.audit.extractor import compose_extractor_extensions` works.
- [ ] `from llmharness.audit.auditor import compose_auditor_extensions` works.
- [ ] `from llmharness import Event, Verdict, Reminder, EventKind, DriftType` still works (rca-autorl public contract).
- [ ] `uv run pytest --tb=short` green (V0 tests gone, V1 integration test from task 07 not yet added — should still pass with smoke + any other surviving tests).
- [ ] `validate_index.py` clean against `project-index.yaml`.

## Notes

- This is the design's mandated hard cut (§11) — do NOT add a parallel
  V0 path or shim.
- Downstream `rca-autorl` consumes only `schema.py` symbols, which V1
  preserves; no coordination required for this delete.
- If any scenario YAML in this repo references V0 modules (`llmharness.audit.submit_tool`), update or delete those references in this task too — grep for `llmharness.audit.submit_tool` and `llmharness.audit.compose_extensions` before declaring done.
