# Task 01: Schema Enum Parity Helper

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-two-phase-audit.md)
**Design**: [design](../designs/llmharness-two-phase-audit.md) §7.2
**Assignee**: implementer
**Size**: S
**Dependencies**: none

## Objective

Factor the JSON-Schema enum lists for `EventKind` and `DriftType` out of
the V0 hand-listed copies into a single source-of-truth helper module so
both extractor and auditor (and any future consumer) derive their schema
from the enum classes.

## Files

Create:
- `/home/ddq/AoyangSpace/AgentM/scenarios/llmharness/src/llmharness/audit/_enum_schema.py`

Read for context:
- `/home/ddq/AoyangSpace/AgentM/scenarios/llmharness/src/llmharness/schema.py` (`EventKind`, `DriftType`)
- `/home/ddq/AoyangSpace/AgentM/scenarios/llmharness/src/llmharness/audit/submit_tool.py` (V0 hand-listed enums; pattern to replace)

## Contents

Module exposes (exact names):

```python
EVENT_KIND_VALUES: list[str]   # = [k.value for k in EventKind]
DRIFT_TYPE_VALUES: list[str | None]  # = [t.value for t in DriftType] + [None]
```

No other public symbols. No runtime mutation guards beyond `Final` typing if
desired.

## Acceptance

- [ ] Module imports cleanly: `from llmharness.audit._enum_schema import EVENT_KIND_VALUES, DRIFT_TYPE_VALUES`.
- [ ] `EVENT_KIND_VALUES == [k.value for k in EventKind]` (length and order).
- [ ] `DRIFT_TYPE_VALUES[-1] is None` and length is `len(DriftType) + 1`.
- [ ] `mypy --strict` clean for the new file.
- [ ] No other files modified (V0 still uses its own hand-listed enums until tasks 02/03 land).

## Notes

- Independent of phase split; can land first as a clean prep commit.
- Keep the module under `audit/` (not `schema.py` or top-level) — it is an
  audit-internal serialization helper, not a public schema contract.
- `Final` annotation optional but encouraged so static checkers catch
  accidental mutation.
