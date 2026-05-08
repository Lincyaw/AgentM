# Task: Schema additions for cognitive audit V0

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-cognitive-audit-v0.md)
**Design**: [design](../designs/llmharness-cognitive-audit.md)
**Assignee**: implementer

## Objective

Add two opt-in fields to `Verdict` in `scenarios/llmharness/src/llmharness/schema.py`:

- `cited_cards: list[str] = field(default_factory=list)` — AFC IDs the
  audit chose to cite (drives V1 training data).
- `downstream_reaction: str | None = None` — free-text note populated
  by the next audit firing.

Both must be additive and non-breaking. No edge_types field, no enum
expansion. Per design §6 the existing `Event.refs` stays plain
`list[int]`.

## Inputs

- Read: `scenarios/llmharness/src/llmharness/schema.py` (current
  `Verdict` shape).
- Read: `scenarios/llmharness/CLAUDE.md` "Schema stability" rule —
  schema is rca-autorl public contract.
- Read: design §6.2 (V0 additions) and §6.3 (schema stability).
- Inspect existing `Verdict` callsites in
  `scenarios/llmharness/src/llmharness/detector.py` and
  `scenarios/llmharness/src/llmharness/agentm_bridge.py` to confirm
  no positional construction breaks.

## Outputs

- Modified: `scenarios/llmharness/src/llmharness/schema.py`
- Modified: `scenarios/llmharness/project-index.yaml` — add REQ-015
  (`title: "Verdict cited_cards / downstream_reaction (cognitive audit)"`,
  `priority: P0`, `status: implemented`, `code: [src/llmharness/schema.py]`,
  `depends_on: [REQ-001]`). No `tests:` field this round — testing pass
  is deferred.

## Acceptance Conditions

- [ ] Both fields present with correct types and defaults
- [ ] All existing `Verdict(...)` constructions in the package still
      compile and produce semantically identical objects (defaults
      fill in)
- [ ] `asdict(verdict)` includes both keys (empty list / None) — confirm
      this is fine for the JSON consumer in `agentm_bridge` (or, if
      the bridge round-trips JSON, ensure no schema mismatch on read)
- [ ] `uv run mypy src/llmharness/schema.py` strict-clean
- [ ] `uv run ruff check src/llmharness/schema.py` clean
- [ ] `validate_index.py project-index.yaml` reports 0 violations after
      the REQ-015 addition

## Notes

- No version bump in `pyproject.toml` — design §6.3 confirms additive
  fields don't require it.
- Do NOT add `to_dict` / `from_dict` helpers on `Verdict` unless absent
  ones already break a consumer; design says nullable defaults are
  enough.
- Do NOT touch `Event.refs` or introduce edge-type vocabulary — design
  §3.4 explicitly rejects this.
