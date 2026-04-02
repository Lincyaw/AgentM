# Task: CodeSanitizer — Deterministic Checks

**Plan**: [investigation-sanitizer](../plans/2026-04-02-investigation-sanitizer.md)
**Phase**: 2 — CodeSanitizer
**Design**: [investigation-sanitizer](../designs/investigation-sanitizer.md) §7
**Depends on**: [investigation-tracker](2026-04-02-investigation-tracker.md)

## Scope

Implement all deterministic checks as pure functions. Each check reads from stores/tracker and returns `list[SanitizerFinding]`.

## Deliverables

### code_sanitizer.py — `src/agentm/scenarios/rca/sanitizer/code_sanitizer.py`

Individual check functions (design §7 has pseudocode for each):

| Function | Code | Trigger |
|----------|------|---------|
| `check_anchoring_bias` | E1 | periodic, pre_finalize |
| `check_dimension_gap` | E2 | periodic, pre_finalize |
| `check_coverage_gap` | E3 | periodic, pre_finalize |
| `check_premature_termination` | E4 | pre_finalize |
| `check_skipped_verify` | C1 | hypothesis_change, pre_finalize |
| `check_unresolved_contradiction` | C2 | hypothesis_change, pre_finalize |
| `check_no_alternative` | C4 | hypothesis_change, pre_finalize |
| `check_investigation_drift` | J2 | every_round |
| `check_incomplete_chain` | J3 | pre_finalize |
| `check_hypothesis_before_scout` | P1 | hypothesis_change |
| `check_profile_write_without_read` | P3 | every_round |

`CodeSanitizer` class:
- Implements `Sanitizer` protocol
- Constructor takes configurable severity mapping (`dict[str, Severity]`) and disabled codes (`set[str]`)
- `check(trigger, ...)` routes trigger to appropriate subset of check functions
- Each check function is a static/module-level function for testability

### Tests — `tests/unit/test_code_sanitizer.py`

Per check, two cases minimum:
1. **Fires**: set up store/tracker state that violates the rule → finding returned with correct code and severity
2. **Does not fire**: set up clean state → no findings

Additional:
- Trigger routing: verify `every_round` only runs J2/P3, `pre_finalize` runs all, etc.
- Severity override via config: check that configured severity is used
- Disabled check: verify disabled code produces no findings
- E2 dimension mapping: test that data_sources_queried entries map correctly to dimension categories
