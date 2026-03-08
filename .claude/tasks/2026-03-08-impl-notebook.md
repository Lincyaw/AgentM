# Task 1A: Implement Notebook Operations

**Status**: PENDING
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [orchestrator.md](../designs/orchestrator.md) § Hypothesis lifecycle
**Assignee**: implementer

## Objective

Implement all 7 functions in `src/agentm/core/notebook.py`. All operations must return a NEW `DiagnosticNotebook` instance (immutability contract).

## Functions

### `validate_hypothesis_transition(current, target) -> bool`
State machine lookup. Use `LEGAL_TRANSITIONS` dict. Return `False` for self-transitions and unlisted targets.

```python
LEGAL_TRANSITIONS = {
    FORMED: {INVESTIGATING},
    INVESTIGATING: {CONFIRMED, REJECTED, REFINED, INCONCLUSIVE},
    REFINED: {INVESTIGATING},
    INCONCLUSIVE: {INVESTIGATING},
    REJECTED: {REFINED},
    CONFIRMED: set(),  # terminal
}
```

### `add_hypothesis(notebook, hypothesis_id, description, created_at) -> DiagnosticNotebook`
`dataclasses.replace()` + copy `hypotheses` dict. Add new `Hypothesis(...)`.

### `update_hypothesis_status(notebook, hypothesis_id, status, last_updated, evidence, counter_evidence) -> DiagnosticNotebook`
Validate transition first. Copy dict + hypothesis. Append to evidence lists (don't replace). Raise `ValueError` on illegal transition.

### `add_exploration_step(notebook, step) -> DiagnosticNotebook`
Copy `exploration_history` list, append `step`, increment `current_step`.

### `add_collected_data(notebook, agent_id, data) -> DiagnosticNotebook`
Copy `collected_data` dict, set `collected_data[agent_id] = data`.

### `set_confirmed_hypothesis(notebook, hypothesis_id) -> DiagnosticNotebook`
Set `confirmed_hypothesis = hypothesis_id`.

### `format_notebook_for_llm(notebook) -> str`
Format into LLM-readable string. Include phase summaries, exploration history, hypotheses with status/evidence.

## Verification

- Remove `@pytest.mark.skip` from `test_notebook_immutability.py` (5 classes, 10 tests)
- Remove `@pytest.mark.skip` from `TestValidateHypothesisTransition` (15 parametrized tests)
- `test_design_contracts.py` still passes

## Notes

- `dataclasses.replace()` is shallow — copy nested mutables explicitly: `{**notebook.hypotheses}`, `list(notebook.exploration_history)`
- `format_notebook_for_llm` tested in Layer 2, no specific unit test yet
