# Task 2: Integration + Cleanup

**Plan**: [2026-03-24-trajectory-analysis](../plans/2026-03-24-trajectory-analysis.md)
**Design**: [trajectory-analysis](../designs/trajectory-analysis.md)
**Status**: TODO
**Depends on**: Tasks 1a, 1b, 1c, 1d

## Scope

Wire the new trajectory_analysis scenario into the system, remove old memory_extraction code.

### 1. Remove old scenario directory

Delete `src/agentm/scenarios/memory_extraction/` entirely:
- `__init__.py`, `state.py`, `strategy.py`, `data.py`, `enums.py`, `config.py`, `formatters.py`, `answer_schemas.py`, `output.py`

### 2. Update CLI references

Check and update:
- `src/agentm/cli/main.py` — any memory_extraction references
- `src/agentm/cli/run.py` — any memory_extraction references

### 3. Update models/types.py

The `TaskType` alias comment mentions memory-extraction task types. Update.

### 4. Update builder.py

Check if builder has any memory_extraction-specific code paths (it shouldn't after the earlier hypothesis_id cleanup, but verify).

### 5. Verify scenarios/__init__.py

Confirm `discover()` no longer imports from `memory_extraction`.

## Files to Delete

- `src/agentm/scenarios/memory_extraction/__init__.py`
- `src/agentm/scenarios/memory_extraction/state.py`
- `src/agentm/scenarios/memory_extraction/strategy.py`
- `src/agentm/scenarios/memory_extraction/data.py`
- `src/agentm/scenarios/memory_extraction/enums.py`
- `src/agentm/scenarios/memory_extraction/config.py`
- `src/agentm/scenarios/memory_extraction/formatters.py`
- `src/agentm/scenarios/memory_extraction/answer_schemas.py`
- `src/agentm/scenarios/memory_extraction/output.py`

## Files to Modify

- `src/agentm/cli/main.py`
- `src/agentm/cli/run.py`
- `src/agentm/models/types.py`
- `src/agentm/scenarios/__init__.py` (if not already done in 1b)
