# Task 2A: Implement Config Validator + PhaseManager.from_config

**Status**: PENDING
**Depends on**: 1B, 1C
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [system-design-overview.md](../designs/system-design-overview.md) § Configuration Validation
**Assignee**: implementer

## Objective

Implement `config/validator.py::validate_references` and `core/phase_manager.py::from_config`.

## Functions

### `validate_references(system, scenario, tool_registry) -> list[str]`
5 checks from docstring:
1. Agent model references exist in `system.models`
2. Orchestrator model exists in `system.models`
3. Agent tool references exist in `tool_registry.has(name)`
4. Prompt file references exist on disk (skip if no base dir)
5. Tool settings keys match tool's declared parameters

Return list of error strings. Empty = valid.

### `PhaseManager.from_config(config: dict) -> PhaseManager`
Parse YAML phase config dict. Build `PhaseDefinition` objects with `handler=None`. Determine `initial_phase` as first key. Raise `ValueError` if invalid.

## Verification

- `validate_references` returns empty list for valid config
- `PhaseManager.from_config` builds valid PhaseManager from minimal dict
- No test regression
