# Task 1B: Implement Config Loader

**Status**: PENDING
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [system-design-overview.md](../designs/system-design-overview.md) § Configuration System
**Assignee**: implementer

## Objective

Implement all 4 functions in `src/agentm/config/loader.py`.

## Functions

### `substitute_env_vars(data: dict) -> dict`
Recursively walk dict. For string values matching `${VAR_NAME}`, replace with `os.environ[VAR_NAME]`. Raise `KeyError` if env var not set. Handle nested dicts and lists.

### `load_system_config(path) -> SystemConfig`
1. `yaml.safe_load` the file
2. `substitute_env_vars` on loaded dict
3. `SystemConfig(**data)` (Pydantic validates)

### `load_scenario_config(path) -> ScenarioConfig`
Same pattern as `load_system_config`.

### `load_tool_definitions(tools_dir) -> dict[str, Any]`
Read all `*.yaml` files in dir. Each has `tools:` key with tool definitions. Merge into single flat dict.

## Verification

- `test_config_validation.py` still passes
- Manual: `load_system_config` on valid YAML returns `SystemConfig`
- `substitute_env_vars` replaces `${VAR}` in nested structures

## Notes

- Add `pyyaml` dependency: `uv add pyyaml`
- Env var resolution happens BEFORE Pydantic parsing
