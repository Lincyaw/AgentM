# Task 1C: Implement Prompt Loader and Tool Registry

**Status**: PENDING
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [sub-agent.md](../designs/sub-agent.md) § Prompt Management, § Tool Registry
**Assignee**: implementer

## Objective

Implement `core/prompt.py` (1 function) and `core/tool_registry.py` (4 methods + `ToolDefinition.create_with_config`).

## Functions

### `load_prompt_template(path, **context) -> str`
Read `.j2` file, render with `Template(text).render(**context)`. Add `jinja2` dependency.

### `ToolRegistry.register(name, func, config_schema)`
Store `ToolDefinition(name, func, config_schema)` in `self._tools[name]`.

### `ToolRegistry.get(name) -> ToolDefinition`
Return `self._tools[name]`. `KeyError` if not found.

### `ToolRegistry.has(name) -> bool`
Return `name in self._tools`.

### `ToolRegistry.load_from_yaml(path)`
Load YAML. For each tool: `importlib.import_module(module)`, `getattr(mod, function)`, `register(name, func, params)`.

### `ToolDefinition.create_with_config(**config) -> Tool`
Create LangChain `Tool` with config params bound via closure.

## Verification

- `test_tool_signatures.py` still passes
- `test_interface_consistency.py` still passes
- Round-trip: register → get → has works

## Notes

- Add `jinja2` dependency: `uv add jinja2`
- Use `langchain_core.tools.Tool` for `create_with_config`
