# Task 02: Extractor Module

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-two-phase-audit.md)
**Design**: [design](../designs/llmharness-two-phase-audit.md) §2 (Phase 1), §6.1, §7.1
**Assignee**: implementer
**Size**: M
**Dependencies**: 01

## Objective

Implement the Phase 1 (extractor) child-session payload as a self-contained
subpackage. Stateless: no cursor read here, the adapter slices new turns
and passes them in.

## Files

Create:
- `scenarios/llmharness/src/llmharness/audit/extractor/__init__.py`
- `scenarios/llmharness/src/llmharness/audit/extractor/prompt.py`
- `scenarios/llmharness/src/llmharness/audit/extractor/submit_tool.py`
- `scenarios/llmharness/src/llmharness/audit/extractor/extensions.py`
- `scenarios/llmharness/src/llmharness/audit/extractor/output.py`

Read:
- `scenarios/llmharness/src/llmharness/audit/_enum_schema.py` (from task 01)
- `scenarios/llmharness/src/llmharness/audit/extensions.py` (V0 reference; do NOT delete yet)
- `scenarios/llmharness/src/llmharness/audit/submit_tool.py` (V0 reference)
- `scenarios/llmharness/src/llmharness/schema.py` (`Event`, `EventKind`)

## Module Contracts

### `prompt.py`
- `EXTRACTOR_SYSTEM_PROMPT: str` — graph-extraction-only system prompt.
  Operates over the new-turn slice (incl. thinking blocks) plus a recent
  graph slice for `refs` resolution. Tells the agent to terminate by
  calling `submit_events` with one or more `Event` objects ref-linked to
  prior events (free-text relation in `summary` per design §5.2).

### `submit_tool.py`
- Module-level `MANIFEST: ExtensionManifest` and `install(api, config)`
  per `single_file_extension_contract`.
- Tool name: `submit_events`.
- JSON Schema for `events`: array of objects whose `kind` enum is built
  from `EVENT_KIND_VALUES` (task 01); `refs: array[int]`; `summary: str`;
  `source_turns: array[int]`. `additionalProperties: False`.
- Returns `ToolTerminate` with structured payload `{"events": [...]}`.
- Empty `events` array IS legal at the schema level — adapter classifies
  as `extractor_empty` based on input window size.
- Exposes constants `SUBMIT_EVENTS_TOOL_NAME`, `SUBMIT_EVENTS_PARAMETERS`.

### `extensions.py`
- `compose_extractor_extensions(*, prompt_override=None, cards_tools_config=_UNSET, observability_config=_UNSET) -> list[tuple[str, dict]]`
- Same overall shape as V0 `compose_extensions`, with these modules in
  this order:
  1. `agentm.extensions.builtin.observability`
  2. `llmharness.atoms.cards_tools`
  3. `llmharness.audit.extractor.submit_tool`
  4. `agentm.extensions.builtin.system_prompt` (prompt = `EXTRACTOR_SYSTEM_PROMPT` or override)

### `output.py`
- `RawExtractorOutput` typed coercion: takes the raw `submit_events`
  payload, validates and returns `list[Event]`. Coercion errors raise a
  named exception (e.g. `ExtractorOutputError`) that the adapter catches
  to write `extractor_error`.

### `__init__.py`
- Re-export `EXTRACTOR_SYSTEM_PROMPT`, `compose_extractor_extensions`,
  `RawExtractorOutput`, `ExtractorOutputError`,
  `SUBMIT_EVENTS_TOOL_NAME`.

## Acceptance

- [ ] All five files exist, compile, pass `mypy --strict`.
- [ ] `submit_events` JSON-Schema `kind` enum derives from `EVENT_KIND_VALUES`.
- [ ] `compose_extractor_extensions()` default returns the four-module list above; `cards_tools_config=None` drops cards; `observability_config=None` drops observability; `submit_tool` and `system_prompt` always survive.
- [ ] `RawExtractorOutput` round-trips a well-formed payload to `list[Event]`; rejects malformed input with a named exception.
- [ ] No imports from `llmharness.audit.{prompt,submit_tool,output,extensions}` (the V0 flat layout) — extractor is self-contained.

## Notes

- Mirror the `_UNSET` sentinel pattern from V0 `compose_extensions` so
  "default include with empty config" stays distinct from "explicit None
  drops".
- Extractor sees thinking blocks — but slicing happens in the adapter
  (task 04). This module only owns the prompt + tool + composition.
- Do NOT register `fetch_turn` or any drill-down tool; design defers to V2.
