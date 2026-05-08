# Task 03: Auditor Module

**Date**: 2026-05-08
**Status**: PENDING
**Plan**: [plan](../plans/2026-05-08-llmharness-two-phase-audit.md)
**Design**: [design](../designs/llmharness-two-phase-audit.md) §2 (Phase 2), §6.2, §7.1
**Assignee**: implementer
**Size**: M
**Dependencies**: 01 (independent of 02; can run in parallel)

## Objective

Implement the Phase 2 (auditor) child-session payload as a self-contained
subpackage. Stateless: the adapter passes in the event graph (Event.to_dict
form) and recent verdicts; auditor does NOT see the raw trajectory.

## Files

Create:
- `scenarios/llmharness/src/llmharness/audit/auditor/__init__.py`
- `scenarios/llmharness/src/llmharness/audit/auditor/prompt.py`
- `scenarios/llmharness/src/llmharness/audit/auditor/submit_tool.py`
- `scenarios/llmharness/src/llmharness/audit/auditor/extensions.py`
- `scenarios/llmharness/src/llmharness/audit/auditor/output.py`

Read:
- `scenarios/llmharness/src/llmharness/audit/_enum_schema.py` (task 01)
- `scenarios/llmharness/src/llmharness/audit/prompt.py` (V0 reference)
- `scenarios/llmharness/src/llmharness/schema.py` (`Verdict`, `Reminder`, `DriftType`, `EventKind`)

## Module Contracts

### `prompt.py`
- `AUDITOR_SYSTEM_PROMPT: str` — graph-only audit prompt. Explicitly
  instructs the agent that it sees the structured event graph, NOT the
  raw trajectory. Three-axis check (backward continuity, forward
  fulfillment, content correctness) operates over events. Terminates by
  calling `submit_verdict`.

### `submit_tool.py`
- `MANIFEST` + `install(api, config)`.
- Tool name: `submit_verdict`.
- JSON Schema for `verdict`: object with `drift: bool`, `type: enum from DRIFT_TYPE_VALUES (nullable)`, `reminder: object|null`, `cited_cards: array[str]|null`, `downstream_reaction: string|null`. `additionalProperties: False`.
- **`if/then` block**: `if drift == true then required: ["type"]` and `type` must NOT be null. Per design §6.2 + §8 (closes the "drift=true with type=null silently dropped" V0 bug at the provider edge).
- Returns `ToolTerminate` with `{"verdict": {...}}`.
- Constants: `SUBMIT_VERDICT_TOOL_NAME`, `SUBMIT_VERDICT_PARAMETERS`.

### `extensions.py`
- `compose_auditor_extensions(*, prompt_override=None, cards_tools_config=_UNSET, observability_config=_UNSET) -> list[tuple[str, dict]]`
- Module order:
  1. `agentm.extensions.builtin.observability`
  2. `llmharness.atoms.cards_tools` (auditor still benefits from card lookup when constructing reminders)
  3. `llmharness.audit.auditor.submit_tool`
  4. `agentm.extensions.builtin.system_prompt` (prompt = `AUDITOR_SYSTEM_PROMPT` or override)

### `output.py`
- `RawVerdictOutput` typed coercion: payload → `Verdict | None`. Named
  exception (`AuditorOutputError`) on malformed.

### `__init__.py`
- Re-export `AUDITOR_SYSTEM_PROMPT`, `compose_auditor_extensions`,
  `RawVerdictOutput`, `AuditorOutputError`, `SUBMIT_VERDICT_TOOL_NAME`.

## Acceptance

- [ ] All five files exist, compile, pass `mypy --strict`.
- [ ] `submit_verdict` JSON Schema includes the `if/then` block enforcing
      `drift=true ⇒ type required` (and non-null).
- [ ] `type` enum derives from `DRIFT_TYPE_VALUES` (task 01).
- [ ] `compose_auditor_extensions()` default returns the four-module list above.
- [ ] `RawVerdictOutput` round-trips a well-formed payload to `Verdict`; rejects malformed input with a named exception.
- [ ] Auditor module makes NO reference to raw trajectory / messages slicing.

## Notes

- The auditor prompt should NOT instruct on serializing turns — it works
  on `Event.to_dict()` form. Adapter (task 04) is responsible for
  formatting the graph into the user-message content.
- `fetch_turn` tool: design defers to V2 (§2 Phase 2, §10). Do NOT
  register it.
- `cited_cards` and `downstream_reaction` remain free-text (no preset
  enums; per design §5.2 + memory `feedback_no_preset_subjective_labels`).
