# Plan: Replace Anthropic-shaped `stop_reason` vocabulary with `TerminationHint`

**Date:** 2026-05-09
**Issue:** Lincyaw/AgentM#75 (parent: #73 — review items A3a, A3b, E15)

## Goal

Stop leaking the Anthropic vendor vocabulary through the kernel termination
path. The kernel must dispatch on a provider-agnostic sum-type that any LLM
adapter can populate, while keeping the raw vendor string available for
observability.

## Background

Before this change:
- `core/abi/messages.py` declared
  `stop_reason: Literal["end_turn", "tool_use", "max_tokens", "error", "aborted"] | None`
  — an Anthropic-shaped vocabulary embedded in the kernel.
- `core/abi/loop.py:_default_action` dispatched with
  `if raw == "max_tokens" / "error" / "tool_use"` — a vendor-shaped
  if-ladder inside the kernel's termination decision.
- Adding a provider whose protocol used different stop reasons required
  editing kernel code, violating the five-axes pluggability mandate.
- `core/abi/stream.py:Model.metadata` docstring named "anthropic-beta"
  and "OpenAI tool-choice quirks" — same drift, smaller surface (E15).

## Tasks

1. Introduce `core/abi/termination.py` defining the `TerminationHint`
   sum-type: `EndTurn | ToolUseExpected | MaxTokens | ProviderError |
   Aborted | VendorSpecific`. All variants are
   `@dataclass(slots=True, frozen=True)`.
2. `core/abi/messages.py`: change `stop_reason` to `str | None`; add
   `termination: TerminationHint | None = None` field to `AssistantMessage`.
   Keep `stop_reason` as raw vendor passthrough for observability.
3. `core/abi/loop.py:_default_action`: dispatch on
   `assistant_msg.termination` via `isinstance` ladder. Keep the legacy
   string-based fallback for providers that haven't migrated yet.
4. `core/abi/stream.py`: rephrase `Model.metadata` docstring to
   "vendor-specific bits, opaque to kernel" (E15).
5. `agentm.llm.anthropic`: replace `_map_stop_reason` so it returns a
   `TerminationHint` (`end_turn`/`stop_sequence` → `EndTurn`,
   `tool_use` → `ToolUseExpected`, `max_tokens` → `MaxTokens`, anything
   else → `VendorSpecific(raw)`). Populate both `stop_reason` (raw) and
   `termination` on the assembled `AssistantMessage`. Set `Aborted()` on
   the abort path.
6. `agentm.llm.openai`: rename `_map_stop_reason` to `_map_finish_reason`
   and translate OpenAI `finish_reason` strings (`stop` → `EndTurn`,
   `length` → `MaxTokens`, `tool_calls`/`function_call` → `ToolUseExpected`,
   `content_filter` → `ProviderError(detail="content_filter")`, else
   `VendorSpecific(raw)`). Populate both fields; set `Aborted()` on abort.
7. Re-export termination types from `agentm.core.abi.__init__`.
8. Add a unit test that drives `_default_action_with_names` with
   `termination=VendorSpecific(raw="custom_stop")` and asserts the loop
   terminates cleanly via `Stop(ModelEndTurn())`.

## Verification gate

- `uv run ruff check src/` — clean
- `uv run mypy src/` — 0 issues
- `uv run pytest --tb=short` — 92 passed (was 91; +1 new test)
- New test
  `tests/unit/core/abi/test_decide_turn_action.py::test_default_action_treats_vendor_specific_termination_as_end_turn`
  passes.

## Out of scope

- The legacy `stop_reason` string fallback in `_default_action` is kept as
  a graceful migration path. It will be removed once every shipped provider
  populates `termination` (tracked separately).
- Compaction synthetics (`core/_internal/compaction/compaction.py:210, 219`)
  still set `stop_reason="end_turn"` on synthesized messages; they don't
  flow through the loop's termination decision so they are unaffected.

## Files touched

- `src/agentm/core/abi/termination.py` (new)
- `src/agentm/core/abi/messages.py`
- `src/agentm/core/abi/loop.py`
- `src/agentm/core/abi/stream.py`
- `src/agentm/core/abi/__init__.py`
- `src/agentm/llm/anthropic.py`
- `src/agentm/llm/openai.py`
- `tests/unit/core/abi/test_decide_turn_action.py`
