# Issue 98: Textual TUI Registries and Monotone Turn IDs

## Requirement

Refactor the Textual presenter so slash commands, stream delta rendering,
event subscriptions, phase values, self-modification reload hints, and turn
identity are table/typed contracts rather than hardcoded presenter policy.

## Implementation Notes

- `BuiltinCommandRegistry` is the TUI command source of truth for built-in and
  `ApiRegisterEvent(kind="command")` commands.
- `ExtensionReloadEvent.is_self_modify` moves reload taxonomy out of the TUI.
- `Phase` and `Theme` make phase glyphs data-driven while keeping mode-level
  theme injection scoped to `AgentMApp.run()`.
- Stream delta handling and child-delta rendering use dispatch tables.
- Live bus subscriptions are declared in `_EVENT_SUBSCRIPTIONS` and registered
  with one loop.
- `AgentLoop` emits monotone `turn_id` values across `run()` calls; the TUI keys
  root turn widgets by `turn_id` instead of prompt-local epochs.

## Verification

- `tests/integration/test_textual_tui.py` covers registry palette + dispatch and
  distinct TUI turns from repeated `turn_index=0` prompts.
- `tests/unit/core/abi/test_turn_id.py` covers monotone turn IDs across prompt
  runs after a branch-style fork.
