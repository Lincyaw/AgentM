# Issue #121 — TUI dispatch + minimal flag follow-up

Closes the residual items from #73 §F that survived #98 / #99:

| Sub-item | Action |
|----------|--------|
| **F2** `--minimal` doc/impl drift | `agentm.minimal` and `--minimal` are gone. Removed the line from `CLAUDE.md`, the README "Minimal mode (recovery floor)" section, and the `agentm --minimal` reference in `.claude/designs/self-modifiable-architecture.md`. The recovery-floor invariant is now phrased as "use `agentm --no-extensions` to bypass atom discovery". |
| **F4** builtin command registry | Already done by #98 — `BuiltinCommandRegistry` + `register_extension_command` replaced the old `_BUILTIN_COMMANDS` dict + if-ladder. No further work needed for this PR. |
| **F5** delta dispatch table | Already done by #98 — `_DELTA_HANDLERS` and `_CHILD_DELTA_HANDLERS` (`dict[type[StreamDelta], Callable]`) plus `_lookup_delta_handler` replaced the `isinstance` ladder. No further work needed. |
| **F7** `is_self_modify` on `ExtensionReloadEvent` | Already done — `harness/events.py` defines `is_self_modify: bool = False`, the harness reloader sets it from `agent_initiated`, and `textual_app.handle_extension_reload` reads `event.is_self_modify` directly. The `_SELF_MODIFY_TRIGGERS` whitelist is gone. |
| **F8** phase glyphs | Moved the canonical glyph table out of `modes/textual_app.py` into a new `agentm.core.abi.presenter` module (`Phase`, `PHASE_GLYPHS` exported from `core.abi`). `DefaultTheme` now defaults `phase_glyphs` to the kernel-owned mapping. Custom themes can still override; the TUI no longer owns the constant. |

## Files touched

- `CLAUDE.md` — drop `--minimal` from the build commands table.
- `README.md` — replace "Minimal mode" section with a "Recovery floor" pointer to `--no-extensions`.
- `.claude/designs/self-modifiable-architecture.md` — update the recovery-floor paragraph (no more `agentm.minimal`).
- `src/agentm/core/abi/presenter.py` — **new**: `Phase` literal + `PHASE_GLYPHS` `MappingProxyType`.
- `src/agentm/core/abi/__init__.py` — export `Phase`, `PHASE_GLYPHS`.
- `src/agentm/modes/textual_app.py` — import kernel `Phase` / `PHASE_GLYPHS`; `DefaultTheme.phase_glyphs` defaults to the kernel mapping.
- `.claude/index.yaml` — append this task to `textual_tui.tasks`.

## Gates

- `uv run pytest --tb=short -q` — 188 passed.
- `uv run pytest -m ui --tb=short -q` — 15 passed.
- `uv run ruff check src/` — clean.
- `uv run mypy src/` — clean (110 files).

## Notes

The presenter glyph table sits in `core.abi` rather than as event-payload
data because it is a stable view-contract (every presenter wants the same
phase taxonomy), not a per-event signal. AC permitted either path; the
frozen-export route requires no event-bus surgery and keeps the TUI side
purely declarative.
