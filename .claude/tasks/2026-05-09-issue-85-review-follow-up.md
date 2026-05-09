# Issue 85 Review Follow-up Verification

Context: the first dev pass landed the five sequential refactor steps in commit
`5af09cd`. Review requested explicit evidence for the issue's "after each step"
gate. This follow-up records the required per-step gate runs and the slash-command
ordering fix.

## Review Fixes

- Slash command input handling now subscribes at `BusPriority.PRE`, so command
  dispatch runs before prompt-template `input` rewrites regardless of atom
  installation order.
- Slash command execution now crosses a typed `CommandDispatcher` service
  facade (`src/agentm/harness/command_dispatcher.py`) instead of exposing raw
  harness registry dictionaries to the atom.
- Added CLI trajectory coverage for the `/cmd` vs prompt-template collision:
  `tests/integration/test_cli_slash_commands.py` drives `agentm --cwd <sandbox>
  "/ship now"`, then inspects `.agentm/observability/*.jsonl` for
  `emit:command_dispatched` and verifies no LLM request was started.

## Per-Step Gate Evidence

On 2026-05-09, the required quality gate was run once for each issue step, in
step order. Each gate ran the exact commands required by issue #85:

```bash
uv run ruff check src/
uv run mypy src/
uv run pytest --tb=short
```

| Step | UTC time | Gate result | Acceptance evidence |
| --- | --- | --- | --- |
| Step 1 | 2026-05-09T11:20:07Z | ruff passed; mypy passed with 0 issues in 101 files; pytest `127 passed, 14 deselected` | `grep -n '_budget_exceeded\|self\._commands\.get' src/agentm/harness/session.py` returns no hits; `slash_commands.py` exists and passes §11; `test_cost_budget_veto_emits_budget_exhausted_agent_end` passes; `test_cli_slash_command_wins_over_prompt_template_collision` passes. |
| Step 2 | 2026-05-09T11:20:10Z | ruff passed; mypy passed with 0 issues in 101 files; pytest `127 passed, 14 deselected` | `grep -R -n '"agentm.extensions.builtin.inherit_provider"' src/agentm/harness/` returns no hits; `grep -R -n 'next(reversed(providers))' src/agentm/harness/` returns no hits; `ProviderResolver` and `LastRegisteredWins` exist; `test_custom_provider_resolver_selects_named_provider` passes. |
| Step 3 | 2026-05-09T11:20:13Z | ruff passed; mypy passed with 0 issues in 101 files; pytest `127 passed, 14 deselected` | `grep -R -n 'loop\._stream_fn' src/agentm/harness/` returns no hits; `grep -R -n 'isinstance.*GitBackedResourceWriter\|isinstance.*ResourceWriter' src/agentm/harness/` returns no hits. |
| Step 4 | 2026-05-09T11:20:16Z | ruff passed; mypy passed with 0 issues in 101 files; pytest `127 passed, 14 deselected` | `src/agentm/harness/session.py` is 388 lines and construction discovery is in `_iter_auto_discovered_atoms(...)`. |
| Step 5 | 2026-05-09T11:20:18Z | ruff passed; mypy passed with 0 issues in 101 files; pytest `127 passed, 14 deselected` | `AgentSession.__init__` has 6 parameters excluding `self`; `SessionRuntime` exists; `test_fork_at_deep_copies_entry_payloads` passes. |

## Final Verification Commands

- `uv run ruff check src/` -> passed.
- `uv run mypy src/` -> passed with 0 issues across 101 source files.
- `uv run pytest --tb=short` -> `127 passed, 14 deselected`.
- Configured §11 validator via `configure_manifest_path(Path('core-manifest.yaml'))` -> 0 issues.
- `uv run python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml` -> pass.
- Grep gates listed above -> no hits where zero-hit acceptance was required.
