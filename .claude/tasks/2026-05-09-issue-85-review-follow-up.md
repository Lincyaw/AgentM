# Issue 85 Review Follow-up Verification

Context: the first dev pass landed the five sequential refactor steps in commit
`5af09cd`. This follow-up addresses review feedback and records replayable
acceptance evidence for the completed branch.

## Review Fixes

- Slash command input handling now subscribes at `BusPriority.PRE`, so command
  dispatch runs before prompt-template `input` rewrites regardless of atom
  installation order.
- Added CLI trajectory coverage for the `/cmd` vs prompt-template collision:
  `tests/integration/test_cli_slash_commands.py` drives `agentm --cwd <sandbox>
  "/ship now"`, then inspects `.agentm/observability/*.jsonl` for
  `emit:command_dispatched` and verifies no LLM request was started.

## Step Gate Evidence

The implementation was already consolidated when this follow-up began, so the
step gates below are acceptance replays against the final branch plus the new
review fix. Each row is backed by the full verification commands in the final
section.

| Step | Acceptance evidence |
| --- | --- |
| Step 1 | `grep -n '_budget_exceeded\|self\._commands\.get' src/agentm/harness/session.py` returns no hits; `slash_commands.py` exists and passes §11; `test_cost_budget_veto_emits_budget_exhausted_agent_end` passes; `test_cli_slash_command_wins_over_prompt_template_collision` passes. |
| Step 2 | `grep -R -n '"agentm.extensions.builtin.inherit_provider"' src/agentm/harness/` returns no hits; `grep -R -n 'next(reversed(providers))' src/agentm/harness/` returns no hits; `ProviderResolver` and `LastRegisteredWins` exist; `test_custom_provider_resolver_selects_named_provider` passes. |
| Step 3 | `grep -R -n 'loop\._stream_fn' src/agentm/harness/` returns no hits; `grep -R -n 'isinstance.*GitBackedResourceWriter\|isinstance.*ResourceWriter' src/agentm/harness/` returns no hits. |
| Step 4 | `src/agentm/harness/session.py` is 388 lines and construction discovery is in `_iter_auto_discovered_atoms(...)`. |
| Step 5 | `AgentSession.__init__` has 6 parameters excluding `self`; `SessionRuntime` exists; `test_fork_at_deep_copies_entry_payloads` passes. |

## Final Verification Commands

- `uv run ruff check src/` -> passed.
- `uv run mypy src/` -> passed with 0 issues across 101 source files.
- `uv run pytest --tb=short` -> `127 passed, 14 deselected`.
- Configured §11 validator via `configure_manifest_path(Path('core-manifest.yaml'))` -> 0 issues.
- `uv run python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml` -> pass.
- Grep gates listed above -> no hits where zero-hit acceptance was required.
