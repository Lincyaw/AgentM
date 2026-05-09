# Task: Issue #119 — Core kernel purification residuals (A5/A12/A13)

Sub-task of issue #73. Three post-review residuals in the core kernel:

- **A5** — Removed the hard-coded `.agentm/skills` and `.agentm/prompts`
  literal fallbacks from `core/_internal/skills.py` and
  `core/_internal/prompt_templates.py`. Project-scope directories now flow
  exclusively through `ProjectLayout.skills_dirs()` /
  `ProjectLayout.prompts_dirs()` (the harness's `DefaultProjectLayout`
  already supplies the historical paths). Sessions without a layout simply
  see no project-scope skills/prompts.
- **A12** — Collapsed `_default_action` and `_default_action_with_names` in
  `core/abi/loop.py` into a single `_default_action(assistant_msg,
  paired_outcomes)` helper. The unnamed variant was the older API and only
  re-entered through the named variant; it is gone. Tests and the loop call
  site updated accordingly.
- **A13** — Dropped the never-wired `on_update` parameter from the
  `Tool.execute` Protocol and from every implementation
  (`FunctionTool`, `tool_grep`, `tool_ls`, `tool_find`,
  `file_mutation_queue`). The Protocol docstring records the rationale: a
  real progress channel will be a deliberate event-bus extension rather
  than a dead Protocol parameter.

Gates: `uv run pytest --tb=short -q`, `uv run ruff check src/`,
`uv run mypy src/` all green.
