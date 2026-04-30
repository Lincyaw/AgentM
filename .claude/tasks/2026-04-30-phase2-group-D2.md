# Task: Phase 2 Group D2 — Scenario Recipes + Loader

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §1, §1.1, §7.2
**Architecture**: [pluggable-architecture.md](../designs/pluggable-architecture.md) §6 (acceptance scenario 8 — plan mode reachable as a scenario, not a core feature)
**Agent**: implementer (sonnet)
**Status**: BLOCKED on Group A (`permission`), Group B (`system_prompt`), Group D1 (tool atoms)

## Why this proves the central claim

> **A scenario is a configuration file.** No new mechanism — just a YAML listing atoms.

After D2, "writing a new scenario" = writing a YAML. No Python module required. Third-party scenarios are structurally identical to built-in scenarios — a YAML in their own repo, fed to `AgentSession.create(extensions=load_scenario(path) + ...)`.

## Scope

### 1. Loader

`src/agentm/extensions/loader.py`:

```python
def load_scenario(name_or_path: str) -> list[tuple[str, dict]]:
    """Load a scenario YAML and return its extensions list.

    `name_or_path` is either:
      - a bare name like "rca" (resolved to <pkg>/extensions/scenarios/rca.yaml), or
      - an absolute filesystem path to a YAML file.
    Returns: list of (module_path, config_dict) tuples ready to pass to AgentSession.create(extensions=...).
    """
```

Implementation: ~30 lines. Uses `importlib.resources` for the bare-name case to find packaged YAMLs, falls back to a filesystem path. Validates: top-level key `extensions`, each entry has `module: str` and optional `config: dict`. Raises `ScenarioLoadError` with a clear message on bad input.

### 2. Scenario YAMLs

Four files under `src/agentm/extensions/scenarios/`:

#### `general_purpose.yaml`
```yaml
name: general_purpose
description: General-purpose coding agent with read/bash/edit/write tools.
extensions:
  - module: agentm.extensions.builtin.tool_read
  - module: agentm.extensions.builtin.tool_bash
  - module: agentm.extensions.builtin.tool_edit
  - module: agentm.extensions.builtin.tool_write
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: |
        You are a helpful coding assistant. Use the available tools to read,
        modify, and execute code. Be careful with destructive operations.
```

#### `rca.yaml`
```yaml
name: rca
description: Root cause analysis — data-only, hypothesis-driven.
extensions:
  - module: agentm.extensions.builtin.tool_read
  - module: agentm.extensions.builtin.tool_hypothesis_store
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: |
        You are an RCA analyst. Collect evidence; never draw conclusions
        without supporting data. Use the hypothesis store to track theories.
  - module: agentm.extensions.builtin.permission
    config:
      deny: ["bash", "edit", "write"]
```

#### `trajectory_analysis.yaml`
```yaml
name: trajectory_analysis
description: Analyze recorded agent trajectories.
extensions:
  - module: agentm.extensions.builtin.tool_read
  - module: agentm.extensions.builtin.tool_trajectory_loader
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: |
        You are a trajectory analyst. Load JSONL trajectories, summarize agent
        behavior, and surface notable events.
```

#### `plan_mode.yaml`
```yaml
name: plan_mode
description: Plan-mode framing — investigate but do not mutate.
extensions:
  - module: agentm.extensions.builtin.tool_submit_plan
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: |
        You are in plan mode. Investigate; do not mutate state. Call submit_plan
        when ready to share your plan with the user.
  - module: agentm.extensions.builtin.permission
    config:
      deny: ["bash", "edit", "write"]
```

### 3. Tests

`tests/unit/extensions/loader/test_load_scenario.py`:

1. **Bare name**: `load_scenario("rca")` returns 4 tuples in declaration order; first tuple's module is `tool_read`.
2. **Filesystem path**: same call but with absolute path to the YAML; same result.
3. **Bad YAML**: file with `extensions: not-a-list` → `ScenarioLoadError`.
4. **Missing module key**: entry with `config:` but no `module:` → `ScenarioLoadError`.
5. **All built-in scenarios are valid**: parameterize over `["general_purpose", "rca", "trajectory_analysis", "plan_mode"]`. For each: `load_scenario(name)` succeeds AND every `module` resolves via `importlib.import_module` AND the resolved module exports an `install` callable.

`tests/integration/scenarios/test_scenarios_smoke.py`:

For each of the 4 scenarios: `AgentSession.create(extensions=load_scenario(name), provider=fake_provider)` succeeds. Run a short prompt that triggers the scenario's primary tool (general_purpose → `read`; rca → `add_hypothesis`; trajectory_analysis → `load_trajectory` on a fixture file; plan_mode → `submit_plan`). Assert: tool registered, prompt invocation succeeds, expected SessionEntry types appear in `session.session_manager.get_active_branch()`.

`tests/integration/scenarios/test_plan_mode_blocks_mutations.py`:

Load `plan_mode` scenario. Fake provider issues a `bash` tool call. Assert: `permission` extension blocks it (returns `{"block": True, "reason": ...}`), no `bash` execution happens, and the agent receives a tool_result message with the rejection.

## HARD constraints

- `loader.py` imports allowed: stdlib + `agentm.harness.extension` (for `ExtensionLoadError` raising path) + a YAML parser (`pyyaml` — add to `pyproject.toml` if not present).
- Scenario YAMLs are pure data — no Jinja, no env-var substitution, no `!include`. Stay simple.
- `tests/integration/scenarios/` may use `agentm.harness.session.AgentSession`.

## Quality gates

```bash
uv run ruff check src/agentm/extensions/loader.py tests/unit/extensions/loader/ tests/integration/scenarios/
uv run mypy src/agentm/extensions/loader.py
uv run pytest tests/unit/extensions/loader/ tests/integration/scenarios/ -q
# All scenarios resolve:
uv run python -c "from agentm.extensions.loader import load_scenario; \
  [load_scenario(n) for n in ['general_purpose','rca','trajectory_analysis','plan_mode']]"
```

All pass.

## Report format (≤250 words)

1. Files added (loader, 4 YAMLs, tests).
2. Loader LoC.
3. Test counts.
4. Anything that revealed a missing config option in an atom (e.g. needed to add `config["mutation_tools"]` to `permission`). If so, the atom in question must be patched in its own group; document the round-trip.
5. Confirmation: 0 lines of Python in the scenarios. They are 100% YAML.
