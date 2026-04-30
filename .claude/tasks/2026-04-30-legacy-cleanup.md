# Task: Phase 2.5 — Legacy Tree Sweep (single PR)

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §8
**Agent**: implementer (sonnet) → reviewer (opus)
**Status**: BLOCKED on Phase 2 Groups A0, A, B, C, D1, D2 + reviewer composition test

## Goal

Delete the legacy code tree in one PR. The new `core/kernel/` + `core/operations.py` + `harness/` (v2) + `extensions/builtin/` + `extensions/scenarios/` + `llm/anthropic.py` is the only tree that ships.

## Deletion checklist

### Source tree

```
src/agentm/core/tool.py                           # legacy @tool decorator
src/agentm/core/trajectory.py                     # superseded by extensions.builtin.trajectory
src/agentm/harness/middleware.py
src/agentm/harness/runtime.py
src/agentm/harness/scenario.py
src/agentm/harness/worker_factory.py
src/agentm/harness/handle.py
src/agentm/harness/permission.py
src/agentm/harness/tool_filter.py
src/agentm/harness/cost_budget.py
src/agentm/harness/tool_result_budget.py
src/agentm/harness/system_reminder.py
src/agentm/harness/micro_compact.py
src/agentm/harness/agent_memory.py
src/agentm/harness/adapters.py                    # legacy langchain adapters
src/agentm/harness/protocols.py                   # legacy AgentLoop / Middleware Protocols
src/agentm/harness/tool.py                        # legacy Tool wrapper
src/agentm/harness/tool_concurrency.py            # superseded by extensions.builtin.file_mutation_queue
src/agentm/harness/types.py                       # legacy AgentHandle / Worker types
src/agentm/harness/loops/                         # entire directory
src/agentm/scenarios/                             # entire directory
src/agentm/tools/                                 # entire directory
src/agentm/agents/                                # entire directory
src/agentm/builder.py
```

After deleting the listed files, **rewrite** `src/agentm/harness/__init__.py`
to re-export ONLY the v2 surface:

```python
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.session_manager import (
    InMemorySessionManager, JsonlSessionManager, SessionEntry, SessionManager,
)
from agentm.harness.resource_loader import (
    DefaultResourceLoader, InMemoryResourceLoader, ResourceLoader,
)
from agentm.harness.extension import (
    CommandSpec, ExtensionAPI, ExtensionLoadError, ProviderConfig,
    ReadonlySession, UnknownCommandError, load_extension,
)
from agentm.harness import events
```

No re-exports of `Middleware`, `AgentRuntime`, `WorkerFactory`, `AgentHandle`,
or any other legacy type. Any test or downstream caller importing these must
migrate.

(If extra legacy files surface during the sweep, add them — the rule is "anything not reachable from `AgentSession.create` goes".)

### Test tree

```
tests/unit/harness/                               # entire directory (legacy harness tests)
tests/unit/scenarios/                             # entire directory
tests/unit/tools/                                 # entire directory
tests/unit/agents/                                # entire directory
tests/snapshot/                                   # entire directory (legacy pipeline snapshots)
tests/integration/v2_skeleton.py                  # superseded by tests/integration/scenarios/ + extension_composition.py
```

Keep:
- `tests/unit/kernel/`
- `tests/unit/harness_v2/`
- `tests/unit/llm/`
- `tests/unit/extensions/` (entire new tree)
- `tests/unit/core/operations/`
- `tests/integration/scenarios/`
- `tests/integration/extension_composition.py`

### Config / packaging

- `pyproject.toml` — remove any `langchain*` deps that were only there for legacy. Keep what `llm.anthropic` needs (the `anthropic` SDK).
- Update the `agentm:main` console-script entry point if it referenced `builder.py` — wire to a tiny `cli.py` that calls `AgentSession.create(...)`.
- Remove any `.claude/agents/` legacy references.

### Design docs

The legacy designs in `.claude/designs/` (`agent-harness.md`, `generic-state-wrapper.md`, `sdk-consistency.md`, `orchestrator.md`, `sub-agent.md`, `system-design-overview.md`, etc.) describe deleted code. Mark them with a status banner at the top:

```markdown
**Status**: HISTORICAL — describes the pre-v2 architecture removed in Phase 2.5 (2026-04-30).
The current architecture lives in [pluggable-architecture.md](pluggable-architecture.md) and
[extension-as-scenario.md](extension-as-scenario.md).
```

Do **not** delete the design docs — they remain as historical context. Update `index.yaml` to mark each affected concept's `status: historical`.

## Verification

After deletion:

```bash
# Nothing imports from deleted modules:
! grep -rEn "from agentm\.harness\.(middleware|runtime|scenario|worker_factory|handle|permission|tool_filter|cost_budget|tool_result_budget|system_reminder|micro_compact|agent_memory|adapters|protocols|tool|tool_concurrency|types|loops)\b|from agentm\.(scenarios|tools|agents|builder)|from agentm\.core\.tool import|from agentm\.core\.trajectory" src/ tests/

# Only the new harness/__init__.py is the v2 surface — no legacy re-exports leaked through:
! grep -E "Middleware|AgentRuntime|WorkerFactory|AgentHandle|build_agent_system" src/agentm/harness/__init__.py

# Full test suite green (only the new tree):
uv run pytest -q

# Lint + types:
uv run ruff check src/ tests/
uv run mypy src/agentm/

# Acceptance scenarios reachable:
uv run python -c "from agentm.extensions.loader import load_scenario; print([n for n in ['general_purpose','rca','trajectory_analysis','plan_mode'] if load_scenario(n)])"
```

All clean.

## HARD constraints

- **One PR**. No staged retirement.
- Do NOT add a compatibility shim or re-export layer. If a third-party caller broke because they imported `agentm.builder`, they migrate; we don't soften the cut.
- Commit message: `refactor: delete legacy harness/scenarios/tools tree (Phase 2.5)`. Body lists categories deleted with bullet counts (LoC removed, files removed).

## Report format (≤200 words)

1. Total LoC and file count removed (separate src/ vs tests/).
2. Any files that surfaced as legacy but weren't on this list — were they deleted?
3. Final `uv run pytest -q` result (number of tests, all green).
4. Final `mypy src/agentm/` clean: yes/no.
5. Confirmation that none of the verification greps found a match.
6. Any design docs that needed `status: historical` updates.
