# Plan: Phase 2 — Parallel Built-in Extensions + Legacy Sweep

**Created**: 2026-04-30 (revised same day for design alignment)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §7, §8, §9, §10b
**Architecture**: [pluggable-architecture.md](../designs/pluggable-architecture.md)
**Predecessors**: Phase 1 (v2 harness foundation) + Phase 1b (Anthropic StreamFn) + Phase 2.0 (events + slash-command runner) + **Phase 2.0b (load-bearing kernel/harness contracts — append_entry, pending-message drain, missing event dataclasses, cost_budget_exceeded subscription, session_ready event, prompt() helper extraction)** — all landed, 60/60 tests passing.

---

## Goal

Re-implement the catalog in [extension-as-scenario.md §7](../designs/extension-as-scenario.md#7-catalog-atomic-extensions--scenario-recipes) on the v2 kernel + ExtensionAPI. **Two layers**:

1. **Atomic extensions** — single-responsibility code modules under `src/agentm/extensions/builtin/`. Each registers exactly one capability. Reused across scenarios.
2. **Scenario recipes** — YAML files under `src/agentm/extensions/scenarios/` listing which atoms to load and how to configure them. **Pure data, no Python.**

Plus a `load_scenario(name) -> list[(module, config)]` helper (~30 lines). Caller flattens scenarios into the `extensions=[...]` argument of `AgentSession.create`.

After all groups land, **delete the entire legacy tree in one sweep** (Phase 2.5) — no compatibility layer, no parallel universe.

## Why this is parallelizable

After Phase 2.0 + Group A0 + Group 0c, every extension has the same shape (locked by design §11):
- Imports: restricted by the §11.1 allow-list (stdlib + `agentm.core.kernel` + `agentm.core.operations` + `agentm.harness.{extension,events,session_manager,resource_loader}` + `agentm.extensions`). **No atom-to-atom imports**, **no `agentm.harness.session` imports**.
- Surface: `install(api, config) -> None | Awaitable[None]` plus a module-level `MANIFEST: ExtensionManifest`.
- File: **exactly one** `.py` file under `src/agentm/extensions/builtin/` per atom. Subpackages forbidden.
- Tests: `tests/unit/extensions/builtin/<name>/test_*.py` — **path locked, design §10b.10**.
- Mechanical gate: `tests/unit/extensions/test_extension_contract.py` runs `validate_builtin()` over the catalog on every PR.

Different extensions touch different files → no merge conflicts. The single-file rule + validator gate were chosen deliberately so future agent-driven self-edits have a tight, verifiable surface (design §11).

## Group breakdown

| Group | Atoms / artifacts | Risk | Depends on |
|---|---|---|---|
| **0b — Pre-parallel gate** | `ReadonlySession.append_entry`; `_pending_user_messages` drain; missing event dataclasses (`CostBudgetExceededEvent`, `PlanSubmittedEvent`, `SessionReadyEvent`); `cost_budget_exceeded` subscription in `AgentSession.create`; `session_ready` emission; `prompt()` helper extraction | Low | Phase 2.0 |
| **0c — Single-file extension contract** | `agentm.extensions.{__init__,discover,validate}`; `ExtensionManifest` + tag parser; `validate_builtin()` + contract test gate; design §11 | Low | Phase 2.0 |
| **A0 — Operations ports** | `core/operations.py` (`FileOperations`, `BashOperations` + Local impls) | Low | Phase 2.0 |
| **A — Policy atoms** | permission · tool_filter · dedup · cost_budget · tool_result_budget | Low | **0b** |
| **B — Context atoms** | turn_reminder · system_prompt · file_mutation_queue · micro_compact · trajectory | Medium | **0b** |
| **C — Multi-agent atom** | sub_agent | High | **0b** |
| **D1 — Tool atoms** | tool_read · tool_bash · tool_edit · tool_write · tool_hypothesis_store · tool_trajectory_loader · tool_submit_plan | Medium | **A0**, **0b** |
| **D2 — Scenario recipes** | YAMLs (general_purpose, rca, trajectory_analysis, plan_mode) + `loader.py` | Low | **A** (permission), **B** (system_prompt), **D1** |

0b and A0 are short serial gates. They unblock the parallel batch (A, B, C, D1). D2 is the closing seam — it consumes everything else.

## Locked design decisions (see `extension-as-scenario.md §10b`)

| # | Decision |
|---|---|
| 10b.4 | Extensions process in declaration order; `file_mutation_queue` MUST appear after scenarios that register `edit`/`write`; fast-fail with `ExtensionLoadError` if not. |
| 10b.5 | `inject_instruction` v0 = queue-next-prompt (no mid-turn streaming). StreamFn closure shared across child sessions; tested explicitly. |
| 10b.6 | `FileOperations` / `BashOperations` ports live in `core/operations.py`; `general_purpose` tools delegate via injected ops. |
| 10b.7 | `ReadonlySession.append_entry(type, payload, parent_id=None) -> str` — added by Group B. |
| 10b.8 | `cost_budget` overflow = emit `cost_budget_exceeded` event; `AgentSession.prompt` subscribes and terminates with `agent_end(stop_reason="budget")`. No exceptions across handler boundaries. |
| 10b.9 | `plan_mode` = extension (not core mode). Prepends prompt, registers `submit_plan`, optionally cooperates with `permission` to deny mutations. |
| 10b.10 | Test path = `tests/unit/extensions/builtin/<name>/`. |

## Acceptance per group

Each group's task file lists its own acceptance criteria. Common ones:
- `uv run pytest tests/unit/extensions/builtin/<group>/ -v` passes.
- `uv run ruff check` and `uv run mypy` clean on new files.
- An end-to-end demo test: `AgentSession.create(extensions=[(<this_extension>, {...})])` + a fake provider + call `prompt(...)` exercises the extension's primary value.
- Layer purity grep: no legacy harness imports, no langchain imports, no `subprocess`/`pathlib` in `general_purpose` tools.

## Reviewer pass — composition test (mandatory)

After all groups land, one reviewer agent writes `tests/integration/extension_composition.py` that:
- Loads `{permission, dedup, cost_budget, tool_result_budget, micro_compact, trajectory, sub_agent, rca}` together
- Runs a session that triggers each extension's value path at least once
- Asserts: no event-handler conflicts, no double-blocking, trajectory captured all events from all extensions, child sessions inherit the configured `inherit_extensions` set

Plus the standard cross-checks:
- Each extension's behavior matches its legacy counterpart (where applicable; legacy is read-only reference until 2.5)
- New events used correctly (`before_compact`, `child_session_*`, `before_send_to_llm`, `cost_budget_exceeded`, `plan_submitted`)
- Acceptance scenarios from `pluggable-architecture.md` §6 — all 8 reachable without core fork

## Phase 2.5 — Legacy sweep (single PR after Phase 2 lands)

See [tasks/2026-04-30-legacy-cleanup.md](../tasks/2026-04-30-legacy-cleanup.md) for the full deletion checklist. High level: delete `src/agentm/{scenarios,tools,agents,builder.py}`, `src/agentm/harness/{middleware,runtime,scenario,worker_factory,handle,permission,tool_filter,cost_budget,tool_result_budget,system_reminder,micro_compact,agent_memory,loops}*`, `src/agentm/core/{tool,trajectory}.py`, plus all corresponding tests under `tests/unit/{harness,scenarios,tools,agents}/` and `tests/snapshot/`. Design docs that describe the deleted code stay as historical context with a status banner.

No staged retirement; the new tree is the only tree.

## Pacing recommendation

1. Sequential prerequisites: **0b** (harness contracts) + **0c** (single-file extension scaffolding) + **A0** (ops ports) — can run in parallel with each other since they touch different files.
2. Parallel batch: **A + B + C + D1** (4 implementer agents — atoms are independent; each one is a single `.py` file under `extensions/builtin/`).
3. Sequential closer: **D2** (scenario YAMLs + loader; consumes A/B/D1 outputs).
4. Reviewer composition test.
5. **Phase 2.5 legacy sweep**.

## Deferred (NOT in Phase 2)

- LangChain bridge (`llm.langchain_bridge`) — schedule when first user wants a non-Anthropic langchain provider.
- OpenAI provider — schedule on demand.
- TUI/interactive mode — Phase 3.
- Render-message extension surface — when interactive mode lands.
