# Design: Testing Strategy

**Status**: APPROVED
**Created**: 2026-03-08
**Last Updated**: 2026-03-28

## Overview

Testing strategy for the AgentM agent framework. Three-layer architecture
aligned with `SimpleAgentLoop` + `AgentRuntime`. No LangGraph.

### Testing Philosophy (from CLAUDE.md)

1. **Test behavior, not structure** -- never test language guarantees.
2. **Every test must answer "what bug does this prevent?"**
3. **Boundaries over happy paths** -- edge cases are where bugs live.
4. **One scenario per test, multiple asserts are fine.**
5. **Do not test other people's code** -- Pydantic, dataclass semantics, etc.

### Layer Architecture

```
+-----------------------------------------------------------+
| Layer 3: Eval Tests (future)                              |
| rcabench-platform, LLM-as-Judge | pre-release / nightly   |
+-----------------------------------------------------------+
| Layer 2: Snapshot / Integration Tests                     |
| Mock LLM + FakeWorkerLoop       | per-commit (CI)         |
+-----------------------------------------------------------+
| Layer 1: Unit Tests                                       |
| Pure functions, protocols        | per-commit (CI)         |
+-----------------------------------------------------------+
```

---

## Layer 1: Unit Tests (`tests/unit/`)

Deterministic tests. No LLM calls, no external dependencies.

**SDK Protocol and Contract Tests**

- **`test_generic_sdk.py`** -- Scenario registry resolution, `Scenario.setup()` wiring. Bug: typo in scenario name silently returns None.
- **`test_tool_signatures.py`** -- Introspection that tool signatures match design. Checks `Literal` values vs `HypothesisStore._VALID_STATUSES`, async markers, removed params. Bug: signature drift causes LLM schema mismatch.
- **`test_interface_consistency.py`** -- Cross-module type/signature compatibility. Bug: Module A expects type X, Module B says type Y.
- **`test_sdk_boundary.py`** -- AST guard: SDK core never imports `agentm.scenarios`. Bug: scenario logic leaks into SDK core.

**Component Behavior Tests**

- **`test_simple_agent_loop.py`** -- ReAct loop: completion, tool execution, multi-step, inbox injection, middleware hooks (`on_llm_start`/`on_llm_end`/`on_tool_call`), `max_steps` exhaustion, `output_schema` retry+fallback, streaming events, `should_terminate`. Uses `MockModel`, `MockTool`, middleware fakes.
- **`test_agent_runtime.py`** -- `AgentRuntime` lifecycle: spawn, abort, get_status, get_result, send, concurrent management. Uses `FakeAgentLoop`.
- **`test_service_profile.py`** -- `ServiceProfileStore`: CRUD, merge logic (topology union, observations append, anomaly upgrade-only), query filters, LLM formatting, thread safety, frozen-dataclass immutability. Bug: concurrent updates lose data.
- **`test_trajectory_collector.py`** -- `TrajectoryCollector` JSONL capture, monotonic sequence. Bug: out-of-order breaks timeline.
- **`test_dedup.py`** -- `DedupTracker`: key generation, FIFO eviction, store/lookup. Bug: identical tool calls re-executed.

**Config, CLI, and Observability Tests**

- **`test_config_validation.py`** -- Pydantic schema rejection boundaries only.
- **`test_cli_parse.py`** -- CLI argument parsing via `CliRunner`.
- **`test_checkpointer_factory.py`** -- `DebugConfig` / `SystemConfig` schema.
- **`test_observability_parameterized.py`** -- Parameterized SQL builders. Bug: SQL injection via LLM-controlled filters.
- **`test_debug_analysis.py`** -- Post-hoc trajectory analysis CLI.

**Vault Tests**

- **`test_vault_parser.py`** -- Markdown + YAML frontmatter pure functions.
- **`test_vault_store.py`** -- `MarkdownVault` CRUD, edit, rename, batch, thread safety.
- **`test_vault_schema.py`** -- SQLite DDL creation, vec support probe.
- **`test_vault_search.py`** -- Keyword, semantic, hybrid search, filters.
- **`test_vault_graph.py`** -- Backlinks, traverse, lint.
- **`test_vault_tools.py`** -- Tool factory closures, JSON returns, errors.
- **`test_vault_integration.py`** -- Public API surface and builder wiring.

**Scenario-Specific Tests**

- **`test_general_purpose.py`** -- `GeneralPurposeScenario` registration, wiring, answer schemas.
- **`test_harness_integration.py`** -- `WorkerLoopFactory` + `AgentRuntime` + builder wiring.

### Naming Conventions

- File: `test_<module_or_concept>.py`
- Class: `TestClassName` grouping related scenarios; docstring states bug prevented
- Method: `test_<behavior_under_test>`

---

## Layer 2: Snapshot / Integration Tests (`tests/snapshot/`)

Integration tests verifying the orchestrator tool pipeline using mock agent
loops. Exercises `AgentRuntime` + `WorkerLoopFactory` + tool functions as a
connected system.

### Mock Pattern

`tests/snapshot/conftest.py` provides:

- **`FakeWorkerLoop`** -- Completes immediately with configurable result.
- **`NeverEndingLoop`** -- Stays running until cancelled (for abort tests).
- **`FakeWorkerFactory`** -- Produces `FakeWorkerLoop` instances.

These plug into `create_orchestrator_tools(runtime, factory)` so real tool
functions execute against a controllable runtime.

**Fixtures**: `runtime` (fresh `AgentRuntime`), `worker_factory` (default),
`worker_factory_with_result` (specific findings).

### Test Files

**`test_tool_pipeline.py`** -- dispatch_agent and check_tasks data flow:
- P1: `dispatch_agent` creates agent in runtime, auto-blocks for single worker, returns completed JSON with task_id/agent_id/status/result, metadata passes through. Bug: dispatch fires but runtime not updated.
- P2: `check_tasks` returns completed/failed results as JSON with counts and arrays. Bug: wrong format causes orchestrator misinterpretation.
- P2b: Terminal agents remain visible across multiple check_tasks calls. Bug: completed results disappear after first read.

**`test_error_recovery.py`** -- abort and error paths:
- P5: `abort_task` sets `AgentStatus.ABORTED` with reason, cancels asyncio.Task, returns "not found" for unknown IDs. Bug: abort succeeds but task continues (resource leak).

---

## Layer 3: Eval Tests (`tests/eval/`)

**Status**: Placeholder -- all test classes are `pytest.mark.skip`-ed.

- **`test_decision_quality.py`** -- Single orchestrator step evaluated by LLM judge (hypothesis generation, contradiction handling, etc).
- **`test_rca_scenarios.py`** -- Complete RCA runs evaluated by judge (DB pool exhaustion, false-lead scenarios).
- **`conftest.py`** -- Minimal `rca_scenario_context` fixture.

The `eval_agent.py` entry point wraps AgentM as `AgentMAgent` for rcabench-platform evaluation, running full E2E scenarios against standardized benchmarks (separate from pytest).

---

## Test Infrastructure

### Running Tests

```bash
uv run pytest tests/unit/ tests/snapshot/ -v        # CI default
uv run pytest tests/unit/test_service_profile.py -v  # Single file
uv run pytest tests/unit/test_simple_agent_loop.py::TestSimpleAgentLoopMiddleware -v
```

### Writing New Tests

**Unit tests**: Create `tests/unit/test_<module>.py`. Import module under test. Class per behavior group with bug-prevention docstring. Use `@pytest.mark.asyncio` for async.

**Snapshot tests**: Import `FakeWorkerLoop`/`NeverEndingLoop`/`FakeWorkerFactory` from `tests/snapshot/conftest.py`. Use `create_orchestrator_tools(runtime, factory)` to get real tool functions. Assert on JSON-parsed return values.

**SimpleAgentLoop tests**: Use `MockModel(responses)`, `MockTool(name, result)`, `_make_loop(...)`, `_collect_events(loop, input)` from `test_simple_agent_loop.py`.

---

## What NOT to Test

- Field existence or default values on dataclasses/TypedDicts/Pydantic models
- Enum value counts or individual member existence
- Import success (`assert X is not None`)
- Type inheritance (`isinstance` checks)
- Stub functions that only `raise NotImplementedError`
- Pydantic validation of valid inputs (only test rejection boundaries)

---

## Current Test Inventory

### `tests/unit/` (23 files)

| File | What it tests |
|------|---------------|
| `test_agent_runtime.py` | AgentRuntime lifecycle and messaging |
| `test_checkpointer_factory.py` | DebugConfig / SystemConfig schema |
| `test_cli_parse.py` | CLI argument parsing |
| `test_config_validation.py` | Config rejection boundaries |
| `test_debug_analysis.py` | Trajectory analysis CLI |
| `test_dedup.py` | DedupTracker key/eviction |
| `test_general_purpose.py` | GeneralPurposeScenario wiring |
| `test_generic_sdk.py` | Scenario registry, setup contracts |
| `test_harness_integration.py` | WorkerLoopFactory + runtime wiring |
| `test_interface_consistency.py` | Cross-module interface compat |
| `test_observability_parameterized.py` | Parameterized SQL builders |
| `test_sdk_boundary.py` | SDK-never-imports-scenarios guard |
| `test_service_profile.py` | ServiceProfileStore behavior |
| `test_simple_agent_loop.py` | SimpleAgentLoop ReAct + middleware |
| `test_tool_signatures.py` | Tool signature introspection |
| `test_trajectory_collector.py` | TrajectoryCollector JSONL |
| `test_vault_graph.py` | Vault backlinks, traverse, lint |
| `test_vault_integration.py` | Vault public API |
| `test_vault_parser.py` | Markdown + YAML parsing |
| `test_vault_schema.py` | SQLite schema DDL |
| `test_vault_search.py` | Keyword/semantic/hybrid search |
| `test_vault_store.py` | MarkdownVault CRUD |
| `test_vault_tools.py` | Vault tool factory |

### `tests/snapshot/` (3 files)

| File | What it tests |
|------|---------------|
| `conftest.py` | FakeWorkerLoop, NeverEndingLoop, FakeWorkerFactory |
| `test_tool_pipeline.py` | dispatch_agent + check_tasks data flow |
| `test_error_recovery.py` | abort_task error paths |

### `tests/eval/` (3 files, all skipped)

| File | What it tests |
|------|---------------|
| `conftest.py` | Eval scenario fixtures |
| `test_decision_quality.py` | LLM-as-Judge decision quality |
| `test_rca_scenarios.py` | E2E scenario evaluation |

---

## Related Concepts

- [Agent Harness](agent-harness.md) -- AgentRuntime, WorkerLoopFactory, SimpleAgentLoop
- [Orchestrator](orchestrator.md) -- Orchestrator tools, Scenario protocol
- [SDK Consistency](sdk-consistency.md) -- SDK/scenario separation boundary
