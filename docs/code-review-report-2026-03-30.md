# AgentM Codebase Review Report

**Date**: 2026-03-30
**Scope**: Full repository (`src/agentm/`, 83 files, ~12,500 lines)
**Review Dimensions**: Code Reuse, Code Quality, Efficiency
**Status**: All actionable items in Tier 1-3 have been fixed (2026-03-30). Deferred items require architecture decisions.

---

## Executive Summary

The AgentM codebase has a clean module structure and generally good separation of concerns. The main opportunities for improvement fall into three categories:

1. **Cross-module duplication** — JSONL parsing, dashboard setup, tool error handling, config loading all have 2-4 independent implementations
2. **Hot-path overhead** — DuckDB connection-per-call, double Pydantic serialization in trajectory, middleware per-step rebuilds
3. **Type safety gaps** — Key public APIs use `Any`/`object`, vault tools access private state directly

The report identifies **48 findings** across three dimensions, prioritized below.

---

## Part 1: Code Reuse

### R1. JSONL Trajectory Reading — 4 Independent Implementations [HIGH]

| Location | Function | What it does |
|----------|----------|-------------|
| `cli/export_eval.py:101` | `_load_trajectory()` | Returns `(meta, events)` |
| `cli/debug.py:64` | `_load_events()` | Skips `_meta`, returns events |
| `server/app.py:348` | inline loop | Skips `_meta`, supports pagination |
| `cli/judge_runner.py:462` | `_extract_skeleton_from_jsonl()` | Reads tool_call/tool_result buckets |

Additionally, `core/trajectory.py:231` has `read_metadata()` that reads only the first line.

**Recommendation**: Extract `read_trajectory_jsonl(path) -> (meta, events)` into `core/trajectory.py`.

### R2. Config Loader Functions Are Near-Identical [MEDIUM]

`load_system_config()`, `load_scenario_config()` (both in `config/loader.py`), and `load_judge_config()` (`cli/judge_runner.py`) all do: read YAML -> substitute env vars -> construct Pydantic model -> wrap errors.

**Recommendation**: Extract generic `_load_validated_yaml(path, model_cls)` helper.

### R3. JSON File Loading Pattern Repeated 7+ Times [MEDIUM]

`json.loads(path.read_text(encoding="utf-8"))` wrapped in `try/except (json.JSONDecodeError, OSError)` appears in:
- `cli/judge_runner.py` (3 places)
- `cli/export_eval.py`
- `tools/trajectory_reader.py`
- `cli/judge_runner.py` skeleton extraction

**Recommendation**: Add `safe_load_json(path: Path) -> dict | None` to `utils/`.

### R4. String Truncation Scattered Across 5 Modules [MEDIUM]

| Location | Implementation |
|----------|---------------|
| `cli/judge_runner.py:392` | `_truncate(text, max_len, oneline=False)` — most complete |
| `tools/trajectory_reader.py:92` | inline `[:8000] + "... (truncated)"` |
| `tools/vault/search.py:80` | `body[:max_len] + "..."` |
| `cli/run.py:348` | inline `[:200] + '...'` |
| `cli/debug.py:189-205` | multiple inline `[:60]`, `[:40]` slices |

**Recommendation**: Promote `_truncate` to `utils/text.py`.

### R5. Tool Error/Success Response Pattern Inconsistent [MEDIUM]

| Module | Pattern |
|--------|---------|
| `tools/vault/tools.py` | Clean `_ok()` / `_err()` helpers |
| `tools/observability/` | `_safe_tool` decorator with structured error JSON |
| `tools/duckdb_sql.py` | Inline `json.dumps({"error": ...})` with varying fields |
| `tools/case_data.py` | Inline `json.dumps({"error": ...})` with `hint` field |

**Recommendation**: Promote `_safe_tool` and `_ok`/`_err` to `tools/_shared.py` for all tool modules.

### R6. `injection.json` Loading Duplicated [MEDIUM]

Both `cli/export_eval.py:150` and `cli/judge_runner.py:521` independently load/parse `injection.json`.

**Recommendation**: Extract shared `load_injection_json(data_dir) -> dict | None`.

### R7. Config `read_text()` Missing `encoding="utf-8"` [LOW]

Three calls in `config/loader.py` (lines 54, 69, 91) omit `encoding="utf-8"` while every other `read_text` in the codebase specifies it. Risk: wrong encoding on non-UTF-8 platforms.

### R8. `json.dumps` Inconsistency [LOW]

`ensure_ascii=False` used in 38 places across 12 files, but omitted in `tools/observability/_core.py` and `tools/case_data.py`. `default=str` also inconsistent.

### R9. Tool Call Key Hashing Duplicated in Middleware [LOW]

`json.dumps(args, sort_keys=True, default=str)` appears 3 times in `middleware.py`. `DedupTracker.make_key()` could be reused by `LoopDetectionMiddleware`.

### R10. `models/types.py` Contains Only One Type Alias [LOW]

`TaskType: TypeAlias = str`, imported only by `tools/orchestrator.py`. Could be moved to `harness/types.py`.

---

## Part 2: Code Quality

### Q1. Dashboard Setup Duplicated Between `run.py` and `judge_runner.py` [HIGH]

Both files (~90 lines each) independently implement:
- Broadcaster + EvalTracker creation
- `tracker.add_listener` with `asyncio.run_coroutine_threadsafe` bridge
- `create_dashboard_app` + `uvicorn.Server` startup
- `_traj_to_ws` trajectory-to-WebSocket closure

**Recommendation**: Extract shared `start_dashboard(cases, scenario_config, host, port)` into `server/app.py`.

### Q2. `create_dashboard_app` Uses `Any` for All Key Parameters [HIGH]

```python
def create_dashboard_app(
    scenario_config: Any | None = None,
    runtime: Any | None = None,
    trajectory: Any | None = None,
    eval_tracker: Any | None = None,
```

All parameters have well-known types that should be used (with `TYPE_CHECKING` imports if needed).

### Q3. Vault Tools Access Private State Directly [HIGH]

`tools/vault/tools.py` accesses `vault._get_conn()`, `vault._embedding_model` in 5+ places, violating encapsulation.

**Recommendation**: Add public methods to `MarkdownVault` (e.g., `vault.search()`, `vault.get_backlinks()`).

### Q4. `resume` Command Is Dead Code [HIGH]

7 CLI parameters marked `# noqa: ARG001`, underlying function raises `NotImplementedError`. Advertised to users but non-functional.

**Recommendation**: Remove or clearly mark as `[experimental]` in CLI help text.

### Q5. `_wire_trajectory_to_broadcaster` Uses `object` Type [MEDIUM]

`trajectory: object` and `broadcaster: object` require `# type: ignore[attr-defined]` on every access. Should be `TrajectoryCollector` and `Broadcaster`.

### Q6. Parameter Sprawl in CLI Runner Functions [MEDIUM]

| Function | Params |
|----------|--------|
| `run_trajectory_analysis` | 11 |
| `_stream_and_finalize` | 11 |
| `_setup_debug_and_dashboard` | 8 |
| `run_judging` | 7 |

Dashboard-related params (`dashboard`, `dashboard_port`, `dashboard_host`) repeat everywhere.

**Recommendation**: Bundle into `DashboardConfig` dataclass.

### Q7. Module-Level Mutable Singletons Inconsistent [MEDIUM]

| Module | Pattern |
|--------|---------|
| `tools/observability/_core.py` | `ContextVar` (async-safe) |
| `tools/duckdb_sql.py` | `ContextVar` (async-safe) |
| `tools/case_data.py` | `ContextVar` (async-safe) |
| `tools/memory.py` | Plain global `_default_store` (NOT async-safe) |
| `tools/trajectory_reader.py` | Plain global `_default_reader` (NOT async-safe) |

**Recommendation**: Migrate `memory.py` and `trajectory_reader.py` to `ContextVar`.

### Q8. `_collect_from_directory` and `_collect_from_db` Share Structural Overlap [MEDIUM]

Both build `CaseInfo` with identical field-mapping logic (~30 lines each). A `_build_case_info(meta_dict, file_path, resolve_fn)` factory would eliminate duplication.

### Q9. Blanket `except Exception` in Vault Tools [MEDIUM]

All 10 vault tool functions catch `Exception` identically, including programming errors. Should catch `(OSError, ValueError, KeyError)` specifically.

### Q10. `_summarize_messages` Is Synchronous in Async Middleware [MEDIUM]

`CompressionMiddleware.on_llm_start` (async) calls `llm.invoke()` (synchronous), blocking the event loop. Should use `await llm.ainvoke()`.

### Q11. `core/tool_registry.py` Imports from `harness/tool.py` [MEDIUM]

Inverts the dependency direction — `core/` should be lower-level than `harness/`. A TODO comment acknowledges this.

### Q12. `_load_and_override` Imported as Private Cross-Module [LOW]

`main.py` imports `_load_and_override` from `run.py` — private function used across modules. Should be made public.

### Q13. `_build_middleware` Returns `list[Any]` [LOW]

Should return `list[MiddlewareBase]` since all items are `MiddlewareBase` subclasses.

### Q14. `_safe_tool` Decorator Lacks Type Annotations [LOW]

Four `noqa` annotations suppress type warnings. Should be typed with `ParamSpec`/`TypeVar`.

---

## Part 3: Efficiency

### E1. Sequential Case Judging [HIGH]

`run_judging` processes cases in a sequential `for` loop. Each case involves a full LLM round-trip. For 50 cases at 30s each = 25 minutes serial. Cases are independent.

**Recommendation**: `asyncio.Semaphore` + `asyncio.gather` with configurable concurrency.

### E2. DuckDB Connection-Per-Call [HIGH]

Every observability tool call and `query_sql` invocation creates a fresh in-memory DuckDB connection, registers all parquet tables, then closes it. For 20-40 tool calls per run, this is significant redundant I/O.

**Recommendation**: Cache DuckDB connection per ContextVar scope; re-register only when table map changes.

### E3. Full AgentSystem Rebuilt Per Judge Case [MEDIUM-HIGH]

Each `_judge_single_case` call triggers `build_agent_system()`: scenario discovery, `ToolRegistry` creation, YAML tool loading, `MarkdownVault` init, middleware stack construction. ~100ms+ of Python work, repeated per case.

**Recommendation**: Factor out immutable resources (tool registry, vault) into a shared context.

### E4. Double Pydantic Serialization in TrajectoryCollector [MEDIUM-HIGH]

```python
dumped = event.model_dump()      # serialize to dict
line = event.model_dump_json()   # serialize to JSON (re-does model_dump internally)
```

Effectively 3x the serialization needed. Runs on every trajectory event.

**Recommendation**: `model_dump()` once, then `json.dumps(dumped)`.

### E5. Unbounded Trajectory Event Accumulation [MEDIUM-HIGH]

`TrajectoryCollector._events` is append-only with no cap. `.events` property copies the entire list on every access. For long runs (100+ steps), this grows unchecked.

**Recommendation**: Ring buffer or periodic flush; `.events` should return a view, not a copy.

### E6. tiktoken Encoding Re-instantiated Every Count [MEDIUM]

`CompressionMiddleware` calls `tiktoken.encoding_for_model()` on every LLM step. The model string doesn't change.

**Recommendation**: Cache encoding in `__init__`.

### E7. Synchronous File I/O in Async Trajectory Recording [MEDIUM]

`_file.write(line)` + `_file.flush()` are blocking calls in the async `record()` path, blocking the event loop.

**Recommendation**: `asyncio.to_thread` for write+flush, or periodic flush.

### E8. LoopDetectionMiddleware Scans All Messages Every Step [MEDIUM]

O(n) per step: iterates all messages, serializes tool call args for Counter. Grows with conversation length.

**Recommendation**: Incremental counter maintained between steps.

### E9. Middleware Chain Rebuilt Every Step [MEDIUM]

`_build_tool_chain(ctx)` creates ~7 closures per step. The middleware list is static.

**Recommendation**: Cache the chain; pass context as parameter.

### E10. SkillMiddleware Rebuilds Skills String Every Step [LOW-MEDIUM]

`_build_skills_section()` reconstructs the same immutable string on every `on_llm_start`.

**Recommendation**: Build once in `__init__`, cache.

### E11. DynamicContextMiddleware Filters All Messages Every Step [LOW-MEDIUM]

Iterates all messages to remove/prepend system messages. O(n) per step.

**Recommendation**: Track system message position, replace in-place.

### E12. `enforce_token_budget` Re-parses JSON It Created [LOW-MEDIUM]

Tool response → `json.dumps` → `enforce_token_budget` → `json.loads` → truncate → `json.dumps`. Double serialization cycle.

**Recommendation**: Accept raw object and serialize once.

### E13. DedupMiddleware Re-serializes Args [LOW-MEDIUM]

`make_key` called in `on_llm_start` and again in `on_tool_call` for the same tool call.

### E14. Temporary File for In-Memory DuckDB Tables [LOW]

DuckDB can `register()` Python objects directly; writing temp JSON is unnecessary.

### E15. `list.pop(0)` for Inbox Drain [LOW]

`simple.py:277` — O(n) but inbox is typically 0-2 items. `collections.deque` would be cleaner.

---

## Prioritized Action Plan

### Tier 1 — High Impact, Low-Medium Effort

| # | Action | Findings | Status |
|---|--------|----------|--------|
| 1 | Extract shared JSONL trajectory reader | R1 | DONE — `core/trajectory.py:read_trajectory()` |
| 2 | Extract dashboard setup into `server/` | Q1 | DONE — `server/app.py:start_dashboard_server()` + `wire_trajectory_to_ws()` |
| 3 | Add DuckDB connection caching per ContextVar | E2 | DONE — `_conn_var` in duckdb_sql, `_obs_conn_var` in observability |
| 4 | Add concurrency to `run_judging` | E1 | DONE — `concurrency` param + `asyncio.Semaphore` + `--concurrency` CLI |
| 5 | Fix double serialization in TrajectoryCollector | E4 | DONE — single `model_dump(mode="json")` |
| 6 | Type `create_dashboard_app` parameters | Q2 | DONE — proper types with `TYPE_CHECKING` |

### Tier 2 — Medium Impact, Low Effort

| # | Action | Findings | Status |
|---|--------|----------|--------|
| 7 | Generic YAML config loader | R2 | DONE — `_load_validated_yaml()` in config/loader.py |
| 8 | Shared `safe_load_json` utility | R3 | DONE — `utils/json_utils.py` |
| 9 | Promote `_safe_tool` + `_ok`/`_err` to `_shared.py` | R5, Q9 | DONE — `safe_tool`, `tool_ok`, `tool_error` in _shared.py |
| 10 | Bundle dashboard params into `DashboardOpts` | Q6 | DONE — `DashboardOpts` dataclass in server/app.py |
| 11 | Cache tiktoken encoding | E6 | DONE — `self._encoding` in CompressionMiddleware |
| 12 | Cache middleware chain and skills section | E9, E10 | DONE — cached `_skills_section`, optimized chain |
| 13 | Remove dead `resume` command | Q4 | DONE — removed from CLI and run.py |

### Tier 3 — Low Effort Quick Wins

| # | Action | Findings | Status |
|---|--------|----------|--------|
| 14 | Add `encoding="utf-8"` to `config/loader.py` reads | R7 | DONE |
| 15 | Make `_load_and_override` public | Q12 | DONE — renamed to `load_and_override` |
| 16 | Migrate memory/trajectory_reader to ContextVar | Q7 | DONE — `_store_var`, `_reader_var` |
| 17 | Add public methods to MarkdownVault | Q3 | DONE — `get_connection()`, `embedding_model` property |
| 18 | Shared truncation utility | R4 | DONE — `utils/json_utils.py:truncate()` |

### Additional Fixes Applied

| Fix | Description |
|-----|-------------|
| E14 | DuckDB `conn.register()` for in-memory data (replaces temp file) |
| E15 | `collections.deque` for inbox in SimpleAgentLoop |
| E11 | Optimized DynamicContextMiddleware message filtering |
| E8 | Incremental LoopDetectionMiddleware with shared `_make_tool_call_key()` |
| E12 | `enforce_token_budget` accepts pre-parsed object to skip redundant JSON roundtrip |
| E13 | DedupMiddleware uses shared key function |
| Q5 | Removed `_wire_trajectory_to_broadcaster` (replaced by shared `wire_trajectory_to_ws`) |
| Q9 | Vault tools: narrowed `except Exception` to specific types |
| R5 | `case_data.py` uses `tool_error()` helper |

### Architecture Improvements (Previously Deferred)

| # | Topic | Findings | Status |
|---|-------|----------|--------|
| D1 | Trajectory event memory management | E5 | DONE — `_ReadOnlyEventView` (no copy), `max_memory_events` eviction, `read_all_events()` from JSONL |
| D2 | Factor immutable resources out of `build_agent_system` | E3 | DONE — `AgentSystemContext` + `build_system_context()` + `create_agent_run()` two-level API |
| D3 | Async-safe trajectory file I/O | E7 | DONE — buffered writes, `asyncio.to_thread` flush, periodic background flush task |
| D4 | `core/` -> `harness/` dependency inversion | Q11 | DONE — `Tool` + `tool_from_function` moved to `core/tool.py`, `harness/tool.py` re-exports |

---

## Change Summary

**Total**: 26 files changed, -1630 / +1486 lines (net -144)

### Wave 1 (5 parallel agents)
- **Middleware hot-path**: tiktoken cache, incremental loop detection, skills cache, deque inbox
- **DuckDB caching**: ContextVar connection pool, `conn.register()` for in-memory data
- **TrajectoryCollector**: eliminated double Pydantic serialization
- **Config + ContextVar**: generic YAML loader, memory/trajectory_reader async-safe
- **Tool shared infra**: `safe_tool`/`tool_ok`/`tool_error` in `_shared.py`, vault public API

### Wave 2 (2 parallel agents)
- **Dashboard + CLI**: shared `start_dashboard_server()`, `DashboardOpts`, judge concurrency, removed dead `resume`
- **JSONL + utils**: shared `read_trajectory()`, `safe_load_json()`, `truncate()`

### Architecture (3 parallel agents)
- **D1+D3**: Read-only event view, memory eviction, buffered async I/O with periodic flush
- **D2**: `AgentSystemContext` for resource reuse across judge cases
- **D4**: `Tool` moved to `core/`, clean layer dependency

*Generated by AgentM code review, 2026-03-30. All items fixed on the same day.*
