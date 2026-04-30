# Task: Phase 2 Group C — Sub-Agent Extension

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §7 (last row)
**Architecture**: [pluggable-architecture.md](../designs/pluggable-architecture.md) §6 (acceptance scenario 7)
**Agent**: implementer (sonnet) — solo, this is the largest single extension
**Status**: READY (depends on Phase 2.0 — `ChildSessionStartEvent`/`ChildSessionEndEvent` already wired into `AgentSession.create`)

## Why this proves a big architectural claim

Per `pluggable-architecture.md` §6 acceptance scenario 7: **"Add a sub-agent system: an extension that registers a `dispatch_agent` tool whose `execute` spawns nested `AgentSession` instances. Core never learns about sub-agents."**

If this extension works, "multi-agent is core-orthogonal" is a proven property of AgentM. The existing AgentRuntime / TaskManager logic in `src/agentm/harness/runtime.py` (~hundreds of lines) collapses into one extension that **uses** the kernel + harness primitives without any new core support.

## Scope

ONE extension at `src/agentm/extensions/builtin/sub_agent.py`. It registers four tools and manages an internal pool of child `AgentSession` tasks.

### Tools to register

| Tool | Purpose | Args |
|---|---|---|
| `dispatch_agent` | Spawn a child agent. Returns immediately with a `task_id`. The child runs as an `asyncio.Task` consuming a child `AgentSession`. | `{"purpose": str, "prompt": str, "extensions": list[tuple[str, dict]] (optional)}` |
| `check_tasks` | List active + completed child tasks with status. | `{}` |
| `inject_instruction` | Push a new user message into a running child's session. **v0 semantics (locked, design §10b.5)**: append to a per-child queue; queue is drained when the current `prompt(...)` resolves (i.e. the next user-turn starts with the queued message concatenated). True streaming-mid-turn injection is deferred. | `{"task_id": str, "message": str}` |
| `abort_task` | Set the child's abort signal; emit `child_session_end(error="aborted")`. | `{"task_id": str}` |

### State management

The extension owns a private `dict[str, _ChildTask]` keyed by `task_id`. Each `_ChildTask` holds:
- the asyncio.Task
- the child `AgentSession`
- an asyncio.Event abort signal
- a status enum: `running | completed | aborted | error`
- the final messages list (once completed)
- a queue of pending `inject_instruction` payloads

Lifetime: from `dispatch_agent` until the child task is `done()`. Don't let entries leak forever; on `agent_end` of the parent session, await all pending children with a 5s grace period, then abort outstanding ones. Subscribe `session_shutdown` for this cleanup.

### Child session construction

When `dispatch_agent` is invoked:
1. Build a child `AgentSessionConfig`:
   - `cwd` = parent's cwd
   - `extensions` = caller-supplied + parent's extensions filtered to "safe to inherit" (config option `inherit_extensions: list[str]` defaults to `["permission", "dedup", "trajectory"]`)
   - `provider` = parent's active provider (re-used; same StreamFn instance is fine, it has no per-session state)
   - `parent_bus` = parent's bus
   - `parent_session_id` = parent's `session_id`
   - `purpose` = caller-supplied `purpose` arg
2. `child = await AgentSession.create(child_config)`
3. Wrap `child.prompt(initial_prompt)` in an `asyncio.Task`; store and return `task_id`.

Per Phase 2.0 wiring, `child.create` will fire `ChildSessionStartEvent` on the parent bus automatically. Don't fire it manually.

On task completion, `child.shutdown()` fires `ChildSessionEndEvent`. Verify by test.

### Concurrency safety

Multiple `dispatch_agent` calls from the same parent must be safe. Use `asyncio.Lock` for the registry mutations if needed. Each child task runs independently; their tool executions are isolated.

Hard limit: `config["max_workers"]` (default 4). If exceeded, `dispatch_agent` returns a `ToolResult(is_error=True, ...)` with a clear message rather than blocking the parent.

## Tests

`tests/unit/extensions/builtin/sub_agent/test_sub_agent.py`:

1. **Smoke**: parent session with `sub_agent` extension. Parent's prompt invokes `dispatch_agent` with prompt "x". Child runs to completion. Parent calls `check_tasks` and gets the child's status. Final assertion: parent saw `child_session_start` and `child_session_end` on its bus.
2. **Inject**: `dispatch_agent` then `inject_instruction` — child sees the new message in its second turn.
3. **Abort**: dispatch a child whose StreamFn is slow (`asyncio.sleep`); call `abort_task`; verify the child terminates with `error="aborted"` and `ChildSessionEndEvent` fires with that error.
4. **Max workers**: dispatch 5 with `max_workers=4`; the 5th returns an error tool result.
5. **Cleanup on parent shutdown**: dispatch a long-running child; call `parent.shutdown()`; verify the child is aborted within the grace period.
6. **StreamFn statelessness assertion (design §10b.5)**: dispatch 2 children concurrently sharing the parent's StreamFn instance; both run prompts that invoke a tool and complete normally. Assert no cross-talk in the recorded `messages` lists between the two children — this proves the same StreamFn closure is safe across concurrent sessions.

Use the `fake_provider` pattern from `tests/unit/harness_v2/_fixtures/fake_provider.py` adapted for child agents.

## HARD constraints

- Same imports rule as Group A/B. The extension is allowed to construct child `AgentSession` via `agentm.harness.session.AgentSession` — that's the whole point.
- No `agentm.harness.runtime` (legacy AgentRuntime). No `agentm.harness.worker_factory`. No `agentm.tools.orchestrator` (legacy multi-agent tools).
- No global state. All state on the extension instance, captured in closures by `install`.

## Quality gates

```bash
uv run ruff check src/agentm/extensions/builtin/sub_agent.py tests/unit/extensions/builtin/sub_agent/
uv run mypy src/agentm/extensions/builtin/sub_agent.py
uv run pytest tests/unit/extensions/builtin/sub_agent/ tests/unit/kernel/ tests/unit/harness_v2/ tests/unit/llm/ -q
```

All tests pass. No flakes from asyncio races — if you see a flake, fix the race, don't add `time.sleep`s.

## Reference for behavior parity

Legacy implementations (READ-ONLY):
- `src/agentm/harness/runtime.py` (AgentRuntime — multi-agent lifecycle)
- `src/agentm/harness/worker_factory.py`
- `src/agentm/tools/orchestrator.py` (the legacy dispatch tools)
- `src/agentm/harness/handle.py`

Capture only the behavioral intent; the new implementation should be ~30% the size of the legacy because the EventBus + AgentSession do most of the lifecycle work for you.

## Report format (≤350 words)

1. File created (one source + tests).
2. Tools registered: list with one-line spec each.
3. Test counts; specifically does the abort test reliably terminate without time.sleep workarounds?
4. Concurrency model: how do you serialize registry mutations? `asyncio.Lock` or single-task assumption?
5. Inheritance config (`inherit_extensions`): defaults you picked.
6. Cleanup on parent shutdown: grace period behavior.
7. One sentence on why this extension is structurally so much smaller than legacy `AgentRuntime`.
