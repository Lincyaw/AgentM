# Session Creation API Reference

Read this when spawning child sessions, booting sessions from host code,
or choosing between the session creation APIs.

## Decision table

Your position in the architecture determines which API you use.

| You are writing… | API to use | Can import `core.runtime`? |
|------------------|------------|----------------------------|
| An **atom** (builtin or contrib extension) | `api.spawn()` or `api.spawn_child_session()` | No — atom contract forbids it |
| An **embedder / host program** (CLI, gateway, eval harness) | `AgentSession.create()` | Yes — you are outside atom scope |

---

## `api.spawn()` — lightweight child (for atoms)

Creates a child that inherits everything from the parent. Override only
what differs.

```python
child = await api.spawn(
    purpose="audit",
    tools=[my_tool],
    model=Model(id="gpt-4o"),
    max_turns=5,
)
result = await child.run("Do the thing.")
await child.shutdown()
```

Automatically wires up:
- **Trace inheritance** — child shares the parent's trace
- **Provider inheritance** — child reuses the parent's LLM config
- **Tool/policy inheritance** — carries over unless overridden

---

## `api.spawn_child_session()` — full config child (for atoms)

Use when the child needs a different scenario or extension set from the
parent.

```python
from agentm.core.abi import AgentSessionConfig, LoopConfig

child = await api.spawn_child_session(AgentSessionConfig(
    cwd=api.ctx.cwd,
    scenario="worker_scenario",
    purpose="audit",
    loop_config=LoopConfig(max_tool_calls=20),
))
result = await child.run("Do the thing.")
await child.shutdown()
```

`AgentSessionConfig` lives in `agentm.core.abi.session_api` (not
`core.runtime`) so atoms can construct configs without violating the
atom import rules.

---

## `AgentSession.create()` — for host programs

Host programs (CLI, gateway, eval harnesses, anything that boots a
session from scratch) use this. It cold-starts a complete session:
creates EventBus, discovers and installs all atoms, wires the full
runtime. Must supply a provider explicitly — no parent to inherit from.

```python
from agentm import AgentSession, AgentSessionConfig, LoopConfig

session = await AgentSession.create(AgentSessionConfig(
    cwd=str(work_dir),
    scenario="chat",
    provider=("agentm.extensions.builtin.llm_openai", {"model": "gpt-4o"}),
    loop_config=LoopConfig(max_tool_calls=15),
))
messages = await session.run("summarize src/agentm/core/abi")
await session.shutdown()
```

The `AgentSession` class lives in `agentm.sdk` (re-exported from the
top-level `agentm` package). It extends `Session` from
`core.runtime.session` — importing it from `agentm.sdk` is safe even
in atom-adjacent code, but atoms should still prefer `api.spawn()`.

---

## Threading with async sessions

If your orchestrator uses `ThreadPoolExecutor`, wrap the async session
in `asyncio.run()` per thread:

```python
def _run_one(args) -> dict:
    return asyncio.run(_async_run(args))

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(_run_one, a) for a in task_args]
```

---

## Common mistakes

- **Using `AgentSession.create()` inside an atom** — Atoms must use
  `api.spawn()` or `api.spawn_child_session()`. The spawn APIs give
  you trace and provider inheritance for free.
- **Importing `AgentSessionConfig` from `core.runtime`** — Import it
  from `agentm.core.abi.session_api`. It lives in ABI so atoms can
  use it without violating the atom import rules.
- **Shelling out to CLI from Python** — Use `AgentSession.create()`.
