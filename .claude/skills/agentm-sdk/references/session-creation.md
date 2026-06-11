# Session Creation API Reference

Read this when spawning child sessions, booting sessions from host code,
or choosing between the session creation APIs.

## Decision table

Your position in the architecture determines which API you use.

| You are writing… | API to use | Can import `core.runtime`? |
|------------------|------------|----------------------------|
| An **atom** (builtin or contrib extension) | `api.spawn_child_session()` | No — atom contract forbids it |
| A **workflow script** (`agent()`) | `agent()` from workflow engine | No — it wraps `spawn_child_session` |
| An **embedder / host program** (CLI, gateway, eval harness) | `AgentSession.create()` | Yes — you are outside atom scope |
| **Framework internals** (session factory, custom session subclass) | `create_agent_session()` | Yes — internal only |

---

## `api.spawn_child_session()` — for atoms

Atoms create child sessions through their `ExtensionAPI` handle. The
runtime automatically wires up:
- **Trace inheritance** — child shares the parent's OTel trace_id
- **Provider inheritance** — child reuses the parent's LLM config
- **Extension hook** — parent atoms can inject extensions into the child
  via `ChildSessionExtendingEvent`

```python
# Inside an atom's tool handler or install()
from agentm.core.abi import AgentSessionConfig  # ABI import, not runtime

child = await api.spawn_child_session(AgentSessionConfig(
    cwd=api.cwd,
    scenario="worker_scenario",
    purpose="audit",
    loop_config=LoopConfig(max_tool_calls=20),
))
result = await child.prompt("Do the thing.")
await child.shutdown()
```

`AgentSessionConfig` lives in `agentm.core.abi.session_config` (not
`core.runtime`) so atoms can construct configs without violating the
atom import rules.

---

## `AgentSession.create()` — for host programs

Host programs (CLI, gateway, eval harnesses, anything that boots a
session from scratch) use this. It cold-starts a complete session:
creates EventBus, discovers and installs all atoms, wires the full
runtime. Must supply a provider explicitly — no parent to inherit from.

```python
from agentm.core.runtime.session import AgentSession
from agentm.core.abi.session_config import AgentSessionConfig, LoopConfig

session = await AgentSession.create(AgentSessionConfig(
    cwd=str(work_dir),
    scenario="verifier_hop",
    provider=("openai", {"model": "gpt-4o", "api_key": "..."}),
    loop_config=LoopConfig(max_tool_calls=15),
    atom_config_overrides={
        "hop_finalize": {"data_dir": str(data_dir)},
    },
))
messages = await session.prompt(prompt_text)
await session.shutdown()
```

This import (`agentm.core.runtime.session`) is forbidden inside atoms
but legitimate in host-level code.

---

## `create_agent_session()` — framework internal

```python
from agentm.core.runtime.session_factory import create_agent_session
session = await create_agent_session(AgentSession, config)
```

`AgentSession.create()` calls this internally, passing `cls` so
subclasses work. Direct use is rare — only for custom session classes
or low-level drivers (e.g., replay engine). If you reach for this, you
are probably writing framework infrastructure, not application code.

---

## Workflow `agent()` — the recommended multi-agent path

For multi-agent orchestration, workflow scripts provide `agent()` which
wraps `spawn_child_session` with journal, budget tracking, concurrency
management, and auto-parsed structured output. Prefer this over manual
`spawn_child_session` + `prompt()` + `shutdown()` when coordinating
more than one child.

```python
# In a workflow script
result = await agent(
    "Verify this edge.",
    scenario="verifier/hop",
    atom_config={"hop_context": {...}},
)
```

---

## Why SDK over subprocess

Never shell out to `agentm -p` from Python. Use `AgentSession.create()`
(host programs) or workflow `agent()` (orchestration).

| `subprocess.run("agentm -p ...")` | SDK API |
|-----------------------------------|---------|
| Build CLI args, manage fallbacks | Direct Python call |
| Stuff env vars for config | Pass via `atom_config_overrides` |
| Parse obs JSONL to extract results | Read from returned messages |
| Manage stdout/stderr log files | Events/logging handled by session |
| Spawns a new process per invocation | In-process, lighter |

## Threading with async sessions

If your orchestrator uses `ThreadPoolExecutor`, wrap the async session
in `asyncio.run()` per thread:

```python
def _run_one_hop(args) -> dict:
    return asyncio.run(_async_run_hop(args))

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(_run_one_hop, a) for a in hop_args]
```

---

## Common mistakes

- **Using `AgentSession.create()` inside an atom** — Atoms must use
  `api.spawn_child_session()`. `AgentSession.create` requires importing
  `core.runtime` which the atom contract forbids. The spawn API also
  gives you trace and provider inheritance for free.
- **Using `create_agent_session()` directly** — Use
  `AgentSession.create()` unless you need a custom session subclass.
- **Importing `AgentSessionConfig` from `core.runtime`** — Import it
  from `agentm.core.abi.session_config`. It lives in ABI so atoms can
  use it without violating the atom import rules.
- **Shelling out to `agentm -p` from Python** — Use
  `AgentSession.create()` or workflow `agent()`.
