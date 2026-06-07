# Dynamic Workflow

Status: implemented (branch `workflow-python`)
Owner: builtin atom `workflow`; reuses `spawn_child_session`, `artifact_store`,
`operations_agent_env`, the event bus.

Reaches into: `extensions/builtin/workflow.py`, `ExtensionAPI.spawn_child_session`,
`artifact_store` service, event bus (`emit(channel, event)` generic overload).
One *potential* substrate touch flagged in §Out of scope.

## What it is

A **dynamic workflow** is an LLM-authored **async Python** script that
orchestrates many child agent sessions deterministically. The model writes the
control flow (loops, branches, fan-out, budget-driven scaling) as code at
runtime; the atom runs it as a single tool call; only the final result returns
to the model's context. This is the inverse of the normal agent loop, where the
model decides the next step turn by turn.

What AgentM already had vs. what is new:

- **Already shipped** — *pre-written* deterministic orchestrators. `tool_eval_run`
  loops `for task: for sample: spawn_child_session(...)` → drive → grade →
  aggregate. Trusted atom code, passes §11.
- **New here** — the model *authoring the orchestration at runtime* and running
  it as one tool call.

This mirrors Claude Code's "Workflow tool": a workflow is a tool the same way
`Read` or `Bash` is. Consistent with [agent_team](agent-team.md): **no `Workflow`
SDK type, no mode switch** — `workflow` is one more always-available tool the
LLM dispatches when the task warrants deterministic fan-out. It composes with
the model-driven path (`sub_agent`): a worker can itself dispatch sub-agents;
the model can run a workflow, read the result, then decide the next one
(deterministic within a phase, adaptive between phases).

## Why Python in-process (and why we reversed the original JS decision)

The first design ran the script in a **QuickJS isolate** on the premise that a
model-authored script is "untrusted" and must be capability-sandboxed. We
reversed that. The premise does not hold:

- **The script runs at the agent's own authority.** The agent that wrote it can
  already call `bash`, `tool_write`, and `install_atom` directly. Capability-
  gating the orchestration script guards a threat the agent isn't subject to —
  if it wanted to do harm it would use `bash`, not smuggle it through a workflow.
- **AgentM already runs LLM-authored Python in-process.** `install_atom` /
  `reload_atom` load model-written module source into the live process
  (`self_modifiable_architecture`). Refusing to run a tamer orchestration script
  in Python "because untrusted Python is unsafe" is self-inconsistent.
- **JS actively cost us native async.** A QuickJS isolate is single-threaded and
  the host bridge had to block a worker thread on each `agent()` call, which made
  a real `pipeline` (interleaving stages across items) impossible. Python with
  native `asyncio` makes `pipeline` trivial.

So the script is plain `async` Python, run as a coroutine **on the host event
loop** in a **curated namespace**. The namespace is a *guardrail, not a
sandbox*: `__builtins__` is data-ops only (no `open`/`__import__`/`eval`), and
`time`/`random` are simply **not injected** (keeping the resume key
deterministic). It is not escape-proof — hard isolation, when actually needed
(multi-tenant), is the worker `operations_agent_env` sandbox or a process
boundary, the same answer as for `bash`.

Two non-overlapping concerns: the **curated namespace** shapes the orchestration
script; **`operations_agent_env`** confines a *worker's* bash/file ops.

## Primitive → existing-API mapping

| Primitive | Built from | Status |
|---|---|---|
| `agent(prompt, *, scenario=, isolation=, tool_allowlist=)` → str | `spawn_child_session(cfg)` → `child.prompt(msg)` → last `AssistantMessage.text` → `child.shutdown()` | reuse (same recipe as `tool_eval_run` + `sub_agent`) |
| `parallel(aws)` → list | `asyncio.gather`; each `agent` self-limits via a shared `Semaphore` | reuse |
| `pipeline(items, *stages)` → list | per-item `_chain` (await each stage, sync or async) under `asyncio.gather`; **no cross-item barrier** | **implemented** (native async) |
| `budget` (`.total`/`.spent()`/`.remaining()`) | `_BudgetService` summing child `TurnEndEvent.message.usage`; ceiling from `budget_tokens` config | new aggregation |
| `args` (dict) / `log(msg)` / `phase(name)` | caller payload / fire-and-forget `WorkflowPhaseEvent` on the parent bus | reuse |
| worker isolation (`isolation="agent_env"`) | include `operations_agent_env` in the child `extensions`, guarded by a runtime `list_atoms()` check | reuse + soft-dep guard |

`agent()` takes **real kwargs** (not a JS opts object); `parallel`/`pipeline`
take **real awaitables/callables**, because it is native Python — a direct win
over the JS bridge.

## Worker session model (option A — slim, mirror the orchestrator)

A worker spawned by `agent()` defaults its `scenario` to the **orchestrator's
own** (`api.scenario`) — a clean reload of that curated atom set. Precedence:
explicit `scenario=` arg > atom `default_scenario` config > parent scenario.

Why not `scenario=None`: that auto-discovers **every builtin**, handing each
worker the full heavy toolset (`bash`/`install_atom`/…), the `workflow` tool
itself (recursion), and — on strict providers (Ark) — a tool schema set that
breaks the provider's tool-call grammar generation. Mirroring the orchestrator's
scenario keeps workers slim and appropriate. (Verified by E2E: naive
`agent("...")` on doubao failed with all-builtins, works once workers load the
slim `local` scenario.)

**Anti-recursion** is enforced two ways: workers are stamped
`purpose="workflow"`, and the atom's `install()` **skips registering the
workflow tool** whenever `api.purpose == "workflow"` — so even a worker that
auto-discovers builtins never gets a recursive workflow tool.

**Do not use `tool_allowlist=[]` to slim workers.** `tool_allowlist` filters the
tool list after extensions install, which can starve an extension that requires
its tools at install time (e.g. `file_mutation_queue` requires `edit`/`write`).
Slimming is the scenario's job, not the allowlist's.

## Net-new components (small)

1. **`workflow.py` atom + curated-namespace runner.** `register_tool("workflow")`
   taking the model-authored script + `args`; compiles it as the body of an
   `async def` and awaits it on the host loop with the SDK injected. Concurrency
   cap `min(16, cpu-2)` via `Semaphore`; 1000-agent lifetime backstop. No JS
   engine, no marshalling, no worker thread.
2. **`_BudgetService` (aggregator).** `cost_budget` gives the usage recipe
   (`TurnEndEvent.message.usage`) but keys spend per-install; there is no
   cross-child aggregation, and child `TurnEndEvent`s do **not** bubble to the
   parent bus, so the service subscribes on **each child's own bus** and sums.
   `budget.total` comes from `budget_tokens` config; `remaining()` derives.
3. **Workflow-local journal for resume.** *Not* `SessionStore.open` (that is
   session-transcript replay, wrong granularity). Each `agent()` result is
   written to `artifact_store` keyed by `hash(prompt, opts)`; on resume the host
   returns the cached body without re-spawning. Determinism comes from *not
   injecting* `time`/`random`, not from interception.

## Quality patterns (library, not mechanism)

Harnesses composed from the primitives, selected per task: adversarial verify
(N skeptics, majority-refute kills it), perspective-diverse verify, judge panel,
loop-until-dry (dedup against `seen`, never `confirmed`), multi-modal sweep,
completeness critic, no-silent-caps (`log()` any truncation). Documented for the
model in the tool's reference surface, not encoded in the runtime.

## §11 / boundary

- `workflow.py` is a single-file atom: stdlib + `agentm.core.abi.*` +
  `agentm.extensions.*`. **No atom-to-atom imports** — `artifact_store` via
  `api.get_service`, workers via `api.spawn_child_session`.
- The curated namespace is handed nothing from `core.runtime.*`; the SDK closures
  close over `api` but the script only sees the injected names + safe builtins.
- Worker isolation (`operations_agent_env`) is selected by composition (listing
  the atom in the child's `extensions`), not a privileged config field.
- **`operations_agent_env` is a *soft* dependency, by design.** NOT in
  `MANIFEST.requires` (that would force every workflow user to pull the agent-env
  extra + a K8s gateway). Enforcement is a **runtime availability guard**
  (`list_atoms()` → clear error if the script asks for agent-env isolation but
  the atom is not loaded). The atom stores only the dotted module path and
  **derives** the bare atom name (`rsplit`), because a bare-name literal would
  trip the §11.4.D4 peer-literal check and read as an *undeclared hard*
  dependency. Recorded here so the pattern is reviewable design, not a buried
  trick. A first-class `MANIFEST.optional_requires` would let the soft intent be
  *declared*; that is a substrate change, deferred.
- **Cross-child budget aggregation reaches `child.bus` directly** (the same
  contract `sub_agent` relies on). `spawn_child_session` is typed `-> Any`, so
  `.bus` is a duck-typed reach; we access it directly (no defensive
  `getattr`/swallow) so a broken child contract fails loudly rather than silently
  zeroing the budget ceiling.

## Decisions / rejected

- **LLM authors the script at runtime** (vs. only pre-written orchestrator
  atoms) — chosen; otherwise it is not *dynamic*.
- **Python in-process, curated namespace** (vs. a QuickJS isolate) — chosen,
  reversing the original JS decision: the isolate guarded a threat model that
  does not exist (script runs at the agent's authority; `install_atom` already
  runs LLM Python) and cost native async / `pipeline`.
- **Slim workers mirroring the orchestrator's scenario** (option A) — chosen over
  inheriting all builtins; with anti-recursion via the `purpose` marker.
- **Workflow-local journal** (vs. `SessionStore.open`) — chosen; correct
  granularity, no substrate dependency.
- **No `Workflow` SDK type, no mode switch** — consistent with `agent_team`.

## Out of scope / flagged

- **Ark guided-decoding quirk.** On Ark/Volcengine, a *bloated* worker toolset
  (all builtins) makes the provider's tool-call grammar converter reject the
  schema set (`minItems > prefixItems`). No AgentM tool ships such a schema — it
  is Ark-internal, triggered by request shape. Mitigated by option A (slim
  workers). litellm / deepseek-official / anthropic providers are unaffected.
- **`schema` (structured output) on `agent()`** — deferred; needs a per-child
  forced-tool mechanism not in the reused recipe. Would map to
  `pydantic_to_openai_tool_schema`.
- **Per-child env overrides** — `AgentSessionConfig` has no per-child env field
  (the old `os.environ` route is racy under parallel runs). The one change that
  would touch the substrate; deferred until a real need.
- **Multi-tenant hard isolation** of a shared gateway running other users'
  scripts — out of this layer; reuse `operations_agent_env` / a process
  boundary, the same answer as for `bash`.
- Cross-session (non-tree) resume, nested workflows beyond one level, binary
  artifacts — deferred.
