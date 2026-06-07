# Dynamic Workflow

Status: design (not yet implemented)
Owner: new builtin atom `workflow` + small services; reuses `sub_agent`,
`cost_budget`, `artifact_store`, `operations_agent_env`.

Reaches into: `extensions/builtin/workflow.py` (new), `ExtensionAPI.spawn_child_session`
(existing, reused), `artifact_store` service (reused), event bus (`emit(channel,
event)` generic overload, reused). One *potential* substrate touch flagged in §7.

## What it is

A **dynamic workflow** is an LLM-authored JavaScript script that orchestrates
many child agent sessions deterministically. The model writes the control flow
(loops, branches, fan-out, budget-driven scaling) as code at runtime; a runtime
executes it; only the final result returns to the model's context. This is the
inverse of the normal agent loop, where the model decides the next step turn by
turn.

The distinction AgentM already embodies vs. what is new:

- **Already shipped** — *pre-written* deterministic orchestrators. `tool_eval_run`
  loops `for task: for sample: spawn_child_session(...)` → drive → grade →
  aggregate. Trusted atom code, passes §11.
- **New here** — the model *authoring the orchestration at runtime* and running
  it as a single tool call. Because the script is model-generated it is
  **untrusted**, which is why a capability-gated sandbox is mandatory.

This mirrors Claude Code's "Workflow tool": a workflow is a tool the same way
`Read` or `Bash` is. Consistent with [agent_team](agent-team.md): **no `Workflow`
SDK type, no mode switch** — `workflow` is one more always-available tool the
LLM dispatches when the task warrants deterministic fan-out.

## Why JS, in-process

The script's only capability must be `agent()`; it must not reach fs / net /
shell (the orchestration plan, not the work, lives here). Once the script is
untrusted, language and isolation are bound together:

- **In-process untrusted Python is unsafe** — `RestrictedPython`/`asteval`
  `__builtins__` escapes are known-breakable, and this code runs inside the
  gateway process that holds *all* sessions. Rejected.
- **A JS isolate (quickjs) is itself a capability sandbox** — nothing is
  reachable unless the host injects it. Inject `agent / parallel / pipeline /
  log / phase / budget / args` as host functions; remove `Date.now` /
  `Math.random` / argless `new Date()` (they break replay determinism). The
  script is capability-gated without an OS-level pod.
- `agent()` is an **in-process callback** straight into
  `api.spawn_child_session` — no RPC, which fits the single-process gateway that
  already holds every session.

Two isolation layers, each its own shape: the **JS isolate** confines the
orchestration script; **`operations_agent_env`** (existing) continues to confine
the *worker* child sessions' bash/file ops. They do not overlap.

JS engine: `quickjs` (light, pure sandbox, sufficient). Bundled as an **optional
extra**, the same treatment as `arl-env` for `operations_agent_env`.

## Primitive → existing-API mapping

| Primitive | Built from | Status |
|---|---|---|
| `agent(prompt, opts)` | `spawn_child_session(cfg)` → `child.prompt(msg)` → last `AssistantMessage.text` → `child.shutdown()` | reuse (identical in `tool_eval_run` + `sub_agent`) |
| `parallel(specs)` | `asyncio.gather` + `Semaphore` + `api.track_background()` bracket | reuse |
| `pipeline(items, ...stages)` (no barrier) | per-item `asyncio.Task` over the same bracket | **deferred** — not in v1 (see §Out of scope) |
| `opts.schema` (structured output) | `agentm.core.lib.pydantic_to_openai_tool_schema` + forced StructuredOutput tool on child | reuse |
| `opts.isolation` (worker sandbox) | include `operations_agent_env` in child `extensions` list, guarded by a runtime `list_atoms()` availability check | reuse + soft-dep guard (see §11/boundary) |
| events → TUI | `api.events.emit("workflow_phase", WorkflowPhaseEvent(...))`, event type defined in the atom | reuse (`emit(channel: str, event: Any)` generic overload; OTLP/observer captures generically) |
| child lifecycle / background / inbox | `sub_agent`'s `_run_child` / `track_background` / `post_inbox` | reuse |

## Net-new components (three, all small)

1. **`workflow.py` atom + quickjs host.** `register_tool("workflow")` taking the
   model-authored script + `args`. The host binds the primitives above to the
   already-existing APIs and strips nondeterministic JS globals. The binding
   layer is thin; the heavy lifting is in the reused APIs. Concurrency cap =
   `min(16, cpu-2)` via a `Semaphore`; lifetime agent-count backstop (1000) as a
   runaway guard.

2. **`BudgetService` (aggregator).** `cost_budget` gives the usage recipe
   (`TurnEndEvent.message.usage` in/out tokens) but keys spend in per-install
   local `state` — there is **no cross-child aggregation**. Correction (found
   during implementation, contra the first draft of this section): child
   `TurnEndEvent`s do **not** bubble to the parent bus, so the service
   subscribes directly on **each child's own bus** right after spawn and sums
   usage there. `budget.total` is a configured token ceiling (`budget_tokens`
   atom config); `budget.remaining()` derives from it. Reuses the usage signal;
   new aggregation + per-child subscription.

3. **Workflow-local journal for resume.** *Not* `SessionStore.open` — that is
   session-level transcript replay, the wrong granularity. Claude Code's "journal
   every `agent()` call" is a **workflow-local journal**: each `agent()` result
   is written to the `artifact_store` service keyed by `hash(prompt, opts)`; on
   resume the host checks the journal first and returns the cached result,
   running only new/changed calls live. Stripping every reachable wall-clock /
   entropy source (`Date.now`, `Math.random`, argless `new Date()`,
   `performance.now`) — inside a bootstrap IIFE so the original `Date` is never
   re-exposed as a global — keeps the key sound. Reuses `artifact_store`
   (`write_artifact` / `read`); new is the hash-keyed wrapper.

## Quality patterns (library, not mechanism)

The value is the harnesses composed from the primitives, selected per task:
adversarial verify (N skeptics per finding, majority-refute kills it),
perspective-diverse verify (distinct lenses), judge panel, loop-until-dry
(K dry rounds before stopping; dedup against `seen`, never against `confirmed`),
multi-modal sweep, completeness critic, no-silent-caps (`log()` any truncation).
These are documented for the model in the tool's reference surface, not encoded
in the runtime.

## §11 / boundary

- `workflow.py` is a single-file atom: stdlib + `agentm.core.abi.*` +
  `agentm.extensions.*` + `quickjs` (optional 3rd-party). **No atom-to-atom
  imports** — it reaches `sub_agent` / `artifact_store` capability via
  `api.get_service` and `api.spawn_child_session`, never by importing the module.
- The quickjs sandbox is never handed anything from `core.runtime.*`; host
  functions close over `api` but expose only the six primitives to JS.
- Worker isolation policy (`operations_agent_env`) is selected by listing the
  atom in the child's `extensions`, not by a privileged config field.
- **`operations_agent_env` is a *soft* dependency, by design.** It is NOT in
  `MANIFEST.requires` — that would force every workflow user to pull the
  agent-env extra + a K8s gateway, when most workflows never request
  `isolation: "agent_env"`. Enforcement is therefore a **runtime
  availability guard** (`list_atoms()` → clear error if the script asks for
  agent-env isolation but the atom is not loaded), not the §11 validator. The
  atom stores only the dotted module path and **derives** the bare atom name at
  runtime (`rsplit`), because a bare-name string literal would trip the
  §11.4.D4 peer-literal check and read as an *undeclared hard* dependency. This
  derive-to-stay-soft pattern is deliberate and recorded here so it is
  reviewable design, not a buried trick — and so the next atom does not
  cargo-cult `rsplit` to smuggle a genuine hard dependency past D4. A
  first-class `MANIFEST.optional_requires` (or `soft:` register tag) would let
  the soft intent be *declared* rather than *inferred from a missing literal*;
  that is a substrate change, deferred.
- **Cross-child budget aggregation reaches `child.bus` directly** (the same
  child-session contract `sub_agent` relies on), because child `TurnEndEvent`s
  do not bubble to the parent bus. `spawn_child_session` is typed `-> Any`, so
  `.bus` is a duck-typed reach with no guaranteeing port; we access it directly
  (no defensive `getattr`/swallow) so a broken child contract fails loudly
  rather than silently zeroing the budget ceiling. If cross-child token
  observation is reused, promote it to a documented child-observation port.

## Decisions / rejected

- **LLM authors the script at runtime** (vs. only pre-written orchestrator
  atoms) — chosen; otherwise it is not *dynamic*.
- **JS isolate, in-process** (vs. Python in an `operations_agent_env` pod) —
  chosen; the pod over-provisions for a script whose only syscall is `agent()`
  and reintroduces RPC, defeating the single-process gateway.
- **Workflow-local journal** (vs. `SessionStore.open` per-call) — chosen;
  correct granularity, no substrate dependency.
- **No `Workflow` SDK type, no mode switch** — consistent with `agent_team`.

## Out of scope / flagged

- **Per-child env overrides.** `tool_eval_run` raises `NotImplementedError`
  because `AgentSessionConfig` has no per-child env field (the old `os.environ`
  route is racy under parallel runs). If a workflow needs per-worker env, that is
  the one change touching the substrate (`AgentSessionConfig` + route through
  `spawn_child_session`). Deferred until a real need appears.
- **`pipeline(items, ...stages)`** — deferred. v1 ships `parallel` only. A
  real no-barrier staged pipeline needs the script to express stages as JS
  callbacks the host invokes per item across the boundary; shipping it as a
  thin alias of `parallel` (no stage args, identical behaviour) would be a
  false affordance, so it is left out until the staged semantics are built
  properly. The concept (pipeline ≠ parallel: streaming vs. barrier) stays part
  of this design; only the v1 implementation is scoped to `parallel`.
- Cross-session (non-tree) resume, nested workflows beyond one level,
  binary artifacts — deferred.
