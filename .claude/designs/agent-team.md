# Agent Team

Status: design (not yet implemented)
Owner: scenario layer + small extensions to `sub_agent`

Reaches into: `extensions/builtin/sub_agent.py` (additions), persona file
frontmatter convention, scenario manifest convention, references
`sub_agent_lifecycle` and `artifact_system` as load-bearing dependencies.

## Why team is not a SDK type

The first instinct on reading multi-agent literature is to introduce a
`Team` first-class type — members, hierarchy, protocols, shared memory,
budget — all bundled under one declaration. CrewAI does this, AutoGen does
this. We deliberately do not.

A `Team` type fuses orthogonal concerns into one abstraction:

- **Identity** — who the agent is (persona, system prompt, tools).
- **Lifecycle** — how a session is spawned and ended.
- **Memory** — what artifacts persist and who can read them.
- **Coordination** — who dispatches whom, who arbitrates conflicts.
- **Budget** — token and tool-call ceilings, scaling rules.

Bundled, every change to one concern ripples through the team type. Worse,
the bundling hides which concerns are actually load-bearing for a given
scenario; teams that only need lightweight delegation pay the full type
weight.

AgentM's `pluggable_architecture` and `extension_as_scenario` already
mandate that **features compose out of orthogonal primitives, not layered
abstractions**. A team is what you get when several primitives stand
together in one scenario. The runtime never knows.

This design therefore documents **the conventions and the small additions**
that make team formation safe, observable, and budget-aware — without
introducing a Team type.

## There is no mode switch

A second instinct is to draw a line between "single-agent mode" and
"team mode" — a boot tool, a runtime flag, a manifest toggle, anything that
flips the system from one regime to the other. We deliberately do not.

The right shape is identical to the one Claude Code uses: an agent always
has a set of capabilities; it decides when to use which. There is no
`enable_team_mode()` call before using `Agent({subagent_type: ...})`. The
Agent tool is always present; the model dispatches when it judges the task
warrants it.

Concretely, in a team-capable AgentM scenario:

- `dispatch_agent`, `artifact_write`, `artifact_read`, `wait_subagent`,
  `check_tasks` are all registered at session start.
- The orchestrator's system prompt may *encourage* delegation but does not
  require it.
- A simple lookup gets answered with a single `query_sql`; a deep
  investigation gets fanned out into N workers; both happen in the same
  session, no transition.
- Workers can recurse: a dispatched specialist may itself dispatch deeper
  specialists if its scenario inherits `sub_agent`.

The runtime never observes which "mode" the system is in, because there is
no mode. There are only:

- Capabilities the scenario manifest decided to register.
- An LLM that calls those capabilities when it judges the task warrants
  them.

This is the same architectural choice as `sub_agent_lifecycle`'s
"runtime guards the boundary, agent owns the decision" — generalised one
level up. The runtime guarantees that team-shaped capabilities (artifacts,
subagent dispatch) are *available*; the agent's prompt and reasoning
decide *whether and when* to use them.

### Implication for runtime extension loading

A separate question: should an agent be able to *acquire new capabilities
at runtime* (load `tool_git` on demand, install a new persona)? That is a
real and useful capability — `self_modifiable_architecture.md` already
specifies `api.reload_atom` with transactional install — but it is
**orthogonal to team formation**. Loading a new tool dynamically is
"acquiring a capability"; deciding to use `dispatch_agent` is "exercising
an already-loaded capability." Conflating them produces the false binary
this section rejects.

If a scenario wants the agent to grow team capabilities mid-flight (e.g.
start without `artifact_store`, install it once a long task warrants),
that goes through `reload_atom`. The pillars in this design assume the
opposite default for *team-capable* scenarios: load all four pillars
upfront, let the agent ignore them on simple tasks.

**Caveat — this is not a universal recommendation.** Each loaded
extension contributes tool schemas to every system prompt of every turn;
on a latency-sensitive single-purpose scenario (e.g. a one-shot
classifier), paying that overhead per call to support delegation that
will never happen is wasteful. The "load upfront" default applies to
scenarios that *expect* multi-step investigation and *might* delegate.
For scenarios where delegation is rare or never, the scenario author
should leave the team pillars out of the manifest entirely.

## The four pillars

A scenario is operating in "team mode" when all four hold:

| Pillar | Mechanism | Status |
|---|---|---|
| A. Lifecycle floor | `BeforeAgentEndEvent` + sub_agent handler | designed in `sub_agent_lifecycle.md` |
| B. Shared memory tier | `artifact_store` extension + sub_agent integration | designed in `artifact-system.md` |
| C. Brief contract | persona `input_schema` advisory + persona-body self-rejection | this doc |
| D. Budget awareness | persona `budget_defaults` + per-dispatch `budget` override | this doc |

Pillars A and B are full designs in their own right; this doc references
them. Pillars C and D are smaller additions that live entirely in the
sub_agent extension and the persona file format.

## Pillar A — Lifecycle floor (reference)

See `sub_agent_lifecycle.md`. Summary of the contract for a team scenario:

- `dispatch_agent` is async; returns task_id immediately.
- Parent's loop will not declare `agent_end` while completed-but-unread
  child findings exist; runtime injects a notification user-message and
  re-enters the loop.
- `check_tasks` blocks until at least one child changes state.
- `wait_subagent(task_id)` blocks on a specific child.
- The parent decides timing; the runtime guarantees no findings are lost.

Without this floor, a team scenario where the orchestrator forgets to poll
silently drops worker output. With it, the architecture is fault-tolerant
at the LLM-decision layer.

## Pillar B — Shared memory tier (reference)

See `artifact-system.md`. Summary:

- Workers write findings to a shared, session-tree-scoped artifact store.
- They return a short `<summary>` plus `<artifacts>` list of references in
  the notification block, not a 5K-character text dump.
- Orchestrator reads artifacts on demand via `artifact_read` /
  `artifact_grep` / `artifact_list`.
- Append-only with provenance — no atomic-update conflicts.

Without this tier, every worker output is a lossy compression and the
orchestrator's context grows linearly with worker count. With it, context
grows logarithmically and intermediate work is recoverable.

## Pillar C — Brief contract (persona-owned, not runtime-enforced)

### Problem

The literature's #1 lesson on multi-agent design (Anthropic): orchestrators
that give vague briefs cause subagents to duplicate work, leave gaps, or
misinterpret scope. Today, `dispatch_agent` accepts a freeform `prompt`
string. Brief quality is implicit, not contractual.

### Why this is *not* runtime-validated

The first design draft had the SDK grep the dispatched prompt for literal
`<objective>...</objective>` markers and reject the dispatch on miss. We
rejected this for three reasons:

1. **Trivially gameable.** An LLM facing a structured error will inject
   `<objective>tbd</objective>` to pass the check; the assertion is zero
   defense against actual brief vagueness.
2. **Runtime forces a brief format.** Hard-coding XML-tag detection into
   `sub_agent` would mean every scenario that wants Markdown sections,
   YAML, or JSON briefs has to reach into SDK code. That violates the
   SDK-vs-scenario separation we already maintain.
3. **Negative reinforcement loop.** Rejection-by-runtime trains the
   orchestrator to satisfy syntax, not to write better briefs.

### Mechanism (persona-owned)

Persona files declare what a good brief looks like — as **prose
expectations the persona's own system prompt enforces**, plus an optional
`input_schema` advisory the orchestrator can read for guidance:

```yaml
---
name: scout
description: First responder; produces multi-dimensional observability map.
tools: list_tables, query_sql
input_schema:
  required: [objective, scope_services, output_format]
  optional: [prior_findings, hypothesis_under_test]
artifact_kinds: [topology, finding, query_result]
---

<expected_brief>
Your dispatcher should provide:
- objective: the question you are answering, one sentence
- scope_services: backtick-quoted service names you should investigate
- output_format: artifact kinds expected (topology / finding / query_result)

If any field is missing or vacuous (e.g. "investigate everything"), your
FIRST action MUST be to write an artifact of kind="brief_rejection"
explaining what's missing and call agent_end. Do not investigate.
</expected_brief>

You are a Scout Agent — ...
```

The persona body, not the runtime, decides what to do when a brief is
inadequate. The orchestrator sees `brief_rejection` artifacts in the
notification block and adapts the next dispatch.

`input_schema` survives in the frontmatter as **advisory metadata**. The
orchestrator's prompt may reference it ("the persona declares
`required: [objective, scope_services, output_format]`; include all three
in your brief"), and tooling can surface it in `<available_agents>`
listings, but the runtime never validates it.

### Observability (the only runtime piece)

`dispatch_agent` records the dispatched prompt's length and the
`subagent_type` to trajectory. Combined with the persona's
`brief_rejection` artifacts, post-hoc analysis can compute "fraction of
dispatches that produced rejections" per scenario — the actual signal of
brief-quality drift, without any false-positive rejection-loop risk.

### What this buys

- No SDK-level brief format lock-in; scenarios can choose XML, Markdown,
  JSON, or none.
- Brief-quality enforcement lives where it can actually reason about
  *content* (the persona's LLM), not where it can only pattern-match.
- Failure mode "subagent misinterpreted scope" (MAST-categorized) becomes
  detectable via `brief_rejection` artifacts.

### What it does not enforce

- Output format compliance. The persona body remains responsible.
- That the persona body actually implements the `<expected_brief>`
  contract — that's authoring discipline, same as any system prompt.

## Pillar D — Budget awareness

### Problem

Anthropic's analysis: multi-agent systems use 15× the tokens of single-agent
chats; token usage explains 80% of performance variance. Without an
explicit budget concept, scenarios over-spend on simple queries and
under-spend on complex ones. The literature's recommendation (Anthropic):
"Simple fact-finding requires one agent with 3–10 tool calls; direct
comparisons need 2–4 subagents with 10–15 calls each; complex research
might use over 10 subagents."

Today, every dispatched worker inherits the same `LoopConfig.max_turns =
32` regardless of task complexity.

### Mechanism

Three layers, **all opt-in, no scenario-wide complexity tier**:

A scenario-level `task_complexity` knob was considered and rejected: a
single rca scenario routinely handles both "list tables" (cheap lookup)
and "find root cause" (deep fan-out). Pinning a complexity at the
manifest level mis-locates the decision — complexity is a property of
the *query*, not the scenario.

**Layer 1 — persona `budget_defaults`**

Persona frontmatter declares per-task budget defaults:

```yaml
budget_defaults:
  max_tool_calls: 15
  max_turns: 10
```

These apply to every dispatch of that persona unless the orchestrator
overrides at call time. A `verify` persona might default to 5 tool
calls; a `deep_analyze` persona to 25. Default-safe per role.

**Layer 2 — per-dispatch budget on the tool call**

`dispatch_agent` accepts an optional `budget` object:

```json
{
  "subagent_type": "scout",
  "purpose": "...",
  "prompt": "...",
  "budget": {"max_tool_calls": 15, "max_turns": 10}
}
```

When set, overrides persona defaults for that dispatch only. The
orchestrator scales depth per query: a quick scout pass gets
`max_tool_calls: 6`; a forensic deep-dive gets `30`.

**Layer 3 — `sub_agent.max_workers` (already exists)**

Caps parallel concurrency at the manifest level. Already in production.

The composition: `max_workers` bounds parallelism, `budget_defaults`
bounds per-persona depth, per-dispatch `budget` lets the orchestrator
adjust per query. No fourth scenario-wide knob is needed; adding one
would mis-locate complexity at the manifest tier.

### What this buys

- Default-safe: a scenario author who declares `task_complexity: simple`
  cannot accidentally have their orchestrator fan out 10 workers.
- Observability: trajectory records the budget that was applied,
  per-dispatch, alongside actual usage. Post-hoc analysis (which scenarios
  are over- or under-spending) becomes mechanical.
- Cost control: a single line in a manifest is the difference between a
  $0.05 query and a $0.75 query.

### What it does not enforce

- Token budgets. Tool-call ceiling is a proxy. Real token enforcement
  belongs in `cost_budget` (existing Tier-2 atom). The two compose: a
  worker can be both tool-call-capped and dollar-capped.
- Adaptive scaling. We do not auto-detect "this query needs deeper
  research" and bump the budget. The orchestrator may dispatch with a
  larger explicit `budget`; the cap is still bounded by `team_profile`.

## Failure-mode mapping

Each pillar defends against named failure modes from the literature.
This is the explicit checklist that justifies the design.

| MAST / literature failure mode | Pillar(s) | Defense |
|---|---|---|
| Lost child output (silent drop on agent_end) | A | Lifecycle floor injects all completed children before exit |
| Polling burns parent budget | A | Blocking `check_tasks` |
| Context loss at handoff (lossy summarization) | B | Artifact store + reference-based handoff |
| Supervisor context bloat | B | Notification block is summary + ids only |
| Inconsistent shared state | B | Append-only model with provenance |
| Subagent misinterpreted scope | C | Persona body self-rejects vacuous briefs via `brief_rejection` artifact |
| Duplicate work between subagents | C | Persona's `<expected_brief>` requires `scope_services` and refuses without it |
| Token over-spend on simple queries | D | Persona `budget_defaults` keep cheap personas cheap; per-dispatch override |
| Token under-spend on complex queries | D | Per-dispatch `budget` override scales the right query, not the whole scenario |
| Cascade context poisoning | A + B | Each worker has private tier-1; tier-2 immutable |
| Open mesh routing (no clear authority) | scenario layout | Orchestrator-only dispatch is the default convention |

## Three usage modes, one mechanism

Same as `sub_agent_lifecycle` (no new modes). The pillars are orthogonal
to mode; any of fire-and-forget, explicit wait, steered fan-out works
identically whether artifacts are written or not, whether budgets are
declared or not.

## Persona file format (consolidated)

After this design, a persona file looks like:

```markdown
---
name: scout
description: First responder; produces multi-dimensional observability map.
tools: list_tables, query_sql
input_schema:
  required: [objective, scope_services, output_format]
  optional: [prior_findings, hypothesis_under_test, budget_tool_calls]
artifact_kinds: [topology, finding, query_result]
budget_defaults:
  max_tool_calls: 15
  max_turns: 10
---

You are a Scout Agent — first responder in a root cause analysis.
... (rest of persona body) ...
```

All new fields are optional. Personas without them behave as today.

## Scenario manifest (consolidated)

```yaml
name: rca
description: Root-cause analysis over parquet observability data.
extensions:
  - module: agentm.extensions.builtin.artifact_store
  - local: orchestrator_setup
  - module: agentm.extensions.builtin.sub_agent
    config:
      max_workers: 4
      inherit_extensions: [duckdb_sql, artifact_store, trajectory]
      available_inherited_extensions:
        duckdb_sql: { module: agentm_rca.tools.duckdb_sql, config: {...} }
        artifact_store: { module: agentm.extensions.builtin.artifact_store, config: {} }
        trajectory: { module: agentm.extensions.builtin.trajectory, config: {path: rca_worker_trajectory.jsonl} }
  - module: agentm_rca.tools.duckdb_sql
    config: {...}
  - module: agentm.extensions.builtin.trajectory
    config: { path: rca_trajectory.jsonl }
```

No new top-level manifest keys. The four pillars are pure composition of
existing primitives plus per-persona / per-dispatch fields.

## What is not in scope

- **First-class Team type** — see opening rationale.
- **Peer-to-peer dispatch** — workers spawning workers without going
  through the orchestrator. Technically possible (each session has its
  own `sub_agent`) but discouraged for default scenarios; the literature
  is consistent that flat mesh communication fails in production.
- **Built-in voting / arbitration** — when an orchestrator wants debate,
  it dispatches N opposing personas and arbitrates in its own prompt.
  Not a runtime concern.
- **Cross-scenario persona registry** — personas live in
  `scenarios/<name>/agents/`. Sharing across scenarios is a filesystem /
  git concern (symlinks, submodules), not a SDK feature.
- **Output schema validation** — runtime cannot enforce the worker's
  final text matches a schema. Persona prompt body is responsible.
  Future work could add this; tracked in `artifact-system.md` open
  questions.
- **Adaptive budget escalation** — orchestrator can request more budget
  per-dispatch but cannot exceed `team_profile` defaults without
  changing the manifest. No runtime "promote complexity" path.

## Compatibility and migration

- All four pillars are opt-in. A scenario that uses `dispatch_agent`
  today (rca) continues to work without `artifact_store`,
  `team_profile`, or `input_schema`.
- Adding `input_schema: required: []` (empty) to a persona is a no-op.
- The `team_profile` key is read by an optional builtin; absent =
  no defaults applied.
- New persona frontmatter fields are additive; old personas valid.
- `dispatch_agent` `budget` argument is optional.

## Effort estimate

This doc on top of `sub_agent_lifecycle.md` and `artifact-system.md`:

- Persona `input_schema` advisory parsing + surface in `<available_agents>`: ~15 lines
- Persona `budget_defaults` parsing + apply to child LoopConfig: ~20 lines
- `dispatch_agent` optional `budget` argument override: ~20 lines
- rca scenario migration (add `<expected_brief>` blocks to personas,
  switch hypothesis tools to artifact_write): ~40 lines

Combined with the dependent designs:

- `sub_agent_lifecycle`: 100–150 lines
- `artifact_system`: 250–300 lines
- This design's additions: ~95 lines

Grand total for full team-mode capability: roughly 460–550 lines of
code across three coordinated PRs. No data migration. Each PR ships
independently and provides incremental value:

1. `sub_agent_lifecycle` first (no team needed; benefits any
   sub-agent dispatch scenario).
2. `artifact_system` second (depends on lifecycle for clean
   integration; benefits scenarios with multi-step worker pipelines).
3. This doc's additions third (refines the API, adds defaults,
   migrates rca to the new shape).
