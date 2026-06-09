---
name: workflow-orchestration
description: >
  How to design, structure, and implement multi-agent workflows in AgentM.
  Covers the three-layer architecture (data prep → workflow script → agent
  units), agent autonomy via atom_config, finalize-tool output contract,
  and common orchestration patterns. Trigger when designing a new workflow,
  refactoring an existing multi-agent pipeline, deciding how to split work
  between workflow scripts and agent units, or when the user says 编排 /
  workflow / 多 agent / 拆分 / 调度. Also trigger when you catch yourself
  putting prompt-construction or domain logic into a workflow script —
  that's a sign the logic belongs in an agent unit's context atom.
---

# Workflow orchestration

## Core principle

A workflow script is **pure control flow**. It decides *which* agents
run, *when*, and *with what data*. It never decides *how* an agent
thinks — that's the agent unit's job.

If you find domain-specific string formatting, threshold checks, or
prompt construction in a workflow script, move it into the agent unit's
context atom.

## Three-layer architecture

```
Layer 1: Data preparation (full Python)
    → DB queries, file I/O, API calls → JSON args dict

Layer 2: Workflow script (curated namespace)
    → agent() / parallel() / pipeline() / phase() / log()
    → passes structured data via atom_config
    → journal / resume / budget / concurrency for free

Layer 3: Agent units (scenario folders)
    → manifest.yaml + context atom + finalize atom
    → each agent does one thing, builds its own prompt
```

Layer 1 can't collapse into Layer 2 (sandboxed namespace — no imports,
no filesystem). Layer 2 can't collapse into Layer 1 (loses
journal/resume/budget, requires manual session management).

## Design decisions

When building a workflow, answer these questions in order:

**1. What are the roles?** Each distinct role becomes an agent unit
(its own manifest + atoms). If two tasks need different tools, different
system prompts, or different output schemas — they're different roles.

**2. What data does each role need?** That becomes `atom_config`. The
workflow passes structured data; a context atom inside the agent unit
reads it and builds the prompt. The workflow never constructs prompts.

**3. How does each role report results?** If structured output is
needed, the agent unit registers a ToolTerminate finalize tool. The
workflow's `agent()` auto-parses the result to `dict`. If no finalize
tool exists, `schema=` synthesizes one from a JSON schema. Both paths
produce the same result type.

**4. What's the control flow?** Choose a pattern:

| Pattern | When to use |
|---------|------------|
| Fan-out (`parallel`) | Same operation across N independent items |
| Pipeline (`pipeline`) | Multi-stage per item, no cross-item barrier |
| Iterative expansion | Each round discovers new work (graph walk, BFS) |
| Multi-phase | Sequential stages where later phases use earlier results |
| Skip-to-phase | Re-run a later stage on existing earlier results |

**5. What does the output look like?** The workflow's return value is
the single source of truth. No intermediate files reshaping the same
data.

For code examples of each pattern and the agent unit structure, read
`references/patterns.md`.

## Key contracts

- **One manifest = one purpose.** Never mix roles.
- **`atom_config` is the only reliable data channel to child agents.**
  Don't rely on env vars or parent state — pass everything explicitly,
  including config for third-party atoms in the manifest.
- **Preserve all agent outputs.** Keep failure rationale alongside
  successes — downstream decisions and debugging depend on both.
- **`agent()` auto-parses.** Returns `dict` for structured output
  (from any ToolTerminate finalize tool), `str` for free text. No
  `json.loads()` needed.
- **`WorkflowRunner` for programmatic invocation.** External code
  uses `session.get_service("workflow_runner")` instead of hacking
  into `session._tools`.

## What the engine gives you for free

- **Journal** — crash mid-run, restart, completed `agent()` calls
  return cached results
- **Budget** — `budget.spent()` / `budget.remaining()` across all
  children
- **Concurrency** — semaphore auto-limits parallel agents
- **Progress** — `phase()` / `log()` surface in the TUI
- **Auto-parse** — structured output → `dict`, free text → `str`

## Smells

These signal the design needs rethinking:

- **Prompt strings in the workflow script** → move to a context atom
- **`json.loads()` on agent output** → `agent()` already auto-parses
- **Intermediate files alongside the main output** → return everything
  in the workflow's result dict
- **Domain thresholds in the workflow** → policy belongs in the agent's
  atoms
- **Env var fallbacks for child agent config** → use `atom_config`
- **`schema=` on an agent that already has a finalize tool** → redundant
