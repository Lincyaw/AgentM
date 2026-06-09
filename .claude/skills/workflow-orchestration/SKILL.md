---
name: workflow-orchestration
description: >
  How to design, structure, and implement multi-agent workflows in AgentM.
  Covers the three-layer architecture (data prep → workflow script → agent
  units), agent autonomy via atom_config, finalize-tool output contract,
  and common patterns (BFS, fan-out, judge). Trigger when designing a new
  workflow, refactoring an existing multi-agent pipeline, deciding how to
  split work between workflow scripts and agent units, or when the user
  says 编排 / workflow / 多 agent / 拆分 / 调度. Also trigger when you
  catch yourself putting prompt-construction or domain logic into a
  workflow script — that's a sign the logic belongs in an agent unit's
  context atom.
---

# Workflow orchestration

## Core principle

A workflow script is **pure control flow**. It decides *which* agents
run, *when*, and *with what data*. It never decides *how* an agent
thinks — that's the agent unit's job.

If a workflow script contains domain-specific string formatting, threshold
checks, or prompt construction, that logic belongs in a context atom inside
the agent unit.

## Architecture: three layers

```
Layer 1: Data preparation (full Python)
    CLI / eval harness — DuckDB, file I/O, API calls
    → produces a JSON args dict

Layer 2: Workflow script (curated namespace)
    agent() / parallel() / pipeline() / phase() / log()
    → orchestrates agent units, passes structured data via atom_config
    → gets journal / resume / budget / concurrency for free

Layer 3: Agent units (scenario folders)
    manifest.yaml + context atom + finalize atom
    → each agent does one thing, builds its own prompt, returns structured output
```

### Why three layers, not two

Collapsing Layer 1 into Layer 2 (data prep inside the workflow script)
fails because the workflow namespace is sandboxed — no `import`, no
filesystem, no DuckDB. Collapsing Layer 2 into Layer 1 (orchestration
in the eval harness) loses the journal/resume/budget machinery and
requires manual session management.

## Agent units

An agent unit is a scenario folder that defines a single-purpose agent:

```
verifier/hop/
  manifest.yaml       ← atoms, system prompt, tool budget
  hop_context.py       ← reads atom_config, builds domain prompt, injects via BeforeAgentStartEvent
  hop_finalize.py      ← registers the finalize tool (ToolTerminate)
```

### Manifest = one agent, one purpose

Never mix roles in one manifest. If an agent needs different tools or
guidance for different jobs, that's two manifests.

### Agent autonomy via atom_config

The workflow passes structured data; the agent builds its own prompt:

```python
# Workflow script — pure data, no prompt knowledge
await agent(
    "Verify this propagation edge.",
    scenario="verifier/hop",
    atom_config={
        "hop_context": {
            "from_service": src,
            "to_service": tgt,
            "rel_type": rel,
            "fault_kind": "PodFailure",
            "all_faults": all_faults,
        },
        "duckdb_sql": {"data_dir": data_dir},
        "hop_finalize": {"data_dir": data_dir},
    },
)
```

```python
# hop_context.py — owns all domain logic
def install(api, config):
    context = _build_hop_prompt(
        from_service=config["from_service"],
        to_service=config["to_service"],
        ...
    )
    def before_start(event):
        event.system = f"{event.system}\n\n{context}"
    api.on(BeforeAgentStartEvent.CHANNEL, before_start)
```

This separation means:
- Workflow scripts stay readable (no 40-line prompt strings)
- Agent units evolve independently (change prompt without touching workflow)
- The same agent unit works in different workflows

### Finalize tools: the output contract

Every agent that returns structured data does so through a
**ToolTerminate finalize tool**. The workflow's `agent()` auto-parses
the JSON result — callers get a `dict`, not a string.

```python
# hop_finalize.py
return ToolTerminate(
    result=ToolResult(content=[TextContent(type="text", text=verdict.model_dump_json())]),
    reason="hop:verdict-submitted",
)
```

```python
# Workflow script — result is already a dict
result = await agent("Verify this edge.", scenario="verifier/hop", atom_config={...})
if result.get("verdict") == "confirmed":
    ...
```

`schema=` on `agent()` is a convenience shortcut: it synthesizes a
finalize tool from a JSON schema, for agents that don't have their own.
Both paths go through the same extraction — ToolTerminate result →
auto-parse JSON → dict.

## Workflow script patterns

### Fan-out (parallel hops)

```python
coros = []
for src, tgt, rel in pending_edges:
    coros.append(check_edge(src, tgt, rel))
results = await parallel(coros)
```

### BFS with parallel rounds

```python
queue = list(seeds)
while queue:
    batch, queue = queue, []
    results = await parallel([check(node) for node in batch])
    for node, result in zip(batch, results):
        if result and result.get("verdict") == "confirmed":
            queue.append(node)
```

### Two-phase: propagate then judge

```python
phase("propagate")
# ... BFS loop ...

phase("judge")
judge_result = await agent(
    "Review the propagation graph.",
    scenario="verifier/judge",
    atom_config={"judge_context": {...}},
)
```

### Skip-to-phase (rerun judge on existing data)

```python
skip_propagate = args.get("skip_propagate", False)
if skip_propagate:
    # Load existing results, jump to judge
    ...
```

## Single source of truth

The workflow's return value is the only output. Don't write intermediate
files (`all_verdicts.json`, `hop_results/`, etc.) that reshape the same
data — consumers read the workflow output directly.

```python
return {
    "confirmed_nodes": sorted(confirmed),
    "edges": edges,
    "node_evidence": node_evidence,   # evidence for ALL hops, confirmed + rejected
    "hop_log": hop_log,
    "rounds": round_n,
}
```

## What the workflow engine gives you

- **Journal** — crash mid-run, restart, completed `agent()` calls return
  cached results. Same prompt + same opts = cache hit.
- **Budget** — `budget.spent()` / `budget.remaining()` across all children.
- **Concurrency** — semaphore auto-limits parallel agents.
- **Progress** — `phase()` / `log()` surface in the TUI.
- **Auto-parse** — `agent()` returns `dict` for structured output,
  `str` for free-text. No `json.loads()` needed.

## Anti-patterns

| Don't | Do |
|-------|-----|
| Build prompts in the workflow script | Pass data via `atom_config`, let the context atom build the prompt |
| `json.loads(result)` on agent output | `agent()` auto-parses; check `isinstance(result, dict)` |
| Write intermediate files alongside the graph | Return everything in the workflow's result dict |
| Hardcode domain thresholds in the workflow | Put policy in the agent's system prompt or context atom |
| Use `load_module()` to import prompt builders | Merge prompt logic into the agent unit's context atom |
| Use `schema=` when the agent already has a finalize tool | `schema=` is for agents without their own ToolTerminate tool |
| Store evidence only for confirmed results | Store evidence for all hops — rejected rationale matters for debugging |
| Assume child agents inherit env vars or parent state | `atom_config` is the only reliable channel — pass every runtime-dependent config explicitly |
