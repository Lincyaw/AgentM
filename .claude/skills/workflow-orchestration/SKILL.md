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

If a workflow script contains domain-specific string formatting, threshold
checks, or prompt construction, that logic belongs in a context atom inside
the agent unit.

## Architecture: three layers

```
Layer 1: Data preparation (full Python)
    CLI / eval harness — DB queries, file I/O, API calls
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
filesystem, no DB access. Collapsing Layer 2 into Layer 1 (orchestration
in the harness) loses the journal/resume/budget machinery and requires
manual session management.

## Agent units

An agent unit is a scenario folder that defines a single-purpose agent:

```
my_workflow/reviewer/
  manifest.yaml       ← atoms, system prompt, tool budget
  reviewer_context.py  ← reads atom_config, builds prompt, injects via BeforeAgentStartEvent
  reviewer_finalize.py ← registers the finalize tool (ToolTerminate)
```

### One manifest, one purpose

Never mix roles in one manifest. If a workflow needs agents with
different tools or different guidance, each role gets its own manifest.
A "reviewer" and a "synthesizer" are two manifests, even if they share
most atoms.

### Agent autonomy via atom_config

The workflow passes structured data; the agent builds its own prompt.
The workflow never knows what the prompt looks like — it only knows
what data the agent needs.

```python
# Workflow script — passes data, not prompts
result = await agent(
    "Review this document for quality issues.",
    scenario="my_workflow/reviewer",
    atom_config={
        "reviewer_context": {
            "document_path": doc_path,
            "review_criteria": criteria,
            "prior_reviews": prior,
        },
        "data_source": {"data_dir": data_dir},
    },
)
```

```python
# reviewer_context.py — owns all domain logic
def install(api, config):
    context = _build_review_prompt(
        document_path=config["document_path"],
        criteria=config["review_criteria"],
        prior_reviews=config.get("prior_reviews", []),
    )
    def before_start(event):
        event.system = f"{event.system}\n\n{context}"
    api.on(BeforeAgentStartEvent.CHANNEL, before_start)
```

This separation means:
- Workflow scripts stay readable (no multi-line prompt strings)
- Agent units evolve independently (change the prompt without touching the workflow)
- The same agent unit works in different workflows

### Finalize tools: the output contract

Every agent that returns structured data does so through a
**ToolTerminate finalize tool**. The workflow's `agent()` auto-parses
the JSON result — callers get a `dict`, not a string.

```python
# finalize atom
return ToolTerminate(
    result=ToolResult(content=[TextContent(
        type="text", text=review.model_dump_json(),
    )]),
    reason="review-submitted",
)
```

```python
# Workflow script — result is already a dict
result = await agent("Review this.", scenario="my_workflow/reviewer", atom_config={...})
if result.get("quality") == "pass":
    ...
```

`schema=` on `agent()` is a convenience shortcut: it synthesizes a
finalize tool from a JSON schema, for agents that don't have their own.
Both paths produce the same result type.

## Orchestration patterns

### Fan-out: same operation across many items

```python
phase("review")
results = await parallel([
    review_item(item) for item in args["items"]
])
passed = [r for r in results if r and r.get("quality") == "pass"]
```

### Pipeline: multi-stage per item, no barrier

```python
results = await pipeline(
    args["documents"],
    lambda doc: agent("Extract key claims.", scenario="extractor", atom_config={...}),
    lambda claims: agent("Verify each claim.", scenario="verifier", atom_config={...}),
)
```

Each item flows through all stages independently — item A can be at
stage 2 while item B is still at stage 1.

### Iterative expansion (BFS / graph walk)

```python
queue = list(args["seeds"])
visited = set(queue)
while queue:
    batch, queue = queue, []
    results = await parallel([explore(node) for node in batch])
    for node, result in zip(batch, results):
        for neighbor in result.get("discovered", []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### Multi-phase: produce then review

```python
phase("produce")
drafts = await parallel([draft(topic) for topic in args["topics"]])

phase("review")
reviewed = await parallel([
    agent("Review this draft.", scenario="reviewer", atom_config={...})
    for d in drafts if d
])
```

### Skip-to-phase (rerun a later stage on earlier results)

```python
if args.get("skip_to_review"):
    drafts = args["existing_drafts"]
else:
    phase("produce")
    drafts = await parallel([...])

phase("review")
# ... review drafts ...
```

## Programmatic invocation

External code (eval harnesses, CLI tools) runs workflows via the
`WorkflowRunner` service — no need to go through the tool interface:

```python
session = await AgentSession.create(config)
runner = session.get_service("workflow_runner")
result = await runner.run_file("path/to/workflow.py", args)
# result is a dict (auto-parsed from finalize tool) or str
```

## Single source of truth

The workflow's return value is the only output. Don't write intermediate
files that reshape the same data — consumers read the workflow output
directly.

```python
return {
    "results": results,
    "summary": summary,
    "metadata": {"rounds": round_n, "items_processed": len(items)},
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
| Write intermediate files alongside the main output | Return everything in the workflow's result dict |
| Hardcode domain thresholds in the workflow | Put policy in the agent's system prompt or context atom |
| Mix roles in one manifest | One manifest = one purpose; split roles into separate agent units |
| Use `schema=` when the agent already has a finalize tool | `schema=` is for agents without their own ToolTerminate tool |
| Keep only successful results, discard failures | Preserve all agent outputs — failure rationale matters for debugging and downstream decisions |
| Assume child agents inherit env vars or parent state | `atom_config` is the only reliable channel — pass every runtime-dependent config explicitly |
